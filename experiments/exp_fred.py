"""
Experiment 2: Monthly FRED Macroeconomic Data (McCracken-Ng subset, 12 series).

Improvements over prior version:
  - Monthly frequency: ~800 obs vs. 263 quarterly
  - Proper 3-way split: 70% train / 10% val / 20% test (no lambda leakage)
  - Additional baselines: Random Walk, VAR(BIC), LSTM-L2
  - Lambda selected on val set for Causal and L2 models
  - Diebold-Mariano (HLN 1997) statistical tests vs LSTM Baseline
  - Block-bootstrap 95% confidence intervals on MSE

Run from project root:
    python -m experiments.exp_fred
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_fred_monthly, TimeSeriesDataset, fit_scaler
from src.causal_discovery import run_pcmci, count_links
from src.models import BaselineLSTM, CausalLSTM, MaskedLSTM
from src.train import fit, predict_one_step
from src.evaluate import (compute_metrics, compute_metrics_per_var,
                           diebold_mariano_test, block_bootstrap_ci)
from src.baselines import VARBaseline, RandomWalkBaseline
from src.plotting import plot_predictions, plot_causal_graph_heatmap, plot_training_history

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW      = 12    # 1 year look-back (monthly)
BATCH       = 32
EPOCHS      = 150
LR          = 5e-4
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.10  # test = remaining 20%
HIDDEN      = 64    # larger network for 12-var monthly data
TAU_MAX     = 6     # 6-month maximum lag for PCMCI

LAMBDA_CAUSAL = [0.001, 0.01, 0.1, 1.0]
LAMBDA_L2     = [1e-5, 1e-4, 1e-3, 1e-2]

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_loaders(norm_data, window, batch):
    ds = TimeSeriesDataset(norm_data, window)
    return DataLoader(ds, batch_size=batch, shuffle=False)


def eval_mse(model, loader, device):
    model.eval()
    mse_fn = nn.MSELoss()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if isinstance(model, CausalLSTM):
                yh, _ = model(xb)
            else:
                yh = model(xb)
            total += mse_fn(yh, yb).item()
    return total / len(loader)


def best_lambda_causal(N, hidden, links, window, train_dl, val_dl, grid, seed):
    best_lam, best_val = grid[0], float("inf")
    for lam in grid:
        torch.manual_seed(seed)
        m = CausalLSTM(N, hidden, links, window, lambda_reg=lam,
                       num_layers=2, dropout=0.1).to(DEVICE)
        fit(m, train_dl, val_dl, epochs=EPOCHS // 3, lr=LR,
            device=DEVICE, verbose=False, patience=10)
        v = eval_mse(m, val_dl, DEVICE)
        if v < best_val:
            best_val, best_lam = v, lam
        print(f"    lambda_causal={lam:.4f} -> val_MSE={v:.4e}")
    return best_lam


def best_lambda_l2(N, hidden, window, train_dl, val_dl, grid, seed):
    best_lam, best_val = grid[0], float("inf")
    for lam in grid:
        torch.manual_seed(seed)
        m = BaselineLSTM(N, hidden, num_layers=2, dropout=0.1).to(DEVICE)
        fit(m, train_dl, val_dl, epochs=EPOCHS // 3, lr=LR, weight_decay=lam,
            device=DEVICE, verbose=False, patience=10)
        v = eval_mse(m, val_dl, DEVICE)
        if v < best_val:
            best_val, best_lam = v, lam
        print(f"    lambda_l2={lam:.1e} -> val_MSE={v:.4e}")
    return best_lam


def dm_summary(e_base, e_model, name):
    dm_stat, pval = diebold_mariano_test(e_base, e_model)
    sig = "**" if pval < 0.05 else ("*" if pval < 0.10 else "ns")
    direction = "better" if dm_stat > 0 else "worse"
    print(f"  DM vs Baseline [{name}]: stat={dm_stat:+.3f}  p={pval:.3f}  {sig}  ({direction})")
    return dm_stat, pval


def main():
    # ── 1. Load monthly FRED data ─────────────────────────────────────────────
    print("Downloading monthly FRED data...")
    try:
        raw_df, data_df = load_fred_monthly(start="1960-01-01")
    except Exception as e:
        print(f"Full download failed ({e}), retrying with core series...")
        from src.data import FRED_MONTHLY
        core = {k: v for k, v in FRED_MONTHLY.items()
                if k in ("INDPRO", "PAYEMS", "UNRATE", "HOUST",
                         "CPIAUCSL", "FEDFUNDS", "GS10", "M2SL")}
        from src.data import load_fred_monthly as lf
        raw_df, data_df = lf(series=core, start="1960-01-01")

    var_names = list(data_df.columns)
    N = len(var_names)
    data = data_df.values
    T_total = len(data)
    print(f"Data shape: {data.shape}  |  Variables: {var_names}")
    print(f"Date range: {data_df.index[0].date()} -> {data_df.index[-1].date()}")

    # ── 2. 3-way chronological split ─────────────────────────────────────────
    n_train = int(T_total * TRAIN_FRAC)
    n_val   = int(T_total * VAL_FRAC)
    n_test  = T_total - n_train - n_val

    train_raw = data[:n_train]
    # val and test sets prepend WINDOW rows from their preceding period
    val_raw   = data[n_train - WINDOW: n_train + n_val]
    test_raw  = data[n_train + n_val - WINDOW:]

    scaler = fit_scaler(train_raw)
    train_norm = scaler.transform(train_raw)
    val_norm   = scaler.transform(val_raw)
    test_norm  = scaler.transform(test_raw)

    T_val_eff  = len(val_norm)  - WINDOW
    T_test_eff = len(test_norm) - WINDOW
    print(f"Train: {len(train_norm)}  Val: {T_val_eff}  Test: {T_test_eff} steps")

    # ── 3. PCMCI on training data ─────────────────────────────────────────────
    print(f"\nRunning PCMCI (tau_max={TAU_MAX})...")
    pcmci_result = run_pcmci(
        train_norm, tau_max=TAU_MAX, alpha=0.05, pc_alpha=0.1,
        test="parcorr", var_names=var_names,
    )
    est_links = pcmci_result["links"]
    n_sig = count_links(est_links)
    print(f"Significant links found: {n_sig}")
    for j, lst in est_links.items():
        for i, lag in lst:
            print(f"  {var_names[i]} (lag {lag}) -> {var_names[j]}")

    plot_causal_graph_heatmap(
        est_links, var_names, tau_max=TAU_MAX,
        title="PCMCI Causal Graph — FRED Monthly",
        save_path=os.path.join(FIGURES_DIR, "causal_graph_fred.png"),
    )

    # ── 4. DataLoaders ────────────────────────────────────────────────────────
    train_dl = make_loaders(train_norm, WINDOW, BATCH)
    val_dl   = make_loaders(val_norm,   WINDOW, BATCH)
    test_dl  = make_loaders(test_norm,  WINDOW, BATCH)

    all_preds  = {}
    all_errors = {}

    # ── 5. Random Walk baseline ───────────────────────────────────────────────
    print("\n--- Random Walk baseline ---")
    rw = RandomWalkBaseline()
    preds_rw_norm = rw.predict(test_norm, WINDOW)
    y_true_norm   = test_norm[WINDOW:]           # (T_test, N) in normalized scale
    preds_rw_orig = scaler.inverse_transform(preds_rw_norm)
    y_true_orig   = scaler.inverse_transform(y_true_norm)
    m_rw = compute_metrics(y_true_orig, preds_rw_orig)
    all_preds["Random Walk"] = preds_rw_orig
    print(f"  RW -> MSE={m_rw['mse']:.4e}  MAE={m_rw['mae']:.4e}")

    # ── 6. VAR baseline ───────────────────────────────────────────────────────
    print("\n--- VAR baseline (BIC lag selection) ---")
    var_mdl = VARBaseline(maxlags=6, ic="bic")
    var_mdl.fit(train_norm)
    print(f"  VAR selected lag order: {var_mdl.lag_order}")
    # predict over val+test period; we only need test portion
    full_history = data[n_train - WINDOW: n_train + n_val]   # raw = val
    preds_var_all_norm = var_mdl.predict(
        train_norm,
        np.vstack([val_norm, test_norm[WINDOW:]])   # continuation after val
    )
    # The predict() method returns T_test predictions for all rows passed after train
    # Recompute cleanly: refit on train, predict only test rows
    var_mdl2 = VARBaseline(maxlags=6, ic="bic")
    var_mdl2.fit(train_norm)
    # Build history up to test start: train + val actual values
    val_actuals_norm = val_norm[WINDOW:]           # (T_val_eff, N) actual val values
    history_to_test  = np.vstack([train_norm, val_actuals_norm])
    # Now predict test
    preds_var_test = np.zeros((T_test_eff, N))
    history = history_to_test.copy()
    lag = var_mdl2.lag_order
    for t in range(T_test_eff):
        fc = var_mdl2._fitted.forecast(history[-lag:], steps=1)
        preds_var_test[t] = fc[0]
        history = np.vstack([history, y_true_norm[t: t + 1]])
    preds_var_orig = scaler.inverse_transform(preds_var_test)
    m_var = compute_metrics(y_true_orig, preds_var_orig)
    all_preds["VAR"] = preds_var_orig
    print(f"  VAR -> MSE={m_var['mse']:.4e}  MAE={m_var['mae']:.4e}")

    # ── 7. LSTM Baseline ──────────────────────────────────────────────────────
    print("\n--- LSTM Baseline ---")
    torch.manual_seed(SEED)
    baseline = BaselineLSTM(N, HIDDEN, num_layers=2, dropout=0.1).to(DEVICE)
    h_base = fit(baseline, train_dl, val_dl, epochs=EPOCHS, lr=LR,
                 device=DEVICE, patience=20)
    preds_base_norm, _ = predict_one_step(baseline, test_dl, DEVICE)
    preds_base_orig    = scaler.inverse_transform(preds_base_norm)
    m_base = compute_metrics(y_true_orig, preds_base_orig)
    all_preds["LSTM Baseline"] = preds_base_orig
    e_base = (y_true_orig - preds_base_orig)
    print(f"  Baseline -> MSE={m_base['mse']:.4e}  MAE={m_base['mae']:.4e}")
    plot_training_history(h_base, "LSTM Baseline — FRED Monthly",
                          os.path.join(FIGURES_DIR, "history_baseline_fred.png"))

    # ── 8. LSTM-L2 (weight decay) ─────────────────────────────────────────────
    print("\n--- LSTM-L2 (lambda selected on val) ---")
    best_lam_l2 = best_lambda_l2(N, HIDDEN, WINDOW, train_dl, val_dl, LAMBDA_L2, SEED)
    print(f"  Best L2 lambda = {best_lam_l2:.1e}")
    torch.manual_seed(SEED)
    lstm_l2 = BaselineLSTM(N, HIDDEN, num_layers=2, dropout=0.1).to(DEVICE)
    h_l2 = fit(lstm_l2, train_dl, val_dl, epochs=EPOCHS, lr=LR, weight_decay=best_lam_l2,
               device=DEVICE, patience=20)
    preds_l2_norm, _ = predict_one_step(lstm_l2, test_dl, DEVICE)
    preds_l2_orig    = scaler.inverse_transform(preds_l2_norm)
    m_l2 = compute_metrics(y_true_orig, preds_l2_orig)
    all_preds["LSTM-L2"] = preds_l2_orig
    print(f"  LSTM-L2 -> MSE={m_l2['mse']:.4e}  MAE={m_l2['mae']:.4e}")

    # ── 9. Causal LSTM (PCMCI prior, lambda on val) ───────────────────────────
    print("\n--- Causal LSTM (PCMCI prior, lambda search) ---")
    best_lam_c = best_lambda_causal(N, HIDDEN, est_links, WINDOW,
                                     train_dl, val_dl, LAMBDA_CAUSAL, SEED)
    print(f"  Best causal lambda = {best_lam_c}")
    torch.manual_seed(SEED)
    causal = CausalLSTM(N, HIDDEN, est_links, WINDOW, lambda_reg=best_lam_c,
                        num_layers=2, dropout=0.1).to(DEVICE)
    h_causal = fit(causal, train_dl, val_dl, epochs=EPOCHS, lr=LR,
                   device=DEVICE, patience=20)
    preds_causal_norm, _ = predict_one_step(causal, test_dl, DEVICE)
    preds_causal_orig    = scaler.inverse_transform(preds_causal_norm)
    m_causal = compute_metrics(y_true_orig, preds_causal_orig)
    all_preds[f"LSTM Causal (lam={best_lam_c})"] = preds_causal_orig
    print(f"  Causal -> MSE={m_causal['mse']:.4e}  MAE={m_causal['mae']:.4e}")
    plot_training_history(h_causal, f"LSTM Causal — FRED Monthly (lam={best_lam_c})",
                          os.path.join(FIGURES_DIR, "history_causal_fred.png"))

    # ── 10. Causal LSTM (Random graph ablation) ───────────────────────────────
    print("\n--- Causal LSTM (Random graph ablation) ---")
    rng_np = np.random.default_rng(SEED + 99)
    rand_links = {j: [] for j in range(N)}
    for j, lst in est_links.items():
        for _ in lst:
            rand_links[j].append((int(rng_np.integers(0, N)),
                                   int(rng_np.integers(1, TAU_MAX + 1))))
    torch.manual_seed(SEED)
    causal_rand = CausalLSTM(N, HIDDEN, rand_links, WINDOW, lambda_reg=best_lam_c,
                              num_layers=2, dropout=0.1).to(DEVICE)
    fit(causal_rand, train_dl, val_dl, epochs=EPOCHS, lr=LR,
        device=DEVICE, verbose=False, patience=20)
    preds_rand_norm, _ = predict_one_step(causal_rand, test_dl, DEVICE)
    preds_rand_orig    = scaler.inverse_transform(preds_rand_norm)
    m_rand = compute_metrics(y_true_orig, preds_rand_orig)
    all_preds["LSTM Causal (Random)"] = preds_rand_orig
    print(f"  Random -> MSE={m_rand['mse']:.4e}  MAE={m_rand['mae']:.4e}")

    # ── 11. Masked LSTM (PCMCI hard mask, no lambda) ─────────────────────────
    print("\n--- Masked LSTM (PCMCI hard mask, no lambda) ---")
    torch.manual_seed(SEED)
    masked = MaskedLSTM(N, HIDDEN, est_links, WINDOW, num_layers=2, dropout=0.1).to(DEVICE)
    h_masked = fit(masked, train_dl, val_dl, epochs=EPOCHS, lr=LR,
                   device=DEVICE, patience=20)
    preds_masked_norm, _ = predict_one_step(masked, test_dl, DEVICE)
    preds_masked_orig    = scaler.inverse_transform(preds_masked_norm)
    m_masked = compute_metrics(y_true_orig, preds_masked_orig)
    all_preds["LSTM Masked (PCMCI)"] = preds_masked_orig
    print(f"  Masked PCMCI -> MSE={m_masked['mse']:.4e}  MAE={m_masked['mae']:.4e}")

    # ── 11b. Masked LSTM (Random graph ablation) ──────────────────────────────
    print("\n--- Masked LSTM (Random graph ablation) ---")
    torch.manual_seed(SEED)
    masked_rand = MaskedLSTM(N, HIDDEN, rand_links, WINDOW, num_layers=2, dropout=0.1).to(DEVICE)
    fit(masked_rand, train_dl, val_dl, epochs=EPOCHS, lr=LR,
        device=DEVICE, verbose=False, patience=20)
    preds_mrand_norm, _ = predict_one_step(masked_rand, test_dl, DEVICE)
    preds_mrand_orig    = scaler.inverse_transform(preds_mrand_norm)
    m_mrand = compute_metrics(y_true_orig, preds_mrand_orig)
    all_preds["LSTM Masked (Random)"] = preds_mrand_orig
    print(f"  Masked Random -> MSE={m_mrand['mse']:.4e}  MAE={m_mrand['mae']:.4e}")

    # ── 12. Diebold-Mariano tests (all vs LSTM Baseline) ─────────────────────
    print("\n--- Diebold-Mariano tests vs LSTM Baseline ---")
    dm_results = {}
    for mname, mpreds in all_preds.items():
        if mname == "LSTM Baseline":
            continue
        e2 = y_true_orig - mpreds
        dm_stat, pval = dm_summary(e_base, e2, mname)
        dm_results[mname] = {"dm_stat": dm_stat, "p_value": pval}

    # ── 12. Block-bootstrap 95% CIs on MSE ────────────────────────────────────
    print("\n--- 95% CI on MSE (block bootstrap, block=12) ---")
    ci_results = {}
    for mname, mpreds in all_preds.items():
        sq_errors = ((y_true_orig - mpreds) ** 2).mean(axis=1)  # (T_test,) per-step MSE
        lo, hi = block_bootstrap_ci(sq_errors, block_size=12, n_bootstrap=500, seed=SEED)
        ci_results[mname] = {"lower": lo, "upper": hi}
        print(f"  {mname:<35} MSE 95% CI = [{lo:.4e}, {hi:.4e}]")

    # ── 13. Per-variable metrics ──────────────────────────────────────────────
    print("\n--- Per-variable MAE (Baseline vs Causal PCMCI vs Masked PCMCI) ---")
    pv_base   = compute_metrics_per_var(y_true_orig, preds_base_orig, var_names)
    pv_causal = compute_metrics_per_var(y_true_orig, preds_causal_orig, var_names)
    pv_masked = compute_metrics_per_var(y_true_orig, preds_masked_orig, var_names)
    print(f"  {'Variable':<35} {'Base MAE':>10} {'Causal MAE':>12} {'Masked MAE':>12} {'Causal%':>8} {'Masked%':>8}")
    print("  " + "-" * 89)
    for vn in var_names:
        b  = pv_base[vn]["mae"]
        c  = pv_causal[vn]["mae"]
        mk = pv_masked[vn]["mae"]
        ic = (b - c)  / b * 100
        im = (b - mk) / b * 100
        print(f"  {vn:<35} {b:>10.4f} {c:>12.4f} {mk:>12.4f} {ic:>+7.1f}% {im:>+7.1f}%")

    # ── 14. Plots ─────────────────────────────────────────────────────────────
    plot_predictions(
        y_true_orig,
        {"LSTM Baseline": preds_base_orig,
         f"LSTM Causal (lam={best_lam_c})": preds_causal_orig,
         "VAR": preds_var_orig},
        var_names=var_names,
        title_prefix="FRED Monthly — ",
        save_path=os.path.join(FIGURES_DIR, "predictions_fred.png"),
        max_vars=6,
    )

    # Lambda sensitivity plot
    print("\nRunning full lambda sensitivity (for figure)...")
    lam_mse = {}
    for lam in LAMBDA_CAUSAL:
        torch.manual_seed(SEED)
        m_tmp = CausalLSTM(N, HIDDEN, est_links, WINDOW, lambda_reg=lam,
                           num_layers=2, dropout=0.1).to(DEVICE)
        fit(m_tmp, train_dl, val_dl, epochs=EPOCHS // 3, lr=LR,
            device=DEVICE, verbose=False, patience=10)
        p_tmp, _ = predict_one_step(m_tmp, test_dl, DEVICE)
        p_tmp_orig = scaler.inverse_transform(p_tmp)
        lam_mse[lam] = compute_metrics(y_true_orig, p_tmp_orig)["mse"]
        print(f"  lambda={lam:.3f} -> test_MSE={lam_mse[lam]:.4e}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(list(lam_mse.keys()), list(lam_mse.values()), "o-",
                color="red", label="LSTM Causal")
    ax.axhline(m_base["mse"], color="steelblue", linestyle="--", label="LSTM Baseline")
    ax.set_xlabel("lambda (causal regularization strength)")
    ax.set_ylabel("MSE (test set)")
    ax.set_title("Sensitivity to lambda — FRED Monthly")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "lambda_sensitivity_fred.png"), dpi=150)
    plt.close()

    # ── 15. Summary ───────────────────────────────────────────────────────────
    all_metrics = {
        "Random Walk": m_rw,
        "VAR": m_var,
        "LSTM Baseline": m_base,
        "LSTM-L2": m_l2,
        f"LSTM Causal (lam={best_lam_c})": m_causal,
        "LSTM Causal (Random)": m_rand,
        "LSTM Masked (PCMCI)": m_masked,
        "LSTM Masked (Random)": m_mrand,
    }

    print("\n" + "=" * 70)
    print("FINAL SUMMARY — FRED MONTHLY EXPERIMENT")
    print("=" * 70)
    print(f"  {'Model':<35} {'MSE':>12} {'RMSE':>12} {'MAE':>12}")
    print("  " + "-" * 75)
    for mname, m in all_metrics.items():
        sig_str = ""
        if mname in dm_results:
            pv = dm_results[mname]["p_value"]
            sig_str = "**" if pv < 0.05 else ("*" if pv < 0.10 else "")
        print(f"  {mname:<35} {m['mse']:>12.4e} {m['rmse']:>12.4e} {m['mae']:>12.4e} {sig_str}")

    # ── 16. Save all CSVs ─────────────────────────────────────────────────────
    with open(os.path.join(RESULTS_DIR, "fred_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mse", "rmse", "mae",
                         "ci_lower", "ci_upper", "dm_stat", "p_value"])
        for mname, m in all_metrics.items():
            ci = ci_results.get(mname, {})
            dm = dm_results.get(mname, {})
            writer.writerow([mname, m["mse"], m["rmse"], m["mae"],
                             ci.get("lower", ""), ci.get("upper", ""),
                             dm.get("dm_stat", ""), dm.get("p_value", "")])

    with open(os.path.join(RESULTS_DIR, "fred_sensitivity.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda", "test_mse"])
        for lam, mse in lam_mse.items():
            writer.writerow([lam, mse])

    with open(os.path.join(RESULTS_DIR, "fred_per_var.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "base_mae", "causal_mae", "masked_mae",
                         "causal_improvement_pct", "masked_improvement_pct"])
        for vn in var_names:
            b  = pv_base[vn]["mae"]
            c  = pv_causal[vn]["mae"]
            mk = pv_masked[vn]["mae"]
            writer.writerow([vn, b, c, mk,
                             (b - c) / b * 100,
                             (b - mk) / b * 100])

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
