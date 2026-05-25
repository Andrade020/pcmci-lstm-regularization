"""
Experiment 1: Synthetic VAR systems.

Three scenarios:
  1. 4-variable linear VAR (multi-lag)
  2. 8-variable linear VAR(1)
  3. 4-variable threshold VAR (nonlinear, same causal structure as Scenario 1)

Key improvements vs. prior version:
  - lambda selected by validation for each scenario (not fixed at 0.01)
  - 3-way split: 60% train / 20% val / 20% test
  - Ablation: Baseline, Causal(PCMCI), Causal(True graph), Causal(Random graph)

Run from project root:
    python -m experiments.exp_synthetic
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import csv
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")

from src.data import generate_var_series, TimeSeriesDataset, fit_scaler
from src.causal_discovery import run_pcmci, links_from_matrix, compare_links
from src.models import BaselineLSTM, CausalLSTM, MaskedLSTM
from src.train import fit, predict_one_step
from src.evaluate import compute_metrics, diebold_mariano_test
from src.plotting import plot_predictions, plot_causal_graph_heatmap, plot_training_history

SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW     = 5
BATCH      = 64
EPOCHS     = 120
LR         = 1e-3
T          = 2000
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20   # test = remaining 20%

LAMBDA_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Scenario 1: 4-variable linear VAR (multi-lag) ────────────────────────────
def generate_4var(T: int, noise_std: float = 0.5, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.zeros((T, 4))
    for t in range(3, T):
        X[t, 0] = 0.8 * X[t-1, 0] + 0.4 * X[t-2, 1] + rng.normal(scale=noise_std)
        X[t, 1] = 0.7 * X[t-1, 1] + 0.5 * X[t-1, 3] + rng.normal(scale=noise_std)
        X[t, 2] = 0.6 * X[t-1, 2] + 0.3 * X[t-2, 1] + 0.2 * X[t-3, 3] + rng.normal(scale=noise_std)
        X[t, 3] = 0.9 * X[t-1, 3] + rng.normal(scale=noise_std)
    return X

TRUE_LINKS_4VAR = {
    0: [(0, 1), (1, 2)],
    1: [(1, 1), (3, 1)],
    2: [(2, 1), (1, 2), (3, 3)],
    3: [(3, 1)],
}


# ── Scenario 2: 8-variable linear VAR(1) ─────────────────────────────────────
A_8VAR = np.array([
    [0.7,  0.0,  0.0,  0.3,  0.0,  0.0,  0.0,  0.0],
    [0.4,  0.6,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [0.0,  0.3,  0.7,  0.0,  0.0,  0.0,  0.0,  0.0],
    [0.0,  0.0,  0.2,  0.8,  0.0,  0.0,  0.0,  0.0],
    [0.0,  0.0,  0.0,  0.0,  0.7,  0.3,  0.0,  0.0],
    [0.0,  0.0,  0.0,  0.0,  0.2,  0.6,  0.0,  0.0],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.3,  0.7,  0.2],
    [0.1,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.8],
])
TRUE_LINKS_8VAR = links_from_matrix(A_8VAR, threshold=1e-8, lags=1)


# ── Scenario 3: 4-variable threshold VAR (nonlinear) ─────────────────────────
# Same causal structure as 4var but effect sizes switch with regime indicators.
# X3 -> X1: strong (gamma=0.8) when X3_{t-1} > 0, weak (gamma=0.2) otherwise.
# X1 -> X0: strong (gamma=0.6) when X1_{t-2} > 0, weak (gamma=0.1) otherwise.
# PCMCI (linear) will detect most links but may have reduced power.
def generate_threshold_4var(T: int, noise_std: float = 0.5, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.zeros((T, 4))
    for t in range(3, T):
        X[t, 3] = 0.9 * X[t-1, 3] + rng.normal(scale=noise_std)
        gamma_13 = 0.8 if X[t-1, 3] > 0 else 0.2
        X[t, 1] = 0.6 * X[t-1, 1] + gamma_13 * X[t-1, 3] + rng.normal(scale=noise_std)
        gamma_01 = 0.6 if X[t-2, 1] > 0 else 0.1
        X[t, 0] = 0.7 * X[t-1, 0] + gamma_01 * X[t-2, 1] + rng.normal(scale=noise_std)
        X[t, 2] = 0.6 * X[t-1, 2] + 0.3 * X[t-2, 1] + 0.2 * X[t-3, 3] + rng.normal(scale=noise_std)
    return X

# Same ground-truth causal structure as the linear 4var
TRUE_LINKS_THRESHOLD = TRUE_LINKS_4VAR


# ── Helper: 3-way split ───────────────────────────────────────────────────────
def three_way_split(data, train_frac, val_frac, window):
    T = len(data)
    n_train = int(T * train_frac)
    n_val   = int(T * val_frac)
    train_raw = data[:n_train]
    val_raw   = data[n_train - window: n_train + n_val]
    test_raw  = data[n_train + n_val - window:]
    return train_raw, val_raw, test_raw


# ── Helper: select best lambda on val set ────────────────────────────────────
def select_lambda(N, hidden, links, window, train_dl, val_dl, seed):
    """Grid-search lambda using val_dl; returns best lambda."""
    best_lam, best_val = LAMBDA_GRID[0], float("inf")
    for lam in LAMBDA_GRID:
        torch.manual_seed(seed)
        m = CausalLSTM(N, hidden, links, window, lambda_reg=lam).to(DEVICE)
        fit(m, train_dl, val_dl, epochs=EPOCHS // 3, lr=LR,
            device=DEVICE, verbose=False, patience=10)
        val_mse = 0.0
        m.eval()
        import torch.nn as nn
        mse_fn = nn.MSELoss()
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                yh, _ = m(xb)
                val_mse += mse_fn(yh, yb).item()
        val_mse /= len(val_dl)
        if val_mse < best_val:
            best_val, best_lam = val_mse, lam
    return best_lam


# ── Core scenario runner ──────────────────────────────────────────────────────
def run_scenario(name: str, data: np.ndarray, true_links: dict, hidden: int):
    print(f"\n{'='*60}\nScenario: {name}\n{'='*60}")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    N = data.shape[1]
    var_names = [f"X{i}" for i in range(N)]

    train_raw, val_raw, test_raw = three_way_split(data, TRAIN_FRAC, VAL_FRAC, WINDOW)
    scaler = fit_scaler(train_raw)
    train_norm = scaler.transform(train_raw)
    val_norm   = scaler.transform(val_raw)
    test_norm  = scaler.transform(test_raw)
    T_test = len(test_norm) - WINDOW

    print(f"  Train: {len(train_norm)}  Val: {len(val_norm)-WINDOW}  Test: {T_test}")

    # Causal discovery on train data
    print("  Running PCMCI...")
    pcmci_result = run_pcmci(train_norm, tau_max=5, alpha=0.05, pc_alpha=0.1,
                              test="parcorr", var_names=var_names)
    est_links = pcmci_result["links"]
    gm = compare_links(true_links, est_links, N, tau_max=5)
    print(f"  Graph recovery -> P={gm['precision']:.2f}  R={gm['recall']:.2f}  F1={gm['f1']:.2f}")

    plot_causal_graph_heatmap(
        est_links, var_names, tau_max=5,
        title=f"Estimated Causal Graph — {name}",
        save_path=os.path.join(FIGURES_DIR, f"causal_graph_{name}.png"),
    )

    train_ds = TimeSeriesDataset(train_norm, WINDOW)
    val_ds   = TimeSeriesDataset(val_norm,   WINDOW)
    test_ds  = TimeSeriesDataset(test_norm,  WINDOW)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

    results = {}

    # ── Baseline LSTM ──────────────────────────────────────────────────────────
    print("  Training Baseline LSTM...")
    torch.manual_seed(SEED)
    baseline = BaselineLSTM(N, hidden).to(DEVICE)
    h_base = fit(baseline, train_dl, val_dl, epochs=EPOCHS, lr=LR,
                 device=DEVICE, patience=15)
    preds_base, y_true = predict_one_step(baseline, test_dl, DEVICE)
    preds_base_orig = scaler.inverse_transform(preds_base)
    y_true_orig     = scaler.inverse_transform(y_true)
    results["Baseline"] = compute_metrics(y_true_orig, preds_base_orig)
    plot_training_history(h_base, f"Baseline — {name}",
                          os.path.join(FIGURES_DIR, f"history_baseline_{name}.png"))

    # ── Causal LSTM (PCMCI prior, lambda selected on val) ─────────────────────
    print("  Selecting lambda for PCMCI prior...")
    best_lam = select_lambda(N, hidden, est_links, WINDOW, train_dl, val_dl, SEED)
    print(f"  Best lambda = {best_lam}")
    torch.manual_seed(SEED)
    causal = CausalLSTM(N, hidden, est_links, WINDOW, lambda_reg=best_lam).to(DEVICE)
    h_causal = fit(causal, train_dl, val_dl, epochs=EPOCHS, lr=LR,
                   device=DEVICE, patience=15)
    preds_causal, _ = predict_one_step(causal, test_dl, DEVICE)
    preds_causal_orig = scaler.inverse_transform(preds_causal)
    results[f"Causal PCMCI (lam={best_lam})"] = compute_metrics(y_true_orig, preds_causal_orig)
    plot_training_history(h_causal, f"Causal (PCMCI) — {name}",
                          os.path.join(FIGURES_DIR, f"history_causal_{name}.png"))

    # ── Causal LSTM (True graph, same lambda) ─────────────────────────────────
    print("  Training Causal LSTM (True Graph)...")
    best_lam_true = select_lambda(N, hidden, true_links, WINDOW, train_dl, val_dl, SEED)
    torch.manual_seed(SEED)
    causal_true = CausalLSTM(N, hidden, true_links, WINDOW, lambda_reg=best_lam_true).to(DEVICE)
    fit(causal_true, train_dl, val_dl, epochs=EPOCHS, lr=LR,
        device=DEVICE, verbose=False, patience=15)
    preds_ct, _ = predict_one_step(causal_true, test_dl, DEVICE)
    preds_ct_orig = scaler.inverse_transform(preds_ct)
    results[f"Causal True (lam={best_lam_true})"] = compute_metrics(y_true_orig, preds_ct_orig)

    # ── Causal LSTM (Random graph, same density + same lambda as PCMCI) ────────
    print("  Training Causal LSTM (Random Graph)...")
    n_links = sum(len(v) for v in est_links.values())
    rng = np.random.default_rng(SEED + 1)
    random_links = {j: [] for j in range(N)}
    for _ in range(n_links):
        random_links[int(rng.integers(0, N))].append(
            (int(rng.integers(0, N)), int(rng.integers(1, WINDOW + 1)))
        )
    torch.manual_seed(SEED)
    causal_rand = CausalLSTM(N, hidden, random_links, WINDOW, lambda_reg=best_lam).to(DEVICE)
    fit(causal_rand, train_dl, val_dl, epochs=EPOCHS, lr=LR,
        device=DEVICE, verbose=False, patience=15)
    preds_cr, _ = predict_one_step(causal_rand, test_dl, DEVICE)
    preds_cr_orig = scaler.inverse_transform(preds_cr)
    results["Causal Random"] = compute_metrics(y_true_orig, preds_cr_orig)

    # ── Masked LSTM (PCMCI graph) ──────────────────────────────────────────────
    print("  Training Masked LSTM (PCMCI graph)...")
    torch.manual_seed(SEED)
    masked_pcmci = MaskedLSTM(N, hidden, est_links, WINDOW).to(DEVICE)
    fit(masked_pcmci, train_dl, val_dl, epochs=EPOCHS, lr=LR,
        device=DEVICE, verbose=False, patience=15)
    preds_mp, _ = predict_one_step(masked_pcmci, test_dl, DEVICE)
    preds_mp_orig = scaler.inverse_transform(preds_mp)
    results["Masked PCMCI"] = compute_metrics(y_true_orig, preds_mp_orig)

    # ── Masked LSTM (True graph) ───────────────────────────────────────────────
    print("  Training Masked LSTM (True graph)...")
    torch.manual_seed(SEED)
    masked_true = MaskedLSTM(N, hidden, true_links, WINDOW).to(DEVICE)
    fit(masked_true, train_dl, val_dl, epochs=EPOCHS, lr=LR,
        device=DEVICE, verbose=False, patience=15)
    preds_mt, _ = predict_one_step(masked_true, test_dl, DEVICE)
    preds_mt_orig = scaler.inverse_transform(preds_mt)
    results["Masked True"] = compute_metrics(y_true_orig, preds_mt_orig)

    # ── Masked LSTM (Random graph, same density) ───────────────────────────────
    print("  Training Masked LSTM (Random graph)...")
    torch.manual_seed(SEED)
    masked_rand = MaskedLSTM(N, hidden, random_links, WINDOW).to(DEVICE)
    fit(masked_rand, train_dl, val_dl, epochs=EPOCHS, lr=LR,
        device=DEVICE, verbose=False, patience=15)
    preds_mr, _ = predict_one_step(masked_rand, test_dl, DEVICE)
    preds_mr_orig = scaler.inverse_transform(preds_mr)
    results["Masked Random"] = compute_metrics(y_true_orig, preds_mr_orig)

    # ── DM tests vs Baseline ───────────────────────────────────────────────────
    e_base = y_true_orig - preds_base_orig
    dm_results = {}
    for mname, mdata in [
        (f"Causal PCMCI (lam={best_lam})", preds_causal_orig),
        (f"Causal True (lam={best_lam_true})", preds_ct_orig),
        ("Causal Random", preds_cr_orig),
        ("Masked PCMCI", preds_mp_orig),
        ("Masked True", preds_mt_orig),
        ("Masked Random", preds_mr_orig),
    ]:
        e2 = y_true_orig - mdata
        dm_stat, pval = diebold_mariano_test(e_base, e2)
        dm_results[mname] = {"dm_stat": dm_stat, "p_value": pval}
        sig = "**" if pval < 0.05 else ("*" if pval < 0.10 else "")
        print(f"  DM({mname} vs Baseline): stat={dm_stat:.3f}  p={pval:.3f} {sig}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n  {'Model':<30} {'MSE':>10} {'MAE':>10}")
    print("  " + "-" * 52)
    for mname, m in results.items():
        print(f"  {mname:<30} {m['mse']:>10.6f} {m['mae']:>10.6f}")

    # ── Predictions plot ──────────────────────────────────────────────────────
    plot_predictions(
        y_true_orig,
        {"Baseline": preds_base_orig, f"Causal (lam={best_lam})": preds_causal_orig},
        var_names=var_names,
        title_prefix=f"{name} — ",
        save_path=os.path.join(FIGURES_DIR, f"predictions_{name}.png"),
        max_vars=min(N, 4),
    )

    return results, gm, dm_results, best_lam


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    all_results = {}

    # Scenario 1: 4var linear
    data_4 = generate_4var(T)
    res_4, gm_4, dm_4, lam_4 = run_scenario("4var", data_4, TRUE_LINKS_4VAR, hidden=8)
    all_results["4var"] = res_4

    # Scenario 2: 8var linear
    data_8 = generate_var_series(A_8VAR, T, noise_std=0.3, seed=SEED)
    res_8, gm_8, dm_8, lam_8 = run_scenario("8var", data_8, TRUE_LINKS_8VAR, hidden=16)
    all_results["8var"] = res_8

    # Scenario 3: 4var threshold (nonlinear)
    data_th = generate_threshold_4var(T)
    res_th, gm_th, dm_th, lam_th = run_scenario("threshold", data_th, TRUE_LINKS_THRESHOLD, hidden=8)
    all_results["threshold"] = res_th

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SYNTHETIC EXPERIMENTS — FINAL SUMMARY")
    print("=" * 70)
    for scenario, models in all_results.items():
        print(f"\n  {scenario}")
        for mname, m in models.items():
            print(f"    {mname:<30} MSE={m['mse']:.6f}  MAE={m['mae']:.6f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "synthetic_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "model", "mse", "rmse", "mae"])
        for scenario, models in all_results.items():
            for mname, m in models.items():
                writer.writerow([scenario, mname, m["mse"], m["rmse"], m["mae"]])

    # Save DM test results
    dm_csv = os.path.join(RESULTS_DIR, "synthetic_dm_tests.csv")
    with open(dm_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "model", "dm_stat", "p_value"])
        for scenario, dm_res in [("4var", dm_4), ("8var", dm_8), ("threshold", dm_th)]:
            for mname, v in dm_res.items():
                writer.writerow([scenario, mname, v["dm_stat"], v["p_value"]])

    # Save graph recovery metrics
    gm_csv = os.path.join(RESULTS_DIR, "synthetic_graph_metrics.csv")
    with open(gm_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "precision", "recall", "f1", "tp", "fp", "fn"])
        for scenario, gm in [("4var", gm_4), ("8var", gm_8), ("threshold", gm_th)]:
            writer.writerow([scenario, gm["precision"], gm["recall"], gm["f1"],
                             gm["tp"], gm["fp"], gm["fn"]])

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
