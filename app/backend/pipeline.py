"""
Core forecasting pipeline — a parametrized, faithful refactor of
`experiments/exp_fred.py::main`.

One function, `run_pipeline`, drives the whole comparison:
    stationarize (done by caller) -> chronological 3-way split -> scale ->
    PCMCI causal discovery -> train {Random Walk, VAR, LSTM baseline, LSTM-L2,
    Causal (soft, lambda on val), Causal-random ablation, Masked (hard),
    Masked-random ablation} -> evaluate with Diebold-Mariano (HLN) tests and
    block-bootstrap CIs, all vs the LSTM baseline.

Both the FastAPI backend and the benchmark harness call this — no pipeline logic
is duplicated. The function returns a plain, JSON-friendly dict (plus a private
`_arrays` block used only for figure rendering); it never touches matplotlib or
FastAPI so it stays reusable and headless.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import TimeSeriesDataset, fit_scaler
from src.causal_discovery import run_pcmci, count_links, compare_links
from src.models import BaselineLSTM, CausalLSTM, MaskedLSTM
from src.train import fit, predict_one_step
from src.evaluate import (compute_metrics, compute_metrics_per_var,
                          diebold_mariano_test, block_bootstrap_ci)
from src.baselines import VARBaseline, RandomWalkBaseline


# ── Configuration ─────────────────────────────────────────────────────────────

def default_config() -> dict:
    """Default pipeline configuration (mirrors exp_fred defaults)."""
    return {
        "window": 12,
        "tau_max": 6,
        "alpha": 0.05,
        "pc_alpha": 0.1,
        "pcmci_test": "parcorr",
        "hidden": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "epochs": 150,
        "lr": 5e-4,
        "batch": 32,
        "train_frac": 0.70,
        "val_frac": 0.10,
        "lambda_causal": [0.001, 0.01, 0.1, 1.0],
        "lambda_l2": [1e-5, 1e-4, 1e-3, 1e-2],
        "seed": 42,
        "device": "cpu",
        # Which models to run (baseline is always run — it's the DM reference).
        "models": ["rw", "var", "l2", "causal", "causal_random",
                   "masked", "masked_random"],
        "bootstrap_block": 12,
        "bootstrap_n": 500,
    }


# ── Small helpers (mirrors exp_fred) ─────────────────────────────────────────

def _make_loader(norm_data, window, batch):
    ds = TimeSeriesDataset(norm_data, window)
    return DataLoader(ds, batch_size=batch, shuffle=False)


def _eval_mse(model, loader, device):
    model.eval()
    mse_fn = nn.MSELoss()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            yh = model(xb)[0] if isinstance(model, CausalLSTM) else model(xb)
            total += mse_fn(yh, yb).item()
    return total / len(loader)


def _best_lambda_causal(N, cfg, links, train_dl, val_dl):
    grid, seed, dev = cfg["lambda_causal"], cfg["seed"], cfg["device"]
    best_lam, best_val = grid[0], float("inf")
    for lam in grid:
        torch.manual_seed(seed)
        m = CausalLSTM(N, cfg["hidden"], links, cfg["window"], lambda_reg=lam,
                       num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(dev)
        fit(m, train_dl, val_dl, epochs=max(1, cfg["epochs"] // 3), lr=cfg["lr"],
            device=dev, verbose=False, patience=10)
        v = _eval_mse(m, val_dl, dev)
        if v < best_val:
            best_val, best_lam = v, lam
    return best_lam


def _best_lambda_l2(N, cfg, train_dl, val_dl):
    grid, seed, dev = cfg["lambda_l2"], cfg["seed"], cfg["device"]
    best_lam, best_val = grid[0], float("inf")
    for lam in grid:
        torch.manual_seed(seed)
        m = BaselineLSTM(N, cfg["hidden"], num_layers=cfg["num_layers"],
                         dropout=cfg["dropout"]).to(dev)
        fit(m, train_dl, val_dl, epochs=max(1, cfg["epochs"] // 3), lr=cfg["lr"],
            weight_decay=lam, device=dev, verbose=False, patience=10)
        v = _eval_mse(m, val_dl, dev)
        if v < best_val:
            best_val, best_lam = v, lam
    return best_lam


def _train_lstm(model, cfg, train_dl, val_dl):
    torch.manual_seed(cfg["seed"])
    model = model.to(cfg["device"])
    history = fit(model, train_dl, val_dl, epochs=cfg["epochs"], lr=cfg["lr"],
                  device=cfg["device"], verbose=False, patience=20)
    return model, history


def _random_links(est_links, N, tau_max, seed):
    """Density-matched random graph: same edge count, random (i, lag)."""
    rng = np.random.default_rng(seed + 99)
    rand = {j: [] for j in range(N)}
    for j, lst in est_links.items():
        for _ in lst:
            rand[j].append((int(rng.integers(0, N)),
                            int(rng.integers(1, tau_max + 1))))
    return rand


def _var_rolling(var_mdl, history_init, y_actual_norm, n_steps, N):
    """Rolling one-step VAR forecast, feeding the true value back each step."""
    preds = np.zeros((n_steps, N))
    hist = history_init.copy()
    lag = max(1, var_mdl.lag_order)
    for t in range(n_steps):
        fc = var_mdl._fitted.forecast(hist[-lag:], steps=1)
        preds[t] = fc[0]
        hist = np.vstack([hist, y_actual_norm[t: t + 1]])
    return preds


def _links_to_json(links):
    """Convert {j: [(i, lag), ...]} to JSON-serializable {str(j): [[i, lag]]}."""
    return {str(j): [[int(i), int(lag)] for (i, lag) in lst]
            for j, lst in links.items()}


# ── Main entry point ─────────────────────────────────────────────────────────

def run_pipeline(data, var_names, cfg=None, true_links=None, progress=None,
                 pcmci_links=None, eval_val=False):
    """
    Run the full causal-forecasting comparison on a (T, N) array.

    Args:
        data: (T, N) stationary numpy array (already stationarized by the caller).
        var_names: list of N variable names.
        cfg: config dict (see `default_config`); missing keys are filled in.
        true_links: optional ground-truth {j: [(i, lag)]} for graph-recovery F1.
        progress: optional callback(message: str, frac: float in [0,1]).
        pcmci_links: optional precomputed {j: [(i, lag)]} to skip PCMCI (caching).

    Returns:
        dict with keys: config, var_names, n_vars, n_obs, splits, links, n_links,
        graph_f1, models (metrics + DM + CI per model), per_var, history,
        best_lambda_causal, best_lambda_l2, and a private `_arrays` block
        {y_true, preds:{name: array}} for figure rendering.
    """
    base = default_config()
    if cfg:
        base.update(cfg)
    cfg = base

    def report(msg, frac):
        if progress is not None:
            progress(msg, frac)

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    data = np.asarray(data, dtype=np.float32)
    N = data.shape[1]
    T_total = len(data)
    window, dev = cfg["window"], cfg["device"]

    # ── Chronological 3-way split (val/test prepend `window` rows) ──────────────
    n_train = int(T_total * cfg["train_frac"])
    n_val = int(T_total * cfg["val_frac"])
    if n_train <= window or n_val <= 0 or (T_total - n_train - n_val) <= window:
        raise ValueError(
            f"Series too short (T={T_total}) for window={window} with the given "
            f"train/val fractions. Need a longer series or a smaller window.")

    train_raw = data[:n_train]
    val_raw = data[n_train - window: n_train + n_val]
    test_raw = data[n_train + n_val - window:]

    scaler = fit_scaler(train_raw)
    train_norm = scaler.transform(train_raw)
    val_norm = scaler.transform(val_raw)
    test_norm = scaler.transform(test_raw)

    T_test_eff = len(test_norm) - window
    y_true_norm = test_norm[window:]
    y_true_orig = scaler.inverse_transform(y_true_norm)

    T_val_eff = len(val_norm) - window
    y_true_val_orig = scaler.inverse_transform(val_norm[window:])

    report("PCMCI causal discovery", 0.05)

    # ── PCMCI (or precomputed links) ────────────────────────────────────────────
    if pcmci_links is not None:
        est_links = pcmci_links
    else:
        pcmci_result = run_pcmci(
            train_norm, tau_max=cfg["tau_max"], alpha=cfg["alpha"],
            pc_alpha=cfg["pc_alpha"], test=cfg["pcmci_test"], var_names=var_names,
        )
        est_links = pcmci_result["links"]

    n_links = count_links(est_links)
    graph_f1 = None
    if true_links is not None:
        graph_f1 = compare_links(true_links, est_links, N, cfg["tau_max"])

    train_dl = _make_loader(train_norm, window, cfg["batch"])
    val_dl = _make_loader(val_norm, window, cfg["batch"])
    test_dl = _make_loader(test_norm, window, cfg["batch"])

    preds = {}           # name -> (T_test, N) original-scale predictions
    metrics = {}         # name -> metrics dict
    history = {}         # name -> training history (LSTM models only)
    lstm_objs = {}       # name -> trained torch model (for optional val eval)
    which = set(cfg["models"])

    # ── Random Walk ─────────────────────────────────────────────────────────────
    if "rw" in which:
        report("Random Walk baseline", 0.10)
        rw = RandomWalkBaseline()
        p = scaler.inverse_transform(rw.predict(test_norm, window))
        preds["Random Walk"] = p
        metrics["Random Walk"] = compute_metrics(y_true_orig, p)

    # ── VAR(BIC) — clean rolling one-step forecast (per exp_fred) ───────────────
    if "var" in which:
        report("VAR(BIC) baseline", 0.15)
        var_mdl = VARBaseline(maxlags=cfg["tau_max"], ic="bic")
        var_mdl.fit(train_norm)
        history_to_test = np.vstack([train_norm, val_norm[window:]])
        preds_var = _var_rolling(var_mdl, history_to_test, y_true_norm, T_test_eff, N)
        p = scaler.inverse_transform(preds_var)
        preds["VAR"] = p
        metrics["VAR"] = compute_metrics(y_true_orig, p)

    # ── LSTM Baseline (always — DM reference) ──────────────────────────────────
    report("LSTM baseline", 0.25)
    baseline, h_base = _train_lstm(
        BaselineLSTM(N, cfg["hidden"], num_layers=cfg["num_layers"],
                     dropout=cfg["dropout"]), cfg, train_dl, val_dl)
    p_base_norm, _ = predict_one_step(baseline, test_dl, dev)
    p_base = scaler.inverse_transform(p_base_norm)
    preds["LSTM Baseline"] = p_base
    metrics["LSTM Baseline"] = compute_metrics(y_true_orig, p_base)
    history["LSTM Baseline"] = h_base
    lstm_objs["LSTM Baseline"] = baseline
    e_base = y_true_orig - p_base

    # ── LSTM-L2 (weight decay, lambda on val) ──────────────────────────────────
    if "l2" in which:
        report("LSTM-L2 (weight decay)", 0.40)
        best_lam_l2 = _best_lambda_l2(N, cfg, train_dl, val_dl)
        torch.manual_seed(cfg["seed"])
        lstm_l2 = BaselineLSTM(N, cfg["hidden"], num_layers=cfg["num_layers"],
                               dropout=cfg["dropout"]).to(dev)
        fit(lstm_l2, train_dl, val_dl, epochs=cfg["epochs"], lr=cfg["lr"],
            weight_decay=best_lam_l2, device=dev, verbose=False, patience=20)
        p = scaler.inverse_transform(predict_one_step(lstm_l2, test_dl, dev)[0])
        preds["LSTM-L2"] = p
        metrics["LSTM-L2"] = compute_metrics(y_true_orig, p)
        lstm_objs["LSTM-L2"] = lstm_l2
    else:
        best_lam_l2 = None

    # ── Causal LSTM (soft penalty, lambda on val) ──────────────────────────────
    best_lam_c = None
    if "causal" in which:
        report("Causal LSTM (soft penalty)", 0.55)
        best_lam_c = _best_lambda_causal(N, cfg, est_links, train_dl, val_dl)
        causal, h_causal = _train_lstm(
            CausalLSTM(N, cfg["hidden"], est_links, window, lambda_reg=best_lam_c,
                       num_layers=cfg["num_layers"], dropout=cfg["dropout"]),
            cfg, train_dl, val_dl)
        p = scaler.inverse_transform(predict_one_step(causal, test_dl, dev)[0])
        preds["LSTM Causal (PCMCI)"] = p
        metrics["LSTM Causal (PCMCI)"] = compute_metrics(y_true_orig, p)
        history["LSTM Causal (PCMCI)"] = h_causal
        lstm_objs["LSTM Causal (PCMCI)"] = causal

    # ── Causal LSTM (random-graph ablation) ────────────────────────────────────
    if "causal_random" in which and best_lam_c is not None:
        report("Causal LSTM (random ablation)", 0.65)
        rand_links = _random_links(est_links, N, cfg["tau_max"], cfg["seed"])
        crand, _ = _train_lstm(
            CausalLSTM(N, cfg["hidden"], rand_links, window, lambda_reg=best_lam_c,
                       num_layers=cfg["num_layers"], dropout=cfg["dropout"]),
            cfg, train_dl, val_dl)
        p = scaler.inverse_transform(predict_one_step(crand, test_dl, dev)[0])
        preds["LSTM Causal (Random)"] = p
        metrics["LSTM Causal (Random)"] = compute_metrics(y_true_orig, p)
        lstm_objs["LSTM Causal (Random)"] = crand

    # ── Masked LSTM (hard mask, no lambda) ─────────────────────────────────────
    if "masked" in which:
        report("Masked LSTM (hard mask)", 0.75)
        masked, h_masked = _train_lstm(
            MaskedLSTM(N, cfg["hidden"], est_links, window,
                       num_layers=cfg["num_layers"], dropout=cfg["dropout"]),
            cfg, train_dl, val_dl)
        p = scaler.inverse_transform(predict_one_step(masked, test_dl, dev)[0])
        preds["LSTM Masked (PCMCI)"] = p
        metrics["LSTM Masked (PCMCI)"] = compute_metrics(y_true_orig, p)
        history["LSTM Masked (PCMCI)"] = h_masked
        lstm_objs["LSTM Masked (PCMCI)"] = masked

    # ── Masked LSTM (random-graph ablation) ────────────────────────────────────
    if "masked_random" in which:
        report("Masked LSTM (random ablation)", 0.85)
        rand_links = _random_links(est_links, N, cfg["tau_max"], cfg["seed"])
        mrand, _ = _train_lstm(
            MaskedLSTM(N, cfg["hidden"], rand_links, window,
                       num_layers=cfg["num_layers"], dropout=cfg["dropout"]),
            cfg, train_dl, val_dl)
        p = scaler.inverse_transform(predict_one_step(mrand, test_dl, dev)[0])
        preds["LSTM Masked (Random)"] = p
        metrics["LSTM Masked (Random)"] = compute_metrics(y_true_orig, p)

    # ── Diebold-Mariano tests + block-bootstrap CIs (all vs LSTM Baseline) ──────
    report("Statistical tests", 0.92)
    for name in preds:
        m = metrics[name]
        sq = ((y_true_orig - preds[name]) ** 2).mean(axis=1)
        lo, hi = block_bootstrap_ci(sq, block_size=cfg["bootstrap_block"],
                                    n_bootstrap=cfg["bootstrap_n"], seed=cfg["seed"])
        m["ci_lower"], m["ci_upper"] = lo, hi
        if name == "LSTM Baseline":
            m["dm_stat"], m["p_value"] = None, None
        else:
            e2 = y_true_orig - preds[name]
            dm_stat, pval = diebold_mariano_test(e_base, e2)
            m["dm_stat"], m["p_value"] = dm_stat, pval

    # ── Per-variable MAE table ─────────────────────────────────────────────────
    per_var = {vn: {} for vn in var_names}
    for name, p in preds.items():
        pv = compute_metrics_per_var(y_true_orig, p, var_names)
        for vn in var_names:
            per_var[vn][name] = pv[vn]["mae"]

    # ── Optional: evaluate every model on the VALIDATION set too ────────────────
    # This is what enables the ex-ante question: does the model that wins on
    # validation also win on test? (i.e. can you pick the right model in advance?)
    val_models = None
    if eval_val:
        report("Validation evaluation", 0.97)
        val_metrics = {}
        for name, m in lstm_objs.items():
            vp = scaler.inverse_transform(predict_one_step(m, val_dl, dev)[0])
            val_metrics[name] = compute_metrics(y_true_val_orig, vp)
        if "rw" in which and "Random Walk" in preds:
            vp = scaler.inverse_transform(rw.predict(val_norm, window))
            val_metrics["Random Walk"] = compute_metrics(y_true_val_orig, vp)
        if "var" in which and "VAR" in preds:
            vp = _var_rolling(var_mdl, train_norm, val_norm[window:], T_val_eff, N)
            val_metrics["VAR"] = compute_metrics(y_true_val_orig,
                                                 scaler.inverse_transform(vp))
        val_models = val_metrics

    report("Done", 1.0)

    return {
        "config": {k: v for k, v in cfg.items() if k != "device"},
        "var_names": var_names,
        "n_vars": N,
        "n_obs": T_total,
        "splits": {"train": int(n_train), "val": int(n_val),
                   "test": int(T_test_eff)},
        "links": _links_to_json(est_links),
        "n_links": int(n_links),
        "graph_f1": graph_f1,
        "tau_max": cfg["tau_max"],
        "best_lambda_causal": best_lam_c,
        "best_lambda_l2": best_lam_l2,
        "models": metrics,
        "val_models": val_models,
        "per_var": per_var,
        "history": history,
        "_arrays": {"y_true": y_true_orig, "preds": preds},
    }
