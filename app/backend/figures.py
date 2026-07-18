"""
Turn pipeline output into base64-encoded PNG figures for the web front-end.

Reuses `src/plotting` for the causal-graph heatmap and forecast overlays, and
adds a compact causal-network diagram. Everything runs on the non-interactive
Agg backend so it is safe inside a background worker thread.
"""
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.plotting import plot_causal_graph_heatmap, plot_predictions, plot_training_history


def _fig_to_b64(fig) -> str:
    """Encode a matplotlib figure as a base64 PNG data string (no data: prefix)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _links_from_json(links_json) -> dict:
    """Convert {str(j): [[i, lag]]} back to {j: [(i, lag)]}."""
    return {int(j): [(int(i), int(lag)) for (i, lag) in lst]
            for j, lst in links_json.items()}


def graph_heatmap_b64(result) -> str:
    links = _links_from_json(result["links"])
    fig = plot_causal_graph_heatmap(
        links, result["var_names"], tau_max=result["tau_max"],
        title=f"PCMCI Causal Graph ({result['n_links']} links)")
    return _fig_to_b64(fig)


def graph_network_b64(result) -> str:
    """
    Draw the lagged causal graph as a circular node-link diagram.

    Node = variable; a directed arrow i->j means i is a causal parent of j
    (aggregated over lags; arrow width scales with the number of lags).
    """
    var_names = result["var_names"]
    N = len(var_names)
    links = _links_from_json(result["links"])

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pos = np.c_[np.cos(angles), np.sin(angles)]

    fig, ax = plt.subplots(figsize=(6, 6))
    # self-loops and cross edges
    for j, lst in links.items():
        counts = {}
        for i, _lag in lst:
            counts[i] = counts.get(i, 0) + 1
        for i, c in counts.items():
            if i == j:
                # self-loop: small circle above the node
                cx, cy = pos[i]
                loop = plt.Circle((cx * 1.12, cy * 1.12), 0.06,
                                  color="0.5", fill=False, lw=1 + 0.6 * c)
                ax.add_patch(loop)
                continue
            x0, y0 = pos[i]
            x1, y1 = pos[j]
            ax.annotate(
                "", xy=(x1 * 0.9, y1 * 0.9), xytext=(x0 * 0.9, y0 * 0.9),
                arrowprops=dict(arrowstyle="-|>", color="#3b6fb0",
                                lw=1 + 0.7 * c, alpha=0.7,
                                shrinkA=12, shrinkB=12,
                                connectionstyle="arc3,rad=0.12"))

    ax.scatter(pos[:, 0], pos[:, 1], s=900, c="#e8eef7",
               edgecolors="#3b6fb0", zorder=3)
    for i, name in enumerate(var_names):
        ax.text(pos[i, 0], pos[i, 1], name, ha="center", va="center",
                fontsize=8, zorder=4)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Causal network (i → j, width ∝ #lags)")
    return _fig_to_b64(fig)


def forecast_b64(result, max_vars=6) -> str:
    arrays = result["_arrays"]
    y_true = arrays["y_true"]
    preds = arrays["preds"]

    # Show at most three curves for legibility: baseline, best causal, one classic.
    keys = list(preds.keys())
    chosen = {}
    if "LSTM Baseline" in preds:
        chosen["LSTM Baseline"] = preds["LSTM Baseline"]
    for cand in ("LSTM Causal (PCMCI)", "LSTM Masked (PCMCI)"):
        if cand in preds:
            chosen[cand] = preds[cand]
            break
    for cand in ("VAR", "Random Walk"):
        if cand in preds:
            chosen[cand] = preds[cand]
            break
    if not chosen:
        chosen = {k: preds[k] for k in keys[:3]}

    fig = plot_predictions(y_true, chosen, var_names=result["var_names"],
                           title_prefix="", max_vars=max_vars)
    return _fig_to_b64(fig)


def history_b64(result) -> str:
    hist = result.get("history", {})
    key = "LSTM Causal (PCMCI)" if "LSTM Causal (PCMCI)" in hist else None
    if key is None:
        key = "LSTM Baseline" if "LSTM Baseline" in hist else None
    if key is None:
        return ""
    fig = plot_training_history(hist[key], title=key)
    return _fig_to_b64(fig)


def build_all_figures(result) -> dict:
    """Render every figure the front-end shows; returns {name: base64 png}."""
    return {
        "graph_heatmap": graph_heatmap_b64(result),
        "graph_network": graph_network_b64(result),
        "forecast": forecast_b64(result),
        "history": history_b64(result),
    }
