"""
Visualization utilities.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_predictions(
    y_true: np.ndarray,
    predictions: dict,
    var_names: list,
    title_prefix: str = "",
    save_path: str = None,
    max_vars: int = None,
    n_cols: int = 2,
):
    """
    Plot true vs predicted values for each variable.

    Args:
        y_true: (T, N) array
        predictions: {'Model Name': (T, N) array, ...}
        var_names: list of N variable names
        title_prefix: prefix for subplot titles
        save_path: if given, saves figure to this path
        max_vars: limit number of variables plotted
        n_cols: number of columns in subplot grid
    """
    N = y_true.shape[1]
    if max_vars is not None:
        N = min(N, max_vars)
        var_names = var_names[:N]
        y_true = y_true[:, :N]
        predictions = {k: v[:, :N] for k, v in predictions.items()}

    n_rows = int(np.ceil(N / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3 * n_rows), squeeze=False)
    t = np.arange(y_true.shape[0])

    colors = ["b", "r", "g", "m", "orange"]
    styles = ["--", ":", "-.", "--", ":"]

    for idx, var in enumerate(var_names):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        ax.plot(t, y_true[:, idx], "k-", lw=1.5, label="Real")
        for j, (name, pred) in enumerate(predictions.items()):
            ax.plot(t, pred[:, idx], colors[j % len(colors)] + styles[j % len(styles)],
                    lw=1.2, label=name)
        ax.set_title(f"{title_prefix}{var}")
        ax.legend(fontsize=8)
        ax.grid(linestyle="--", alpha=0.5)

    # hide empty subplots
    for idx in range(N, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_causal_graph_heatmap(
    links: dict,
    var_names: list,
    tau_max: int,
    title: str = "Causal Graph",
    save_path: str = None,
):
    """
    Show a heatmap of significant causal links (collapsed over lags).
    Entry (i, j) = number of significant lags from i to j.
    """
    N = len(var_names)
    matrix = np.zeros((N, N), dtype=int)
    for j, lst in links.items():
        for i, lag in lst:
            matrix[i, j] += 1

    fig, ax = plt.subplots(figsize=(max(6, N * 0.6), max(5, N * 0.55)))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(var_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(var_names, fontsize=8)
    ax.set_xlabel("Effect (j)"); ax.set_ylabel("Cause (i)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="# significant lags")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_history(history: dict, title: str = "Training History", save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_mse"]) + 1)

    axes[0].plot(epochs, history["train_mse"], label="Train MSE")
    axes[0].plot(epochs, history["val_mse"], label="Val MSE")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE")
    axes[0].set_title(f"{title} — MSE")
    axes[0].legend(); axes[0].grid(linestyle="--", alpha=0.5)
    axes[0].set_yscale("log")

    axes[1].plot(epochs, history["train_pen"], color="red", label="Train Penalty")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Causal Penalty")
    axes[1].set_title(f"{title} — Regularization Term")
    axes[1].legend(); axes[1].grid(linestyle="--", alpha=0.5)
    if any(v > 0 for v in history["train_pen"]):
        axes[1].set_yscale("log")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
