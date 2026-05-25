"""
LSTM models for causal regularization experiments.

Three model variants:
  BaselineLSTM  – standard LSTM, no causal constraint.
  CausalLSTM   – soft penalty: penalises input gradients for non-causal pairs.
  MaskedLSTM   – hard mask: zeroes non-causal inputs before the LSTM sees them.
                 Same architecture / parameter count as BaselineLSTM (fair comparison).
                 No hyperparameter λ needed.
"""
import torch
import torch.nn as nn


class BaselineLSTM(nn.Module):
    def __init__(self, n_vars: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_vars,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_size, n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_vars)
        _, (h_n, _) = self.lstm(x)
        return self.out(h_n[-1])


class CausalLSTM(nn.Module):
    """
    LSTM with causal regularization.

    During forward() we need gradients w.r.t. the input, so we return both
    the prediction and the input tensor with requires_grad=True.  The caller
    must use physics_penalty() before calling loss.backward().

    Args:
        n_vars: number of variables (N)
        hidden_size: LSTM hidden units
        causal_links: dict {j: [(i, lag), ...]} listing *existing* causal edges,
                      where lag > 0 means i at time t-lag causes j at time t.
        window_len: input window length L
        lambda_reg: regularization strength
        num_layers: number of LSTM layers
        dropout: dropout probability (only active with num_layers > 1)
    """

    def __init__(
        self,
        n_vars: int,
        hidden_size: int,
        causal_links: dict,
        window_len: int,
        lambda_reg: float = 1e-2,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.window_len = window_len
        self.lambda_reg = lambda_reg

        self.lstm = nn.LSTM(
            input_size=n_vars,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_size, n_vars)

        # mask_non_edge[t, i, j] = True means (i -> j at lag window_len-t) is NOT a causal edge
        # i.e., we penalize the gradient of ŷ_j w.r.t. x_{t, i}
        mask = torch.ones(window_len, n_vars, n_vars, dtype=torch.bool)
        for j, links in causal_links.items():
            for i, lag in links:
                if 1 <= lag <= window_len:
                    # position in window: most recent is index window_len-1, lag 1 → index window_len-1
                    mask[window_len - lag, i, j] = False
        self.register_buffer("mask_non_edge", mask)

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.out(h_n[-1])

    def forward(self, x: torch.Tensor):
        x_req = x.detach().requires_grad_(True)
        y_hat = self._predict(x_req)
        return y_hat, x_req

    def causal_penalty(self, y_hat: torch.Tensor, x_req: torch.Tensor) -> torch.Tensor:
        """
        Penalizes ∂ŷ_j/∂x_{t,i} for all (i,j,t) pairs that are non-edges in the causal graph.

        penalty = λ * (1/B) * Σ_{j,t,i: non-edge} (∂ŷ_j / ∂x_{t,i})²
        """
        batch_size = y_hat.shape[0]
        penalty = torch.tensor(0.0, device=y_hat.device)

        for j in range(self.n_vars):
            grad_j = torch.autograd.grad(
                outputs=y_hat[:, j].sum(),
                inputs=x_req,
                create_graph=True,
                retain_graph=True,
            )[0]  # (batch, window_len, n_vars)

            # mask_non_edge[:, :, j]: (window_len, n_vars) broadcast over batch
            non_edge_mask = self.mask_non_edge[:, :, j]  # (L, N)
            penalty = penalty + (grad_j ** 2 * non_edge_mask).sum()

        return self.lambda_reg * penalty / batch_size


class MaskedLSTM(nn.Module):
    """
    Hard causal masking LSTM (Zhang et al. 2024 / Khosravinia et al. 2026 style).

    Before every forward pass, non-causal input positions are zeroed out so the
    LSTM never "sees" inputs that the causal graph says should not matter.

    The mask is lag-specific: entry (t_idx, i) = 1 iff variable i is a causal
    parent of *some* output j at the lag corresponding to window position t_idx.
    This faithfully encodes the directed, lagged PCMCI graph — more precise than
    simple variable-level selection.

    Key properties vs CausalLSTM:
      - No λ hyperparameter (binary: on/off).
      - Same hidden_size / num_layers / parameter count as BaselineLSTM.
      - Gradient cannot flow through masked positions at all during training.

    Args:
        n_vars:       number of variables N
        hidden_size:  LSTM hidden units
        causal_links: dict {j: [(i, lag), ...]} of causal parents per target
        window_len:   input window length L
        num_layers:   LSTM depth
        dropout:      dropout (only active with num_layers > 1)
    """

    def __init__(
        self,
        n_vars: int,
        hidden_size: int,
        causal_links: dict,
        window_len: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.window_len = window_len

        self.lstm = nn.LSTM(
            input_size=n_vars,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_size, n_vars)

        # Build lag-specific binary mask: mask[t_idx, i] = 1 iff variable i is
        # a causal parent of any j at lag = window_len - t_idx.
        mask = torch.zeros(window_len, n_vars)
        for j, links in causal_links.items():
            for i, lag in links:
                if 1 <= lag <= window_len:
                    mask[window_len - lag, i] = 1.0

        # Safety: if the graph is empty (no links at all), fall back to no masking.
        if mask.sum() == 0:
            mask = torch.ones(window_len, n_vars)

        self.register_buffer("input_mask", mask)  # (window_len, n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_len, n_vars)
        x_masked = x * self.input_mask.unsqueeze(0)  # broadcast over batch dim
        _, (h_n, _) = self.lstm(x_masked)
        return self.out(h_n[-1])
