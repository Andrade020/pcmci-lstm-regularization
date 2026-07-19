# Ex-ante decidability — can you tell in advance if causal LSTM helps?

Every model is scored on a held-out **validation** split and on **test**. If the validation ranking predicts the test ranking, you can choose the right model in advance without ground truth.

## 1. Does 'causal beats baseline on validation' predict test?

Gain = baseline MSE − model MSE (>0 means the causal model is better). A call is **correct** when the sign matches on val and test.

| Dataset | F1 | Soft val-gain | Soft test-gain | Soft call | Masked val-gain | Masked test-gain | Masked call |
|---|---|---|---|---|---|---|---|
| var4 | 0.84 | -2.67e-02 | -6.60e-03 | ✓ | -1.55e-01 | -9.36e-02 | ✓ |
| var8 | 0.36 | +4.68e-01 | +7.27e-01 | ✓ | +2.67e-01 | +3.86e-01 | ✓ |
| threshold4 | 0.80 | +2.45e-02 | +2.18e-02 | ✓ | +2.83e-02 | +2.85e-02 | ✓ |
| lorenz96 | 0.25 | -3.53e-02 | -2.84e-02 | ✓ | +1.32e-02 | +1.46e-02 | ✓ |
| kuramoto | 0.41 | -6.96e-04 | -7.41e-04 | ✓ | -5.35e-06 | -9.43e-06 | ✓ |
| netsim | 0.59 | -2.75e-04 | +1.27e-03 | ✗ | +2.69e-03 | +2.97e-03 | ✓ |
| climate | — | -7.08e-03 | -4.39e-02 | ✓ | +1.68e-02 | -2.92e-02 | ✗ |
| climate_ext | — | -3.22e-02 | -3.97e-02 | ✓ | +3.90e-03 | -4.11e-02 | ✗ |

**Correct ex-ante calls: 13/16.**

## 2. If you pick the validation-best model, how bad is the regret?

Regret = (test MSE of val-pick − test MSE of oracle) / oracle. 0 = you picked the test-best model. `agree` = val-pick == test-best.

| Dataset | Val-pick (LSTM set) | Oracle (LSTM) | Regret | agree | Val-pick (all) | Oracle (all) | Regret | agree |
|---|---|---|---|---|---|---|---|---|
| var4 | LSTM Baseline | LSTM Baseline | +0.00 | ✓ | VAR | VAR | +0.00 | ✓ |
| var8 | LSTM Causal (PCMCI) | LSTM Causal (PCMCI) | +0.00 | ✓ | VAR | VAR | +0.00 | ✓ |
| threshold4 | LSTM Masked (PCMCI) | LSTM Masked (PCMCI) | +0.00 | ✓ | VAR | VAR | +0.00 | ✓ |
| lorenz96 | LSTM Masked (PCMCI) | LSTM Masked (PCMCI) | +0.00 | ✓ | LSTM Masked (PCMCI) | LSTM Masked (PCMCI) | +0.00 | ✓ |
| kuramoto | LSTM Baseline | LSTM Baseline | +0.00 | ✓ | VAR | VAR | +0.00 | ✓ |
| netsim | LSTM Masked (PCMCI) | LSTM Masked (PCMCI) | +0.00 | ✓ | VAR | VAR | +0.00 | ✓ |
| climate | LSTM Masked (PCMCI) | LSTM Baseline | +0.04 | ✗ | VAR | VAR | +0.00 | ✓ |
| climate_ext | LSTM Masked (PCMCI) | LSTM Baseline | +0.05 | ✗ | VAR | VAR | +0.00 | ✓ |

**Val-pick == test-best: LSTM set 6/8, all set 8/8. Median LSTM-set regret: +0.000.**

## 3. Random-graph control on validation (informative-graph signal)

Does the real PCMCI graph beat a random graph of the same density on **validation**? If yes on val, does it hold on test? A ground-truth-free detector of whether the graph carries signal.

| Dataset | Soft: PCMCI>rand (val) | (test) | Masked: PCMCI>rand (val) | (test) |
|---|---|---|---|---|
| var4 | ✓ | ✓ | — | — |
| var8 | ✓ | ✓ | — | — |
| threshold4 | ✓ | ✓ | — | — |
| lorenz96 | ✓ | ✓ | — | — |
| kuramoto | ✓ | ✓ | — | — |
| netsim | ✓ | ✓ | — | — |
| climate | ✓ | ✗ | — | — |
| climate_ext | ✗ | ✗ | — | — |

**Control agreement (val predicts test): 7/8.**

## Verdict

- Core ex-ante call (validation predicts whether causal beats baseline on test): correct **13/16** (81%).
- Val-best LSTM == test-best LSTM: **6/6** on stationary/simulated datasets, **0/2** on real climate.
- Adding VAR to the candidate set: val-best == test-best **8/8** (validation reliably identifies VAR where it wins).

**Read:** *mostly yes, with one honest caveat.*
- **On stationary / simulated data** (synthetic, chaotic, physiology): validation is a **perfect** ex-ante guide here — val-pick == test-best in 6/6, regret 0. You do not need to know F1 or the regime: train baseline + soft + masked, compare on a validation split, deploy the winner.
- **On real climate data**: validation **misfires** on the causal-vs-baseline call — it picked the masked causal model, but the plain LSTM was actually best on test (regret ~4–5%). The cause is non-stationarity: the validation period does not represent the test period, exactly where the paper's stationarity assumption is weakest. A single split is not enough; use walk-forward CV.
- **Two safety nets that work ex-ante:** (a) always include VAR in the candidate set — validation then picks the overall winner in 8/8 datasets; (b) the random-graph control on validation is a ground-truth-free red flag (on climate_ext it correctly warned, on the validation set, that the PCMCI graph does not beat a random graph).
