# Benchmark verdict — causal-regularized LSTM forecasting

Honest read of whether causal regularization (soft penalty / hard mask, PCMCI graph) beats **strong baselines** — Random Walk, VAR, LSTM baseline, LSTM-L2 — with Diebold–Mariano significance (p < 0.05, positive = causal better), across datasets.

## Per-dataset summary

| Dataset | Group | Graph F1 | Best model (MAE) | Soft beats all baselines? | Masked beats all baselines? |
|---|---|---|---|---|---|
| var4 | synthetic | 0.84 | VAR | no | no |
| var8 | synthetic | 0.36 | VAR | no | no |
| threshold4 | synthetic | 0.80 | VAR | no | no |
| lorenz96 | chaotic | 0.25 | LSTM Causal (PCMCI) | no | no |
| kuramoto | chaotic | 0.41 | VAR | no | no |
| netsim | physiology | 0.59 | VAR | no | no |
| climate | climate | — | VAR | no | no |
| climate_ext | climate | — | VAR | no | no |

## Pairwise DM (causal vs each baseline)

Positive DM = causal model better; ** p<0.05, * p<0.10.

### var4
| Causal model | vs baseline | DM | p | better? |
|---|---|---|---|---|
| LSTM Causal (PCMCI) | Random Walk | +2.72** | 0.007 | ✓ |
| LSTM Causal (PCMCI) | VAR | -9.78** | 0.000 | ✗ |
| LSTM Causal (PCMCI) | LSTM Baseline | -0.22 | 0.824 | ✗ |
| LSTM Causal (PCMCI) | LSTM-L2 | +0.44 | 0.657 | ✗ |
| LSTM Masked (PCMCI) | Random Walk | -0.47 | 0.635 | ✗ |
| LSTM Masked (PCMCI) | VAR | -11.60** | 0.000 | ✗ |
| LSTM Masked (PCMCI) | LSTM Baseline | -5.04** | 0.000 | ✗ |
| LSTM Masked (PCMCI) | LSTM-L2 | -4.97** | 0.000 | ✗ |

### var8
| Causal model | vs baseline | DM | p | better? |
|---|---|---|---|---|
| LSTM Causal (PCMCI) | Random Walk | -42.65** | 0.000 | ✗ |
| LSTM Causal (PCMCI) | VAR | -42.77** | 0.000 | ✗ |
| LSTM Causal (PCMCI) | LSTM Baseline | -31.54** | 0.000 | ✗ |
| LSTM Causal (PCMCI) | LSTM-L2 | -33.49** | 0.000 | ✗ |
| LSTM Masked (PCMCI) | Random Walk | -42.44** | 0.000 | ✗ |
| LSTM Masked (PCMCI) | VAR | -42.63** | 0.000 | ✗ |
| LSTM Masked (PCMCI) | LSTM Baseline | +40.10** | 0.000 | ✓ |
| LSTM Masked (PCMCI) | LSTM-L2 | +38.98** | 0.000 | ✓ |

### threshold4
| Causal model | vs baseline | DM | p | better? |
|---|---|---|---|---|
| LSTM Causal (PCMCI) | Random Walk | +3.15** | 0.002 | ✓ |
| LSTM Causal (PCMCI) | VAR | -7.29** | 0.000 | ✗ |
| LSTM Causal (PCMCI) | LSTM Baseline | -3.98** | 0.000 | ✗ |
| LSTM Causal (PCMCI) | LSTM-L2 | -4.30** | 0.000 | ✗ |
| LSTM Masked (PCMCI) | Random Walk | +4.57** | 0.000 | ✓ |
| LSTM Masked (PCMCI) | VAR | -5.45** | 0.000 | ✗ |
| LSTM Masked (PCMCI) | LSTM Baseline | +0.36 | 0.716 | ✗ |
| LSTM Masked (PCMCI) | LSTM-L2 | +0.05 | 0.961 | ✗ |

### lorenz96
| Causal model | vs baseline | DM | p | better? |
|---|---|---|---|---|
| LSTM Causal (PCMCI) | Random Walk | +23.88** | 0.000 | ✓ |
| LSTM Causal (PCMCI) | VAR | +16.79** | 0.000 | ✓ |
| LSTM Causal (PCMCI) | LSTM Baseline | +0.83 | 0.405 | ✗ |
| LSTM Causal (PCMCI) | LSTM-L2 | +1.24 | 0.216 | ✗ |
| LSTM Masked (PCMCI) | Random Walk | +23.88** | 0.000 | ✓ |
| LSTM Masked (PCMCI) | VAR | +16.91** | 0.000 | ✓ |
| LSTM Masked (PCMCI) | LSTM Baseline | +1.76* | 0.079 | ✗ |
| LSTM Masked (PCMCI) | LSTM-L2 | +2.12** | 0.034 | ✓ |

### kuramoto
| Causal model | vs baseline | DM | p | better? |
|---|---|---|---|---|
| LSTM Causal (PCMCI) | Random Walk | +61.90** | 0.000 | ✓ |
| LSTM Causal (PCMCI) | VAR | -32.28** | 0.000 | ✗ |
| LSTM Causal (PCMCI) | LSTM Baseline | +19.12** | 0.000 | ✓ |
| LSTM Causal (PCMCI) | LSTM-L2 | +44.28** | 0.000 | ✓ |
| LSTM Masked (PCMCI) | Random Walk | +62.29** | 0.000 | ✓ |
| LSTM Masked (PCMCI) | VAR | -31.66** | 0.000 | ✗ |
| LSTM Masked (PCMCI) | LSTM Baseline | +24.36** | 0.000 | ✓ |
| LSTM Masked (PCMCI) | LSTM-L2 | +47.67** | 0.000 | ✓ |

### netsim
| Causal model | vs baseline | DM | p | better? |
|---|---|---|---|---|
| LSTM Causal (PCMCI) | Random Walk | +3.55** | 0.000 | ✓ |
| LSTM Causal (PCMCI) | VAR | -7.22** | 0.000 | ✗ |
| LSTM Causal (PCMCI) | LSTM Baseline | -2.83** | 0.005 | ✗ |
| LSTM Causal (PCMCI) | LSTM-L2 | -3.16** | 0.002 | ✗ |
| LSTM Masked (PCMCI) | Random Walk | +3.85** | 0.000 | ✓ |
| LSTM Masked (PCMCI) | VAR | -7.02** | 0.000 | ✗ |
| LSTM Masked (PCMCI) | LSTM Baseline | -2.39** | 0.017 | ✗ |
| LSTM Masked (PCMCI) | LSTM-L2 | -2.81** | 0.005 | ✗ |

### climate
| Causal model | vs baseline | DM | p | better? |
|---|---|---|---|---|
| LSTM Causal (PCMCI) | Random Walk | +1.95* | 0.051 | ✗ |
| LSTM Causal (PCMCI) | VAR | -5.95** | 0.000 | ✗ |
| LSTM Causal (PCMCI) | LSTM Baseline | +1.20 | 0.232 | ✗ |
| LSTM Causal (PCMCI) | LSTM-L2 | +0.02 | 0.982 | ✗ |
| LSTM Masked (PCMCI) | Random Walk | +2.88** | 0.004 | ✓ |
| LSTM Masked (PCMCI) | VAR | -5.63** | 0.000 | ✗ |
| LSTM Masked (PCMCI) | LSTM Baseline | +3.03** | 0.003 | ✓ |
| LSTM Masked (PCMCI) | LSTM-L2 | +1.86* | 0.063 | ✗ |

### climate_ext
| Causal model | vs baseline | DM | p | better? |
|---|---|---|---|---|
| LSTM Causal (PCMCI) | Random Walk | +3.59** | 0.000 | ✓ |
| LSTM Causal (PCMCI) | VAR | -5.14** | 0.000 | ✗ |
| LSTM Causal (PCMCI) | LSTM Baseline | +4.62** | 0.000 | ✓ |
| LSTM Causal (PCMCI) | LSTM-L2 | +2.52** | 0.012 | ✓ |
| LSTM Masked (PCMCI) | Random Walk | +3.40** | 0.001 | ✓ |
| LSTM Masked (PCMCI) | VAR | -5.76** | 0.000 | ✗ |
| LSTM Masked (PCMCI) | LSTM Baseline | +3.64** | 0.000 | ✓ |
| LSTM Masked (PCMCI) | LSTM-L2 | +1.72* | 0.086 | ✗ |

## Overall verdict

**The bar (your standard): beat *all four* strong baselines (Random Walk, VAR, LSTM baseline, LSTM-L2) with significance.**

- Soft penalty clears that bar in **0/8** datasets.
- Hard mask clears that bar in **0/8** datasets.

**What actually happens:**

- **VAR is the wall:** in **7/8** datasets neither causal model beats a BIC-tuned VAR. VAR is the best or co-best model almost everywhere; the only place it breaks is the fully chaotic Lorenz-96, where the LSTM family wins but causal ≈ plain LSTM.
- **The mechanism is real, but intra-LSTM:** a causal model beats the LSTM baseline (the paper's own reference) in **4/8** datasets — strongly on kuramoto and climate_ext. So causal regularization does help the LSTM; it just doesn't lift it above the right classical baseline.
- **Caveat in the method's favour:** var4/var8/netsim are (M)VAR systems by construction, so VAR winning there is partly circular. But climate and kuramoto are not VAR and VAR still wins, so the caveat does not rescue the conclusion.

**Read:** this is an honest **negative/conditional** result. The artifact is a genuinely useful *LSTM regularizer* (and a reproducible way to see the paper's mechanism), **not** a forecaster that beats strong classical baselines. The paper's headline climate gain was measured against the LSTM baseline; add a tuned VAR and the advantage disappears. Publish for what it is — a mechanism study with an honest benchmark — not as a state-of-the-art forecaster.
