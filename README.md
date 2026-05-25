# Causal Regularization for Time Series Forecasting with LSTM Networks

Code and data for the paper:

> **Ensaio em Regularização Causal para Previsão de Séries Temporais com Redes LSTM**
> Lucas Rafael de Andrade (2026)

## Overview

This repository implements a hybrid methodology that integrates causal structure discovery (PCMCI via [Tigramite](https://github.com/jakobrunge/tigramite)) into LSTM training through two complementary mechanisms:

- **Soft penalty** — suppresses via autograd the input gradients associated with non-causal pairs, preserving the full historical window.
- **Hard masking** — zeroes out non-causal inputs at the lag level before LSTM processing, imposing the causal structure more aggressively.

A PAC-Bayes generalization bound for α-mixing processes (Alquier & Wintenberger, 2012) formalizes why a causally informed prior reduces the KL divergence and, therefore, the generalization gap — with the improvement being proportional to the quality of the estimated causal graph.

## Key Results

| Experiment | Best model | MAE improvement | DM statistic | p-value |
|---|---|---|---|---|
| Synthetic 4-var (F1=0.84) | Soft PCMCI | — | +4.97 | <10⁻⁶ |
| FRED macro (τ_max < L) | Soft PCMCI | −3.1% | +1.39 | 0.16 |
| NOAA climate (τ_max = L = 12) | Soft PCMCI | **−25.6%** | **+11.89** | <10⁻³⁰ |

The climate experiment (7 monthly teleconnection indices, T=900, 10-fold walk-forward CV) is the strongest result: the 95% CI of the causal model MSE [0.546, 0.720] does not overlap with the baseline CI [0.959, 1.271].

## Repository Structure

```
.
├── src/                        # Core library
│   ├── models.py               # CausalLSTM (soft penalty) and MaskedLSTM (hard masking)
│   ├── causal_discovery.py     # PCMCI wrapper and graph utilities
│   ├── train.py                # Training loop with early stopping
│   ├── evaluate.py             # Diebold-Mariano HLN test, bootstrap CI
│   ├── data.py                 # Data loading: FRED, NOAA, synthetic generators
│   ├── baselines.py            # Random Walk and VAR baselines
│   └── plotting.py             # Causal graph heatmaps and prediction plots
│
├── experiments/
│   ├── exp_synthetic.py        # Three synthetic scenarios (4-var, 8-var, threshold)
│   ├── exp_fred.py             # FRED macroeconomic case study
│   └── exp_climate.py          # NOAA teleconnection climate case study
│
├── data/
│   └── climate/                # Monthly NOAA teleconnection indices (1950–2024)
│       ├── NINO34.csv          # ENSO Niño 3.4
│       ├── PDO.csv             # Pacific Decadal Oscillation
│       ├── TSA.csv             # Tropical Southern Atlantic
│       ├── TNA.csv             # Tropical Northern Atlantic
│       ├── NAO.csv             # North Atlantic Oscillation
│       ├── AO.csv              # Arctic Oscillation
│       └── PNA.csv             # Pacific North American pattern
│
├── results/                    # Pre-computed results (CSV + PCMCI graphs)
│   ├── synthetic_results.csv
│   ├── synthetic_dm_tests.csv
│   ├── synthetic_graph_metrics.csv
│   ├── fred_results.csv
│   ├── fred_per_var.csv
│   ├── fred_sensitivity.csv
│   ├── climate_results.csv
│   ├── climate_dm_tests.csv
│   ├── climate_per_var.csv
│   ├── climate_pcmci_graphs.json
│   └── climate_ckpt/           # PCMCI graphs per fold (JSON, ~40–60 links each)
│       └── pcmci_fold{0-9}.json
│
├── main.tex                    # Full paper (LaTeX source)
├── run_all.py                  # Run all three experiments sequentially
├── requirements.txt
└── Causal_inference_time_series.ipynb   # Exploratory notebook
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, Tigramite ≥ 5.2.

> **Note:** FRED data is downloaded automatically via `pandas-datareader` on first run. NOAA climate data is bundled in `data/climate/`.

## Reproducing the Experiments

Run all three experiments sequentially (estimated time: ~4 h on 12-core CPU):

```bash
python run_all.py
```

Or run each experiment individually:

```bash
# Synthetic scenarios (fast, ~5 min)
python experiments/exp_synthetic.py

# FRED macroeconomic case study (~30 min)
python experiments/exp_fred.py

# NOAA climate case study (~3.5 h, 230 jobs on 12 cores)
python experiments/exp_climate.py
```

The climate experiment caches PCMCI graphs in `results/climate_ckpt/pcmci_fold*.json` and training results in `results/climate_ckpt/all_results.pkl`. If the training checkpoint exists, re-running the script skips the ~3.5 h parallel training and only recomputes the aggregated metrics.

## Practical Decision Criteria

Based on the experimental findings:

| Condition | Recommendation |
|---|---|
| τ_max < L | **Soft penalty** — hard masking truncates the historical window |
| τ_max ≈ L, non-linear regime | **Hard masking** — more robust when gradients are unstable |
| τ_max ≈ L, linear regime, high-quality graph (F1 ≥ 0.75) | Either mechanism; soft penalty slightly better |
| F1 < 0.40 or N > 15 | Neither — use standard LSTM regularization (dropout, L2) |

## Citation

If you use this code, please cite the paper (BibTeX will be updated upon publication):

```bibtex
@article{andrade2026causal,
  title  = {Ensaio em Regulariza\c{c}\~ao Causal para Previs\~ao de S\'eries
             Temporais com Redes {LSTM}},
  author = {Andrade, Lucas Rafael de},
  year   = {2026},
  note   = {Preprint}
}
```

## Data Sources

- **FRED** macroeconomic data: [Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org/) (public, via `pandas-datareader`)
- **NOAA** teleconnection indices: [NOAA Physical Sciences Laboratory](https://psl.noaa.gov/) and [NOAA Climate Prediction Center](https://www.cpc.ncep.noaa.gov/) (public domain)

## License

MIT License — see [LICENSE](LICENSE) for details.
