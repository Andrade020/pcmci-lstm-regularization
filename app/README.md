> **Honest headline (see full benchmark below):** across 8 datasets, the causal
> models beat *all four* strong baselines (Random Walk, VAR, LSTM baseline,
> LSTM-L2) in **0/8**. A BIC-tuned **VAR is the wall** (best or co-best in 7/8).
> Causal regularization *does* beat the LSTM baseline in **4/8** (strongly on
> kuramoto and climate_ext) — so the paper's mechanism is real, but it is an
> LSTM regularizer, **not** a forecaster that tops strong classical baselines.

# Causal Forecasting App

A plug-and-play front-end and an **honest benchmark harness** built on top of the
paper's `src/` library (PCMCI causal discovery via Tigramite + causal-regularized
LSTM forecasting). You feed in a multivariate time series; it discovers a lagged
causal graph, trains the causal-regularized LSTMs (soft penalty and hard mask),
and compares them against strong baselines with Diebold–Mariano significance.

This sub-app exists to answer one question without spin: **is the causal method
genuinely better than strong baselines, or only in the favourable regime the
paper already reported?** The `benchmark/` harness runs the comparison across
synthetic, chaotic, physiological, and climate datasets and writes a verdict.

## What it is (and is not)

- **Is:** a research artifact — a reproducible way to plug data in, see the
  causal graph and forecasts, and get a statistically honest model comparison.
- **Is not:** a general-purpose forecasting product. The paper shows causal
  regularization helps only when the estimated graph is good (F1 ≥ 0.75), the
  window is aligned to the max lag (τ_max ≈ L), and dimensionality is moderate
  (N ≤ ~15). The app surfaces those moderators rather than hiding them.

## Install

```bash
pip install -r requirements.txt -r app/requirements-app.txt
```

(The root `requirements.txt` provides torch, tigramite, pandas, numpy,
scikit-learn, statsmodels, matplotlib; `app/requirements-app.txt` adds FastAPI,
uvicorn, python-multipart.)

## Run the app

```bash
python -m uvicorn app.backend.main:app --reload
# open http://127.0.0.1:8000
```

Pick an example dataset (or upload a wide CSV: a datetime index column followed
by one numeric column per series), set the parameters, and run. Training happens
in a background thread; the page polls for progress and then shows the causal
graph (heatmap + network), the forecasts, the training history, and a model
comparison table with DM significance vs the LSTM baseline.

## Run the benchmark

```bash
python -m app.benchmark.run_benchmark            # all datasets
python -m app.benchmark.run_benchmark var4 netsim  # a subset
```

Outputs (committed under `app/benchmark/results/`):

- `benchmark_metrics.csv` — per-dataset, per-model MSE/RMSE/MAE + graph F1 + DM vs LSTM baseline.
- `benchmark_pairwise.csv` — DM of each causal model vs **each** strong baseline.
- `verdict.md` — the honest read (reproduced below).

## Datasets

| Name | Group | Ground-truth graph | Notes |
|---|---|---|---|
| `var4` | synthetic | exact | Linear VAR, favourable regime (F1≈0.84). |
| `var8` | synthetic | exact | Linear VAR(1), hard regime (F1≈0.36). |
| `threshold4` | synthetic | exact | Nonlinear threshold VAR. |
| `lorenz96` | chaotic | structural | Lorenz-96 ring; coupling topology as lag-1 truth. |
| `kuramoto` | chaotic | structural | Coupled oscillators; ring neighbours as truth. |
| `netsim` | physiology | exact | NetSim-style fMRI effective-connectivity MVAR with a known DAG. |
| `climate` | climate | none | 7 bundled NOAA teleconnection indices. |
| `climate_ext` | climate | none | Bundled + downloadable extras (SOI/AMO/DMI). |

Generated/downloaded data is cached under `app/datasets/` (gitignored). The
generators and the exact `var4`/`var8`/`threshold4` ground-truth links are
imported from `experiments/exp_synthetic.py` so the benchmark matches the paper.

For the chaotic systems, the "ground truth" is the coupling topology mapped to
lag 1 — an approximation, since a continuous chaotic flow has no exact
discrete-lag DAG. Read those F1 values as structural recovery, not exact scores.

## Architecture

```
app/
  backend/
    pipeline.py   # parametrized refactor of experiments/exp_fred.py::main (the one true pipeline)
    main.py       # FastAPI: async thread jobs, polling, static frontend
    jobs.py       # thread-safe job registry (no joblib — loky+torch deadlocks on Windows)
    figures.py    # matplotlib figures -> base64 PNG (reuses src/plotting)
    schemas.py    # pydantic request models
  frontend/       # zero-build HTML + vanilla JS + CSS
  benchmark/
    datasets.py       # registry: loader + ground-truth graph + recommended config
    synthetic.py      # Lorenz-96 / Kuramoto generators (+ reused exp_synthetic ones)
    simulate_physio.py# NetSim-style MVAR with a known DAG
    download_climate.py# extra NOAA indices with offline fallback
    run_benchmark.py  # runs the full comparison across datasets
    report.py         # writes benchmark CSVs + verdict.md
```

Both the app and the benchmark call the **same** `run_pipeline`, so there is no
duplicated modelling logic. Everything reuses `src/` (models, PCMCI wrapper,
training loop, evaluation, plotting) unchanged; the only core addition is
`load_wide_csv` + `stationarize` in `src/data.py`.

## Benchmark results (honest)

Full detail in [`benchmark/results/verdict.md`](benchmark/results/verdict.md) and
the two CSVs. Single chronological split per dataset; DM = Diebold–Mariano (HLN),
positive = causal better, ** p<0.05. `W` = the causal model beats that baseline
with p<0.05 (Causal / Masked).

| Dataset | Graph F1 | Best model | vs RW | vs **VAR** | vs LSTM base | vs LSTM-L2 |
|---|---|---|---|---|---|---|
| var4 | 0.84 | VAR | W / · | · / · | · / · | · / · |
| var8 | 0.36 | VAR | · / · | · / · | · / W | · / W |
| threshold4 | 0.80 | VAR | W / W | · / · | · / · | · / · |
| lorenz96 | 0.25 | Causal | W / W | **W / W** | · / · | · / W |
| kuramoto | 0.41 | VAR | W / W | · / · | W / W | W / W |
| netsim | 0.59 | VAR | W / W | · / · | · / · | · / · |
| climate | — | VAR | · / W | · / · | · / W | · / · |
| climate_ext | — | VAR | W / W | · / · | W / W | W / · |

**Bottom line.** The bar was *beat all four strong baselines with significance*.
The method clears it in **0/8** datasets. VAR wins almost everywhere; the causal
regularization helps only *within the LSTM family* (beats the plain LSTM in 4/8).
The paper's headline climate gain was measured against the LSTM baseline — add a
tuned VAR and it is beaten (DM −5.95 on climate). This is a legitimate, honest
**mechanism study**, not a state-of-the-art forecasting claim.

Reproduce: `python -m app.benchmark.run_benchmark` (regenerate just the verdict
from the committed CSVs with `python -m app.benchmark.report`).

## Ex-ante decidability

Follow-up question: *can you tell in advance — without ground truth — whether the
causal LSTM will help, using only a validation split?* `exante.py` scores every
model on both validation and test and checks whether the validation ranking
predicts the test ranking. Full detail in
[`benchmark/results/exante_report.md`](benchmark/results/exante_report.md).

| | Validation predicts test? |
|---|---|
| Core causal-vs-baseline call | correct **13/16** (81%) |
| Val-best LSTM == test-best (stationary/simulated) | **6/6**, regret 0 |
| Val-best LSTM == test-best (real climate) | **0/2**, regret ~4–5% |
| Val-best model incl. VAR == test-best | **8/8** |

**Answer: mostly yes, with a caveat.** On stationary/simulated data, validation
is a perfect ex-ante guide — pick the validation winner and you get the test
winner, no need to know F1. On **real climate** data it misfires on the
causal-vs-baseline call (non-stationarity: the validation period doesn't
represent the test period), so a single split is not enough there — use
walk-forward CV. Two ex-ante safety nets: always include VAR in the candidate set
(validation then nails the overall winner 8/8), and use the random-graph control
on validation as a ground-truth-free red flag.

Reproduce: `python -m app.benchmark.exante` (each dataset is cached to
`results/exante_cache.jsonl`, so the run resumes if interrupted). The committed
ex-ante numbers use lighter training settings than the headline benchmark
(fewer epochs, single λ) — they probe ranking stability, not absolute error.

