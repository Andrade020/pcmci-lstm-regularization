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

Pick an example dataset (or upload a wide CSV: a datetime/index column followed by
one numeric column per series — a `Download example CSV` link is provided). The
imported series are charted immediately. Set the parameters and run; training
happens in a background thread and the page polls for progress. Results are drawn
entirely with interactive SVG (no server-side images): a plain-language **verdict**
(is there causal structure? does the graph beat a random graph? did causality
help?), an **interactive causal graph** (hover a node to highlight its links),
per-variable **forecast overlays** (real vs each model, with crosshair tooltips),
and a **model-comparison table** with Diebold–Mariano significance. Colours follow
a colorblind-validated palette.

## Guarding against spurious graphs (common-cause confounding)

PCMCI assumes *causal sufficiency* — no unobserved common cause. Real panels often
violate it: blood-donation series co-move through the calendar (day-of-week,
campaigns); river gauges co-move through shared rainfall. PCMCI then paints a dense
"everything causes everything" graph of **spurious** links. The app now does three
things about this:

1. **Automatic confounding alarm.** After each run it checks three
   ground-truth-free red flags — a near-complete **graph density**, a **local peak
   in the seasonal autocorrelation** (a periodic driver; a plain AR/VAR does not
   trip it), and whether the **density-matched random graph matches PCMCI** (if it
   does, the structure is not the active ingredient — generic shrinkage is). When
   the signature appears, the verdict raises *"Suspeita de causa comum
   (confundidor)"* and names the reasons.

2. **Leakage-free deseasonalization.** A `deseason_period` control removes a
   periodic driver by subtracting a seasonal mean **estimated on the training rows
   only** (7 = weekly, 12 = monthly). On a synthetic series where four channels
   share one weekly driver, it collapses the graph density from **0.94 → 0.31** and
   turns the alarm off. It fixes *periodic* drivers; irregular ones (rainfall) need
   the next tool.

3. **PCMCI+ / LPCMCI.** A method selector. **LPCMCI** is built for latent
   confounders: it marks a shared-cause edge as bidirected (`<->`) instead of
   forcing a spurious directed link, and the app keeps only the genuine directed
   (`-->`) edges. On the real `rivers` set, switching from PCMCI to LPCMCI/PCMCI+
   cuts graph density from **0.58 → 0.25**; on direction-only recovery (the lag is
   unknown), **PCMCI+ gives the best F1 (0.47)** vs PCMCI's 0.37, by pruning the
   rainfall-driven false positives.

None of this turns the app into a magic causal-discovery product — LPCMCI is slower
and has its own assumptions, and period detection can be wrong. It is a *guided,
honest* workflow that surfaces when **not** to trust the graph, in the same spirit
as the paper's moderators.

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
| `rivers` | rivers | structural | **CausalRivers** (Stein et al. 2025): 6 real Elbe/Jahna gauge stations; upstream→downstream direction is physically certain (travel-time lag mapped to lag 1). |
| `climate` | climate | none | 7 bundled NOAA teleconnection indices. |
| `climate_ext` | climate | none | Bundled + downloadable extras (SOI/AMO/DMI). |

The `rivers` set is a genuine real-world benchmark with a *known* causal
direction, and — because every gauge in a basin shares the same rainfall — it also
carries the exact common-driver confounding the app now diagnoses (below). Its
127 MB archive is downloaded once and cached (gitignored) on first use.

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
    main.py       # FastAPI: async thread jobs, polling, JSON data + preview endpoints
    jobs.py       # thread-safe job registry (no joblib — loky+torch deadlocks on Windows)
    schemas.py    # pydantic request models
  frontend/       # zero-build HTML + vanilla JS (interactive SVG charts + causal graph) + CSS
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

