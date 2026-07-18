"""
Run the full causal-forecasting comparison across every registered dataset and
write an honest verdict.

For each dataset: run the pipeline (all models + random-graph ablations), record
graph-recovery F1 (where ground truth exists), and compute pairwise
Diebold–Mariano tests of each causal model against *each* strong baseline
(Random Walk, VAR, LSTM baseline, LSTM-L2) — not just against the LSTM baseline.

Run from project root:
    python -m app.benchmark.run_benchmark              # all datasets
    python -m app.benchmark.run_benchmark var4 netsim  # a subset
"""
import sys
import time

import numpy as np
import torch

from src.evaluate import diebold_mariano_test
from app.backend.pipeline import run_pipeline
from app.benchmark.datasets import DATASETS, load_dataset, get_config
from app.benchmark import report

CAUSAL_MODELS = ["LSTM Causal (PCMCI)", "LSTM Masked (PCMCI)"]
BASELINES = ["Random Walk", "VAR", "LSTM Baseline", "LSTM-L2"]

FULL_MODELS = ["rw", "var", "l2", "causal", "causal_random", "masked", "masked_random"]


def pairwise_dm(y_true, preds):
    """
    For each (causal_model, baseline) pair, DM test.

    diebold_mariano_test(e1, e2) > 0 means e1 has higher loss (model1 worse),
    so passing e1 = baseline errors, e2 = causal errors gives a positive stat
    when the causal model is better. `better` = positive stat AND p < 0.05.
    """
    out = {}
    for cm in CAUSAL_MODELS:
        if cm not in preds:
            continue
        e_causal = y_true - preds[cm]
        for bl in BASELINES:
            if bl not in preds:
                continue
            e_base = y_true - preds[bl]
            dm, p = diebold_mariano_test(e_base, e_causal)
            if np.isnan(dm):
                out[(cm, bl)] = (None, None, False)
            else:
                out[(cm, bl)] = (float(dm), float(p), bool(dm > 0 and p < 0.05))
    return out


def run_one(name: str) -> dict:
    print(f"\n{'='*66}\nDATASET: {name}\n{'='*66}")
    t0 = time.time()
    data, var_names, true_links = load_dataset(name)
    cfg = get_config(name)
    cfg["models"] = FULL_MODELS

    def prog(msg, frac):
        print(f"  [{frac:4.0%}] {msg}", flush=True)

    result = run_pipeline(data, var_names, cfg, true_links=true_links, progress=prog)

    y_true = result["_arrays"]["y_true"]
    preds = result["_arrays"]["preds"]
    pw = pairwise_dm(y_true, preds)

    # console summary
    f1 = result["graph_f1"]
    print(f"  graph F1: {'—' if f1 is None else round(f1['f1'], 3)}  "
          f"({result['n_links']} links)  |  {time.time()-t0:.0f}s")
    for cm in CAUSAL_MODELS:
        if cm not in preds:
            continue
        verdicts = [f"{bl}:{'W' if pw[(cm, bl)][2] else '·'}"
                    for bl in BASELINES if (cm, bl) in pw]
        print(f"  {cm:24s} vs baselines -> {'  '.join(verdicts)}")

    return {
        "dataset": name,
        "group": DATASETS[name]["group"],
        "graph_f1": result["graph_f1"],
        "models": result["models"],
        "pairwise": pw,
    }


def main(names):
    np.random.seed(42)
    torch.manual_seed(42)
    records = []
    for name in names:
        try:
            records.append(run_one(name))
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] {name}: {type(exc).__name__}: {exc}")
    if not records:
        print("No datasets completed.")
        return
    paths = report.write_all(records)
    print("\n" + "=" * 66)
    print("BENCHMARK COMPLETE")
    for k, v in paths.items():
        print(f"  {k:10s} -> {v}")

    # echo the verdict tail
    with open(paths["verdict"], encoding="utf-8") as f:
        text = f.read()
    print("\n----- verdict.md (overall) -----")
    print(text.split("## Overall verdict")[-1])


if __name__ == "__main__":
    requested = sys.argv[1:] or list(DATASETS.keys())
    unknown = [n for n in requested if n not in DATASETS]
    if unknown:
        print(f"Unknown datasets: {unknown}\nAvailable: {list(DATASETS.keys())}")
        sys.exit(1)
    main(requested)
