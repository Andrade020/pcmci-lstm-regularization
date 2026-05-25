#!/usr/bin/env python3
"""
experiments/exp_climate.py

Experimento com índices de teleconexão climática mensais (NOAA, 1950-2024).

Protocolo
---------
- 8 índices: NINO34, PDO, AMO, NAO, AO, PNA, QBO30, TSA
- Walk-forward CV com 10 folds (treino expansivo, 1 ano de teste por fold)
- PCMCI (ParCorr) rodado por fold sobre dados de treino (sem vazamento)
- Modelos: Baseline | Causal PCMCI (lambda search) | Causal Random | Masked PCMCI | Masked Random
- lambda ∈ {1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0, 5.0}  — selecionado por val-MSE
- 12 núcleos em paralelo (joblib Loky)
- DM-HLN e bootstrap por blocos (bloco=12, B=500)

Saída
-----
  results/climate_results.csv
  results/climate_dm_tests.csv
  results/climate_per_var.csv
  results/climate_pcmci_graphs.json
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, sys, json, time, warnings, hashlib
from pathlib import Path
from io import StringIO

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from scipy import stats
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from src.models import BaselineLSTM, CausalLSTM, MaskedLSTM
from src.data import TimeSeriesDataset
from src.train import fit, predict_one_step

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÃO GLOBAL
# ══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    window    = 12,      # L = τ_max  (critério τ_max ≈ L para máscara)
    tau_max   = 12,
    hidden    = 64,
    n_layers  = 2,
    dropout   = 0.1,
    batch     = 32,
    epochs    = 300,
    patience  = 30,
    n_jobs    = 12,
    # busca extensiva de lambda
    lambdas   = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0, 5.0],
    val_size  = 24,      # 2 anos de validação
    test_size = 12,      # 1 ano de teste por fold
    n_folds   = 10,
    min_train = 180,     # 15 anos mínimo de treino
    alpha_pc  = 0.05,    # nível de significância do PCMCI
    block     = 12,      # bloco para bootstrap (ciclo sazonal)
    n_boot    = 500,
    seed      = 42,
    lr        = 1e-3,
)

CACHE   = ROOT / "data" / "climate"
RESULTS = ROOT / "results"
CKPT    = RESULTS / "climate_ckpt"
for _d in [CACHE, RESULTS, CKPT]:
    _d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

# ══════════════════════════════════════════════════════════════════════════════
#  DEFINIÇÃO DOS ÍNDICES CLIMÁTICOS
# ══════════════════════════════════════════════════════════════════════════════
INDICES = {
    # ── ENSO e Pacífico ─────────────────────────────────────────────────────
    "NINO34": dict(
        url="https://psl.noaa.gov/data/correlation/nina34.data",
        fmt="psl", desc="Niño 3.4 SST anomaly (ENSO)"),
    "PDO": dict(
        url="https://psl.noaa.gov/data/correlation/pdo.data",
        fmt="psl", desc="Pacific Decadal Oscillation"),
    "TSA": dict(
        url="https://psl.noaa.gov/data/correlation/tsa.data",
        fmt="psl", desc="Tropical Southern Atlantic SST"),
    # ── Atlântico (TNA cobre 1948-presente; substitui AMO que está truncado) ─
    "TNA": dict(
        url="https://psl.noaa.gov/data/correlation/tna.data",
        fmt="psl", desc="Tropical Northern Atlantic SST"),
    # ── Padrões atmosféricos ────────────────────────────────────────────────
    "NAO": dict(
        url="https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii",
        fmt="cpc", desc="North Atlantic Oscillation"),
    "AO": dict(
        url="https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table",
        fmt="cpc", desc="Arctic Oscillation"),
    "PNA": dict(
        url="https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.pna.monthly.b5001.current.ascii",
        fmt="cpc", desc="Pacific North American pattern"),
    # ── EP/NP cobre 1950-presente (substitui QBO30 que começa em 1979) ──────
    "EPNP": dict(
        url="https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.epnp.monthly.b5001.current.ascii",
        fmt="cpc", desc="East Pacific / North Pacific teleconnection"),
}

MISSING = {-99.99, -999.0, 99.99, 999.0, -9.99, 9.99}


# ══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD E PARSE
# ══════════════════════════════════════════════════════════════════════════════

def _is_missing(v):
    return any(abs(v - m) < 0.05 for m in MISSING) or abs(v) > 900


def _parse_psl(text, name):
    rows = []
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) == 13:
            try:
                year = int(parts[0])
                if not (1900 <= year <= 2100):
                    continue
                for mo, v in enumerate([float(x) for x in parts[1:]], 1):
                    if not _is_missing(v):
                        rows.append({"date": pd.Timestamp(year, mo, 1), name: v})
            except (ValueError, OverflowError):
                continue
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df[~df.index.duplicated(keep="first")]


def _parse_cpc(text, name):
    rows = []
    for line in text.strip().split("\n"):
        parts = line.strip().split()
        try:
            if len(parts) == 13 and 1900 <= int(parts[0]) <= 2100:
                year = int(parts[0])
                for mo, v in enumerate([float(x) for x in parts[1:]], 1):
                    if not _is_missing(v):
                        rows.append({"date": pd.Timestamp(year, mo, 1), name: v})
            elif len(parts) == 3:
                year, mo, v = int(parts[0]), int(parts[1]), float(parts[2])
                if 1900 <= year <= 2100 and 1 <= mo <= 12 and not _is_missing(v):
                    rows.append({"date": pd.Timestamp(year, mo, 1), name: v})
        except (ValueError, OverflowError):
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df[~df.index.duplicated(keep="first")]


def download_index(name, url, fmt, timeout=45):
    cache_file = CACHE / f"{name}.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"  [{name}] cache local ({len(df)} obs, {df.index[0].year}-{df.index[-1].year})")
        return df

    print(f"  [{name}] baixando...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        text = r.text
    except Exception as e:
        print(f"FALHOU ({e})")
        return None

    df = _parse_psl(text, name) if fmt == "psl" else _parse_cpc(text, name)
    if df is None or len(df) < 100:
        print("parse falhou")
        return None

    df.to_csv(cache_file)
    print(f"OK ({len(df)} obs, {df.index[0].year}-{df.index[-1].year})")
    return df


def load_climate_data():
    """Baixa e alinha todos os índices. Retorna DataFrame com as séries disponíveis."""
    print("\n=== Carregando índices climáticos ===")
    frames = {}
    for name, info in INDICES.items():
        df = download_index(name, info["url"], info["fmt"])
        if df is not None:
            frames[name] = df[name]

    if len(frames) < 4:
        raise RuntimeError(f"Apenas {len(frames)} índices disponíveis; mínimo é 4.")

    panel = pd.DataFrame(frames)
    panel = panel.loc["1950-01-01":"2024-12-01"].dropna()
    print(f"\nDados alinhados: {len(panel)} meses ({panel.index[0].date()}-{panel.index[-1].date()})")
    print(f"Variáveis disponíveis ({len(panel.columns)}): {list(panel.columns)}")
    return panel


# ══════════════════════════════════════════════════════════════════════════════
#  GERAÇÃO DE FOLDS WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════════

def make_folds(n_total, min_train, val_size, test_size, n_folds):
    """
    Retorna lista de (train_end, val_end, test_end) com janela expansiva.
    test_end é o índice exclusive do fim do conjunto de teste.
    """
    # Primeira posição possível para o fim do primeiro teste
    first_test_end = min_train + val_size + test_size
    # Última posição possível
    last_test_end = n_total
    # Posições de test_end igualmente espaçadas
    endpoints = np.linspace(first_test_end, last_test_end, n_folds, dtype=int)

    folds = []
    for test_end in endpoints:
        test_start = test_end - test_size
        val_end    = test_start
        val_start  = val_end - val_size
        train_end  = val_start
        if train_end < min_train:
            continue
        folds.append((int(train_end), int(val_end), int(test_end)))
    return folds


# ══════════════════════════════════════════════════════════════════════════════
#  PREPARAÇÃO DE DADOS POR FOLD
# ══════════════════════════════════════════════════════════════════════════════

def prepare_fold(data_arr, train_end, val_end, test_end, window):
    """
    Padroniza e recorta os dados de um fold.

    Retorna:
        train_sc : array para treino (usado no PCMCI e no dataset de treino)
        val_sc   : array para val (inclui buffer de 'window' amostras)
        test_sc  : array para teste (inclui buffer de 'window' amostras)
        scaler   : StandardScaler ajustado no treino
    """
    scaler = StandardScaler()
    scaler.fit(data_arr[:train_end])
    scaled = scaler.transform(data_arr)

    train_sc = scaled[:train_end]
    val_sc   = scaled[train_end - window : val_end]   # buffer para janela inicial
    test_sc  = scaled[val_end - window : test_end]     # buffer para janela inicial

    return train_sc, val_sc, test_sc, scaler


# ══════════════════════════════════════════════════════════════════════════════
#  PCMCI POR FOLD
# ══════════════════════════════════════════════════════════════════════════════

def run_pcmci(train_arr, var_names, tau_max, alpha_pc, fold_id):
    """Executa PCMCI sobre dados de treino. Retorna causal_links (dict j->[(i,lag)])."""
    ckpt_file = CKPT / f"pcmci_fold{fold_id}.json"
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            raw = json.load(f)
        # JSON keys são strings; converter para int
        links = {int(k): [(i, lag) for i, lag in v] for k, v in raw.items()}
        n_links = sum(len(v) for v in links.values())
        print(f"  [fold {fold_id}] PCMCI cache ({n_links} links)")
        return links

    t0 = time.time()
    dataframe = pp.DataFrame(train_arr, var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe,
                  cond_ind_test=ParCorr(significance="analytic"),
                  verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha_pc)

    p_mat  = results["p_matrix"]   # (N, N, tau_max+1)
    causal_links = {}
    N = len(var_names)
    for j in range(N):
        links = []
        for i in range(N):
            for tau in range(1, tau_max + 1):
                if p_mat[i, j, tau] <= alpha_pc:
                    links.append((i, tau))
        causal_links[j] = links

    n_links = sum(len(v) for v in causal_links.values())
    elapsed = time.time() - t0
    print(f"  [fold {fold_id}] PCMCI: {n_links} links em {elapsed:.0f}s")

    with open(ckpt_file, "w") as f:
        json.dump({k: v for k, v in causal_links.items()}, f)
    return causal_links


def random_links_same_density(causal_links, n_vars, tau_max, seed):
    """Gera grafo aleatório com mesma densidade que causal_links."""
    rng = np.random.default_rng(seed)
    n_links_total = sum(len(v) for v in causal_links.values())
    if n_links_total == 0:
        n_links_total = n_vars  # fallback
    rand_links = {j: [] for j in range(n_vars)}
    placed = 0
    while placed < n_links_total:
        j = int(rng.integers(0, n_vars))
        i = int(rng.integers(0, n_vars))
        tau = int(rng.integers(1, tau_max + 1))
        if (i, tau) not in rand_links[j]:
            rand_links[j].append((i, tau))
            placed += 1
    return rand_links


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER DE TREINAMENTO (roda em processo separado via joblib)
# ══════════════════════════════════════════════════════════════════════════════

def _train_one(job):
    """
    Treina um modelo em um fold. Cada invocação cria o modelo do zero.

    job: dict com todas as configurações necessárias.
    Retorna: dict com resultados.
    """
    fold_id    = job["fold_id"]
    model_name = job["model_name"]
    lam        = job["lambda"]
    train_sc   = job["train_sc"]
    val_sc     = job["val_sc"]
    test_sc    = job["test_sc"]
    links      = job["links"]
    n_vars     = job["n_vars"]
    cfg        = job["cfg"]

    # Semente por worker para reproducibilidade
    worker_seed = cfg["seed"] + fold_id * 100 + hash(model_name) % 97
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)

    W  = cfg["window"]
    H  = cfg["hidden"]
    NL = cfg["n_layers"]
    D  = cfg["dropout"]

    # DataLoaders
    train_ds = TimeSeriesDataset(train_sc, W)
    val_ds   = TimeSeriesDataset(val_sc,   W)
    test_ds  = TimeSeriesDataset(test_sc,  W)

    if len(train_ds) < cfg["batch"] or len(val_ds) < 1 or len(test_ds) < 1:
        return None  # fold muito curto, pular

    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch"], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch"], shuffle=False)

    # Construir modelo
    if model_name == "Baseline":
        model = BaselineLSTM(n_vars, H, NL, D)
    elif model_name in ("Causal PCMCI", "Causal Random"):
        model = CausalLSTM(n_vars, H, links, W,
                           lambda_reg=lam, num_layers=NL, dropout=D)
    elif model_name in ("Masked PCMCI", "Masked Random"):
        model = MaskedLSTM(n_vars, H, links, W, num_layers=NL, dropout=D)
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    device = "cpu"  # joblib workers não compartilham CUDA
    model = model.to(device)

    history = fit(
        model, train_loader, val_loader,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        device=device,
        verbose=False,
        patience=cfg["patience"],
    )

    val_mse = min(history["val_mse"]) if history["val_mse"] else float("inf")

    preds, targets = predict_one_step(model, test_loader, device)
    errors = preds - targets                           # (T_test, N)
    mse    = float(np.mean(errors ** 2))
    mae    = float(np.mean(np.abs(errors)))
    # Por variável
    mae_per_var = np.mean(np.abs(errors), axis=0).tolist()

    return dict(
        fold      = fold_id,
        model     = model_name,
        lam       = lam,
        val_mse   = val_mse,
        mse       = mse,
        mae       = mae,
        mae_pv    = mae_per_var,
        errors    = errors,   # (T_test, N)  — para DM test
        preds     = preds,
        targets   = targets,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SELECIONA MELHOR lambda POR FOLD (usando val_mse)
# ══════════════════════════════════════════════════════════════════════════════

def best_lambda(results_list, model_name, fold_id):
    """Entre todos os resultados de um (model_name, fold_id), retorna o de menor val_mse."""
    candidates = [r for r in results_list
                  if r is not None and r["model"] == model_name and r["fold"] == fold_id]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r["val_mse"])


# ══════════════════════════════════════════════════════════════════════════════
#  ESTATÍSTICAS
# ══════════════════════════════════════════════════════════════════════════════

def dm_test_hln(e1, e2, h=1):
    """
    Teste Diebold-Mariano com correcao Harvey-Leybourne-Newbold (1997).
    e1, e2: erros de previsao (1D); positivo = modelo 2 é melhor.
    """
    d = (e1 ** 2) - (e2 ** 2)
    T = len(d)
    if T < 4:
        return 0.0, 1.0
    d_bar   = np.mean(d)
    gamma0  = np.mean((d - d_bar) ** 2)
    if gamma0 < 1e-15:
        return 0.0, 1.0
    DM      = d_bar / np.sqrt(gamma0 / T)
    k       = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    DM_hln  = DM / k if k > 0 else DM
    p_val   = 2.0 * stats.norm.sf(abs(DM_hln))
    return float(DM_hln), float(p_val)


def block_bootstrap_ci(errors_sq, block=12, n_boot=500, seed=0):
    """Intervalo de confiança 95% para o MSE via bootstrap por blocos."""
    rng = np.random.default_rng(seed)
    T = len(errors_sq)
    n_blocks = int(np.ceil(T / block))
    boot_mses = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, T - block + 1), size=n_blocks)
        sample = np.concatenate([errors_sq[s : s + block] for s in starts])[:T]
        boot_mses.append(float(np.mean(sample)))
    return tuple(np.percentile(boot_mses, [2.5, 97.5]))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 70)
    print("EXPERIMENTO CLIMÁTICO — TELECONEXÕES MENSAIS (NOAA)")
    print("=" * 70)

    # ── 1. Dados ──────────────────────────────────────────────────────────────
    panel = load_climate_data()
    var_names = list(panel.columns)
    N = len(var_names)
    data_arr  = panel.values.astype(np.float32)  # (T, N)
    T_total   = len(data_arr)
    print(f"\nN={N} variáveis, T={T_total} meses: {var_names}")

    # ── 2. Folds ──────────────────────────────────────────────────────────────
    folds = make_folds(
        T_total,
        CFG["min_train"],
        CFG["val_size"],
        CFG["test_size"],
        CFG["n_folds"],
    )
    print(f"\n{len(folds)} folds gerados:")
    for fid, (te, ve, se) in enumerate(folds):
        yr_tr = panel.index[te - 1].year
        yr_va = panel.index[ve - 1].year
        yr_te = panel.index[se - 1].year
        print(f"  fold {fid}: treino até {yr_tr} | val até {yr_va} | teste até {yr_te}"
              f"  (T_train={te}, T_val={ve-te}, T_test={se-ve})")

    # ── 3. PCMCI por fold (sequencial para evitar deadlocks com tigramite) ────
    print("\n=== PCMCI por fold ===")
    pcmci_results = {}
    graph_summary = {}
    for fid, (train_end, val_end, test_end) in enumerate(folds):
        train_sc, val_sc, test_sc, scaler = prepare_fold(
            data_arr, train_end, val_end, test_end, CFG["window"]
        )
        est_links = run_pcmci(
            train_sc, var_names, CFG["tau_max"], CFG["alpha_pc"], fid
        )
        rand_links = random_links_same_density(
            est_links, N, CFG["tau_max"], CFG["seed"] + fid
        )
        n_est  = sum(len(v) for v in est_links.values())
        n_rand = sum(len(v) for v in rand_links.values())
        pcmci_results[fid] = dict(
            est_links  = est_links,
            rand_links = rand_links,
            train_sc   = train_sc,
            val_sc     = val_sc,
            test_sc    = test_sc,
        )
        graph_summary[fid] = dict(n_links_pcmci=n_est, n_links_random=n_rand)

    # salva resumo dos grafos
    with open(RESULTS / "climate_pcmci_graphs.json", "w") as f:
        json.dump(graph_summary, f, indent=2)

    # ── 4. Montar lista de jobs ───────────────────────────────────────────────
    print("\n=== Montando jobs de treinamento ===")
    jobs = []
    for fid, (train_end, val_end, test_end) in enumerate(folds):
        fd = pcmci_results[fid]
        base_job = dict(
            fold_id  = fid,
            n_vars   = N,
            cfg      = CFG,
            train_sc = fd["train_sc"],
            val_sc   = fd["val_sc"],
            test_sc  = fd["test_sc"],
        )

        # Baseline (lambda=0, irrelevante)
        jobs.append({**base_job, "model_name": "Baseline",
                     "lambda": 0.0, "links": {}})

        # Masked PCMCI e Masked Random (sem busca de lambda)
        jobs.append({**base_job, "model_name": "Masked PCMCI",
                     "lambda": 0.0, "links": fd["est_links"]})
        jobs.append({**base_job, "model_name": "Masked Random",
                     "lambda": 0.0, "links": fd["rand_links"]})

        # Causal PCMCI e Causal Random — busca extensiva de lambda
        for lam in CFG["lambdas"]:
            jobs.append({**base_job, "model_name": "Causal PCMCI",
                         "lambda": lam, "links": fd["est_links"]})
            jobs.append({**base_job, "model_name": "Causal Random",
                         "lambda": lam, "links": fd["rand_links"]})

    print(f"Total de jobs: {len(jobs)}")
    print(f"Usando {CFG['n_jobs']} núcleos — tempo estimado: "
          f"{len(jobs) * 2.5 / CFG['n_jobs']:.0f}-"
          f"{len(jobs) * 4.0 / CFG['n_jobs']:.0f} min")

    # ── 5. Treino paralelo ────────────────────────────────────────────────────
    import pickle as _pkl
    _ckpt_results = CKPT / "all_results.pkl"
    if _ckpt_results.exists():
        print(f"\n[checkpoint] Carregando resultados de {_ckpt_results}")
        with open(_ckpt_results, "rb") as _f:
            all_results = _pkl.load(_f)
        print(f"[checkpoint] {len(all_results)} resultados carregados — pulando treinamento")
    else:
        print("\n=== Treinamento paralelo ===")
        t_train = time.time()
        all_results = Parallel(n_jobs=CFG["n_jobs"], backend="loky", verbose=5)(
            delayed(_train_one)(job) for job in jobs
        )
        all_results = [r for r in all_results if r is not None]
        elapsed = (time.time() - t_train) / 60
        print(f"\nTreinamento concluido em {elapsed:.1f} min "
              f"({len(all_results)}/{len(jobs)} jobs bem-sucedidos)")
        with open(_ckpt_results, "wb") as _f:
            _pkl.dump(all_results, _f)
        print(f"[checkpoint] Resultados salvos em {_ckpt_results}")

    # ── 6. Selecionar melhor lambda por fold (para Causal PCMCI e Causal Random) ──
    print("\n=== Selecionando melhor lambda por fold ===")
    model_names = ["Baseline", "Causal PCMCI", "Causal Random",
                   "Masked PCMCI", "Masked Random"]
    selected = []
    for fid in range(len(folds)):
        for mname in model_names:
            best = best_lambda(all_results, mname, fid)
            if best is not None:
                selected.append(best)
                if mname in ("Causal PCMCI", "Causal Random"):
                    print(f"  fold {fid} | {mname:18s} -> lambda={best['lam']:.0e}"
                          f"  val_mse={best['val_mse']:.4f}")

    # ── 7. Agregar erros por modelo (concatenar todos os folds) ───────────────
    print("\n=== Agregando resultados ===")
    agg = {}  # model_name -> list of error arrays (T_test, N)
    for r in selected:
        agg.setdefault(r["model"], []).append(r["errors"])

    # Concatenar folds
    agg_errors = {}
    for mname, errs in agg.items():
        if errs:
            agg_errors[mname] = np.concatenate(errs, axis=0)  # (T_total_test, N)

    T_test_total = len(agg_errors.get("Baseline", []))
    print(f"Amostras de teste totais por fold: {T_test_total}")

    # ── 8. Métricas globais + bootstrap CI ────────────────────────────────────
    print("\n=== Calculando métricas ===")
    summary_rows = []
    for mname in model_names:
        if mname not in agg_errors:
            continue
        e = agg_errors[mname]          # (T_test_total, N)
        e_flat = e.flatten()
        mse_val = float(np.mean(e_flat ** 2))
        mae_val = float(np.mean(np.abs(e_flat)))
        rmse_val = float(np.sqrt(mse_val))

        # Bootstrap CI
        ci_lo, ci_hi = block_bootstrap_ci(
            e_flat ** 2, block=CFG["block"], n_boot=CFG["n_boot"], seed=CFG["seed"]
        )

        # DM vs Baseline
        if mname != "Baseline" and "Baseline" in agg_errors:
            e_base = agg_errors["Baseline"].flatten()
            dm_stat, p_val = dm_test_hln(e_base, e_flat)
        else:
            dm_stat, p_val = float("nan"), float("nan")

        summary_rows.append(dict(
            model=mname, mse=mse_val, rmse=rmse_val, mae=mae_val,
            ci_lower=ci_lo, ci_upper=ci_hi,
            dm_stat=dm_stat, p_value=p_val,
        ))
        print(f"  {mname:20s}  MSE={mse_val:.4f}  MAE={mae_val:.5f}"
              f"  DM={dm_stat:+.2f}  p={p_val:.3f}")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(RESULTS / "climate_results.csv", index=False)

    # ── 9. DM tests detalhados (todas as combinações) ─────────────────────────
    dm_rows = []
    if "Baseline" in agg_errors:
        e_base = agg_errors["Baseline"].flatten()
        for mname in model_names:
            if mname == "Baseline" or mname not in agg_errors:
                continue
            e_alt = agg_errors[mname].flatten()
            dm_stat, p_val = dm_test_hln(e_base, e_alt)
            dm_rows.append(dict(model=mname, dm_stat=dm_stat, p_value=p_val))
    pd.DataFrame(dm_rows).to_csv(RESULTS / "climate_dm_tests.csv", index=False)

    # ── 10. Resultados por variável ───────────────────────────────────────────
    pv_rows = []
    for vi, vname in enumerate(var_names):
        row = {"variable": vname}
        for mname in model_names:
            if mname not in agg_errors:
                continue
            e_v = agg_errors[mname][:, vi]
            row[f"mae_{mname}"] = float(np.mean(np.abs(e_v)))
        pv_rows.append(row)
    df_pv = pd.DataFrame(pv_rows)
    # calcular % melhoria vs Baseline
    for mname in model_names:
        if mname == "Baseline" or f"mae_{mname}" not in df_pv.columns:
            continue
        col = f"impr_{mname}_pct"
        df_pv[col] = (
            (df_pv["mae_Baseline"] - df_pv[f"mae_{mname}"]) / df_pv["mae_Baseline"] * 100
        )
    df_pv.to_csv(RESULTS / "climate_per_var.csv", index=False)

    # ── 11. Relatório final ───────────────────────────────────────────────────
    total_min = (time.time() - t_start) / 60
    print("\n" + "=" * 70)
    print(f"EXPERIMENTO CONCLUÍDO em {total_min:.1f} min")
    print("=" * 70)
    print("\nResultados globais:")
    print(df_summary[["model", "mse", "mae", "dm_stat", "p_value"]].to_string(index=False))

    print("\nResultados por variável (melhoria % vs Baseline):")
    impr_cols = [c for c in df_pv.columns if c.startswith("impr_")]
    if impr_cols:
        print(df_pv[["variable"] + impr_cols].to_string(index=False))

    print(f"\nArquivos salvos em: {RESULTS}")
    print("  climate_results.csv   — métricas globais + DM + CI bootstrap")
    print("  climate_dm_tests.csv  — teste DM detalhado por modelo")
    print("  climate_per_var.csv   — MAE e melhoria por variável")
    print("  climate_pcmci_graphs.json — resumo dos grafos por fold")

    # ── 12. Verificação rápida: lambda ótimo por modelo ────────────────────────────
    print("\nMelhor lambda por fold e modelo:")
    for mname in ("Causal PCMCI", "Causal Random"):
        lams = []
        for fid in range(len(folds)):
            best = best_lambda(all_results, mname, fid)
            if best:
                lams.append(best["lam"])
        if lams:
            print(f"  {mname}: lambda ótimos = {[f'{l:.0e}' for l in lams]}"
                  f"  (mediana={np.median(lams):.0e})")

    return df_summary


if __name__ == "__main__":
    main()
