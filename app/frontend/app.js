"use strict";

const $ = (id) => document.getElementById(id);
let activeTab = "example";

// ── Tabs ───────────────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    activeTab = btn.dataset.tab;
    $("tab-example").classList.toggle("hidden", activeTab !== "example");
    $("tab-upload").classList.toggle("hidden", activeTab !== "upload");
  });
});

// ── Load example datasets ──────────────────────────────────────────────────
let datasetsMeta = {};
async function loadDatasets() {
  const res = await fetch("/api/datasets");
  const data = await res.json();
  const sel = $("dataset");
  sel.innerHTML = "";
  data.datasets.forEach((d) => {
    datasetsMeta[d.name] = d;
    const opt = document.createElement("option");
    opt.value = d.name;
    opt.textContent = `${d.name}  —  ${d.group}`;
    sel.appendChild(opt);
  });
  if (data.datasets.length) applyDatasetDefaults(sel.value);
}

function applyDatasetDefaults(name) {
  const d = datasetsMeta[name];
  if (!d) return;
  $("dataset-desc").textContent = d.description;
  const c = d.cfg || {};
  if (c.window) $("window").value = c.window;
  if (c.tau_max) $("tau_max").value = c.tau_max;
  if (c.alpha) $("alpha").value = c.alpha;
  if (c.hidden) $("hidden").value = c.hidden;
  if (c.epochs) $("epochs").value = c.epochs;
  if (c.num_layers) $("num_layers").value = c.num_layers;
}
$("dataset").addEventListener("change", (e) => applyDatasetDefaults(e.target.value));

// ── Run ────────────────────────────────────────────────────────────────────
function readConfig() {
  return {
    window: +$("window").value,
    tau_max: +$("tau_max").value,
    alpha: +$("alpha").value,
    pc_alpha: 0.1,
    hidden: +$("hidden").value,
    num_layers: +$("num_layers").value,
    epochs: +$("epochs").value,
    lr: 1e-3,
    batch: 64,
  };
}

$("run-btn").addEventListener("click", async () => {
  $("run-btn").disabled = true;
  $("results-panel").classList.add("hidden");
  showStatus("submitting…", 0);
  try {
    let jobId;
    if (activeTab === "example") {
      const res = await fetch("/api/run/example", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset: $("dataset").value, config: readConfig() }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      jobId = (await res.json()).job_id;
    } else {
      const file = $("csvfile").files[0];
      if (!file) throw new Error("Choose a CSV file first.");
      const fd = new FormData();
      fd.append("file", file);
      const c = readConfig();
      Object.entries(c).forEach(([k, v]) => fd.append(k, v));
      fd.append("transform", $("transform").value);
      const res = await fetch("/api/run/upload", { method: "POST", body: fd });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      jobId = (await res.json()).job_id;
    }
    await pollJob(jobId);
  } catch (err) {
    showStatus("error: " + err.message, 0);
  } finally {
    $("run-btn").disabled = false;
  }
});

function showStatus(msg, frac) {
  $("status").classList.remove("hidden");
  $("status-msg").textContent = msg;
  $("bar-fill").style.width = Math.round(frac * 100) + "%";
}

async function pollJob(jobId) {
  while (true) {
    await new Promise((r) => setTimeout(r, 1200));
    const res = await fetch(`/api/jobs/${jobId}`);
    const s = await res.json();
    showStatus(s.message, s.progress);
    if (s.status === "done") break;
    if (s.status === "error") throw new Error(s.error || "job failed");
  }
  const res = await fetch(`/api/jobs/${jobId}/result`);
  const payload = await res.json();
  renderResults(payload.result, payload.figures);
  showStatus("done", 1);
}

// ── Render ─────────────────────────────────────────────────────────────────
const ABLATIONS = new Set(["LSTM Causal (Random)", "LSTM Masked (Random)"]);

function fmt(x, e = false) {
  if (x === null || x === undefined) return "—";
  return e ? x.toExponential(3) : x.toFixed(3);
}

function renderResults(result, figures) {
  $("results-panel").classList.remove("hidden");
  $("results-meta").textContent =
    `${result.n_vars} vars · ${result.n_obs} obs · ${result.n_links} causal links · ` +
    `test=${result.splits.test} steps`;

  // Graph F1 box
  const f1box = $("f1-box");
  if (result.graph_f1) {
    const g = result.graph_f1;
    f1box.classList.remove("hidden");
    f1box.innerHTML =
      `<b>Graph recovery vs ground truth:</b> F1 = ${g.f1.toFixed(2)} ` +
      `(precision ${g.precision.toFixed(2)}, recall ${g.recall.toFixed(2)}; ` +
      `TP ${g.tp}, FP ${g.fp}, FN ${g.fn}). ` +
      `Per the paper, causal regularization helps most when F1 ≥ 0.75.`;
  } else {
    f1box.classList.add("hidden");
  }

  // Metrics table
  const models = result.models;
  const names = Object.keys(models);
  let bestMae = Infinity, bestName = null;
  names.forEach((n) => { if (models[n].mae < bestMae) { bestMae = models[n].mae; bestName = n; } });

  let html = "<thead><tr><th>Model</th><th>MSE</th><th>RMSE</th><th>MAE</th>" +
             "<th>DM</th><th>p</th></tr></thead><tbody>";
  names.forEach((n) => {
    const m = models[n];
    const cls = [];
    if (n === "LSTM Baseline") cls.push("baseline");
    if (n === bestName) cls.push("best");
    if (ABLATIONS.has(n)) cls.push("ablation");
    let dmCell = "—", pCell = "—";
    if (m.dm_stat !== null && m.dm_stat !== undefined) {
      const stars = m.p_value < 0.05 ? "**" : (m.p_value < 0.10 ? "*" : "");
      const wcls = m.dm_stat > 0 ? "win" : "lose";
      dmCell = `<span class="${wcls}">${m.dm_stat > 0 ? "+" : ""}${m.dm_stat.toFixed(2)}${stars}</span>`;
      pCell = m.p_value.toFixed(3);
    }
    html += `<tr class="${cls.join(" ")}"><td>${n}</td>` +
      `<td>${fmt(m.mse, true)}</td><td>${fmt(m.rmse, true)}</td>` +
      `<td>${fmt(m.mae, true)}</td><td>${dmCell}</td><td>${pCell}</td></tr>`;
  });
  html += "</tbody>";
  $("metrics-table").innerHTML = html;

  // Figures
  setImg("fig-heatmap", figures.graph_heatmap);
  setImg("fig-network", figures.graph_network);
  setImg("fig-forecast", figures.forecast);
  setImg("fig-history", figures.history);
}

function setImg(id, b64) {
  const el = $(id);
  if (b64) { el.src = "data:image/png;base64," + b64; el.classList.remove("hidden"); }
  else { el.classList.add("hidden"); }
}

loadDatasets();
