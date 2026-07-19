"use strict";
const $ = (id) => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
const SERIES = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];

function svg(tag, attrs = {}) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

// shared tooltip
const tip = document.createElement("div");
tip.className = "tooltip";
document.body.appendChild(tip);
function showTip(html, x, y) {
  tip.innerHTML = html;
  tip.style.opacity = "1";
  const pad = 14;
  let left = x + pad, top = y + pad;
  const r = tip.getBoundingClientRect();
  if (left + r.width > window.innerWidth) left = x - r.width - pad;
  if (top + r.height > window.innerHeight) top = y - r.height - pad;
  tip.style.left = left + "px";
  tip.style.top = top + "px";
}
function hideTip() { tip.style.opacity = "0"; }

const fmt = (v) => (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01))
  ? v.toExponential(2) : v.toFixed(3);

/* ── Line chart (multi-series, with crosshair + tooltip) ─────────────────── */
function lineChart(mount, series, opts = {}) {
  const height = opts.height || 150;
  const W = Math.max(220, mount.clientWidth || 300);
  const H = height;
  const m = { top: 10, right: 14, bottom: 20, left: 46 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const n = series[0].values.length;

  let lo = Infinity, hi = -Infinity;
  for (const s of series) for (const v of s.values) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;

  const X = (i) => m.left + (n <= 1 ? 0 : (i / (n - 1)) * iw);
  const Y = (v) => m.top + ih - ((v - lo) / (hi - lo)) * ih;

  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });

  // gridlines + y labels (min / mid / max)
  const g = svg("g", { class: "axis" });
  [lo, (lo + hi) / 2, hi].forEach((v) => {
    const y = Y(v);
    g.appendChild(svg("line", { class: "gridline", x1: m.left, y1: y, x2: W - m.right, y2: y }));
    const t = svg("text", { x: m.left - 6, y: y + 3, "text-anchor": "end" });
    t.textContent = fmt(v); g.appendChild(t);
  });
  // x labels (start / end)
  [[0, opts.xstart ?? 0], [n - 1, opts.xend ?? (n - 1)]].forEach(([i, lab], k) => {
    const t = svg("text", { x: X(i), y: H - 5, "text-anchor": k === 0 ? "start" : "end" });
    t.textContent = lab; g.appendChild(t);
  });
  root.appendChild(g);

  // paths
  for (const s of series) {
    let d = "";
    s.values.forEach((v, i) => { d += (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1); });
    const p = svg("path", { class: "series-path", d });
    p.style.stroke = s.color;
    if (s.dash) p.style.strokeDasharray = "4 3";
    if (s.width) p.style.strokeWidth = s.width;
    root.appendChild(p);
  }

  // hover layer
  const cross = svg("line", { class: "crosshair", y1: m.top, y2: m.top + ih, x1: 0, x2: 0 });
  cross.style.opacity = "0";
  root.appendChild(cross);
  const hit = svg("rect", { x: m.left, y: m.top, width: iw, height: ih, fill: "transparent" });
  hit.style.cursor = "crosshair";
  root.appendChild(hit);
  hit.addEventListener("mousemove", (ev) => {
    const rect = root.getBoundingClientRect();
    const mx = (ev.clientX - rect.left) * (W / rect.width);
    let i = Math.round(((mx - m.left) / iw) * (n - 1));
    i = Math.max(0, Math.min(n - 1, i));
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.style.opacity = "1";
    let rows = series.map((s) =>
      `<div style="display:flex;gap:8px;justify-content:space-between">
         <span><span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${s.color};margin-right:5px"></span>${s.label}</span>
         <b>${fmt(s.values[i])}</b></div>`).join("");
    showTip(`<div style="color:var(--muted);margin-bottom:3px">t = ${i}</div>${rows}`, ev.clientX, ev.clientY);
  });
  hit.addEventListener("mouseleave", () => { cross.style.opacity = "0"; hideTip(); });

  mount.appendChild(root);
  return root;
}

function miniChart(parent, title, series, opts) {
  const box = document.createElement("div");
  box.className = "mini";
  const h = document.createElement("div");
  h.className = "mini-title"; h.textContent = title;
  box.appendChild(h);
  const chartMount = document.createElement("div");
  box.appendChild(chartMount);
  parent.appendChild(box);
  lineChart(chartMount, series, opts);   // measured after append
}

/* ── Causal network (interactive SVG) ────────────────────────────────────── */
function renderGraph(mount, links, varNames) {
  mount.innerHTML = "";
  const N = varNames.length;
  // aggregate edge counts: count[i][j] = # lags i->j
  const count = {};
  let total = 0;
  for (const jStr in links) {
    const j = +jStr;
    for (const [i] of links[jStr]) {
      const key = i + ">" + j; count[key] = (count[key] || 0) + 1; total++;
    }
  }
  if (total === 0) {
    mount.innerHTML = `<div class="empty-graph">Nenhuma ligação causal significativa encontrada
      pelo PCMCI com os parâmetros atuais.</div>`;
    return;
  }

  const W = 460, H = 460, cx = W / 2, cy = H / 2, R = 150, nr = 22;
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "graphsvg" });
  const defs = svg("defs");
  const mk = svg("marker", { id: "arrow", markerWidth: 9, markerHeight: 9, refX: 8, refY: 3,
    orient: "auto", markerUnits: "strokeWidth" });
  const mkPath = svg("path", { d: "M0,0 L8,3 L0,6 Z" });
  mkPath.style.fill = "var(--accent)"; mkPath.style.fillOpacity = "0.55";
  mk.appendChild(mkPath); defs.appendChild(mk); root.appendChild(defs);

  const pos = varNames.map((_, i) => {
    const a = (i / N) * 2 * Math.PI - Math.PI / 2;
    return [cx + R * Math.cos(a), cy + R * Math.sin(a)];
  });

  const edgeEls = [];
  const edgesG = svg("g");
  for (const key in count) {
    const [i, j] = key.split(">").map(Number);
    const c = count[key];
    const w = 1.2 + 0.9 * c;
    if (i === j) {
      const [x, y] = pos[i];
      const dir = Math.atan2(y - cy, x - cx);
      const ox = Math.cos(dir), oy = Math.sin(dir);
      const lx = x + ox * nr, ly = y + oy * nr;
      const loop = svg("path", {
        class: "edge self",
        d: `M ${lx - oy * 8} ${ly + ox * 8} A 13 13 0 1 1 ${lx + oy * 8} ${ly - ox * 8}`,
      });
      loop.style.strokeWidth = w; loop.dataset.i = i; loop.dataset.j = j;
      edgesG.appendChild(loop); edgeEls.push(loop);
      continue;
    }
    const [x0, y0] = pos[i], [x1, y1] = pos[j];
    const mxp = (x0 + x1) / 2, myp = (y0 + y1) / 2;
    const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy) || 1;
    const curve = 0.16;   // consistent bend so i->j and j->i don't overlap
    const ctrlx = mxp - dy * curve, ctrly = myp + dx * curve;
    // trim endpoints to the node radius so arrows start/end at the circle edge
    const sx = x0 + Math.cos(Math.atan2(ctrly - y0, ctrlx - x0)) * nr;
    const sy = y0 + Math.sin(Math.atan2(ctrly - y0, ctrlx - x0)) * nr;
    const ex = x1 + Math.cos(Math.atan2(ctrly - y1, ctrlx - x1)) * (nr + 3);
    const ey = y1 + Math.sin(Math.atan2(ctrly - y1, ctrlx - x1)) * (nr + 3);
    const p = svg("path", { class: "edge", d: `M ${sx} ${sy} Q ${ctrlx} ${ctrly} ${ex} ${ey}`,
      "marker-end": "url(#arrow)" });
    p.style.strokeWidth = w; p.dataset.i = i; p.dataset.j = j;
    edgesG.appendChild(p); edgeEls.push(p);
  }
  root.appendChild(edgesG);

  // nodes
  const nodeEls = [];
  varNames.forEach((name, i) => {
    const [x, y] = pos[i];
    const gg = svg("g", { class: "node" }); gg.dataset.i = i;
    const c = svg("circle", { cx: x, cy: y, r: nr });
    const t = svg("text", { x, y });
    t.textContent = name.length > 6 ? name.slice(0, 6) : name;
    gg.appendChild(c); gg.appendChild(t); root.appendChild(gg); nodeEls.push(gg);

    gg.addEventListener("mouseenter", () => {
      mount.classList.add("dim");
      gg.classList.add("hot");
      edgeEls.forEach((e) => {
        if (+e.dataset.i === i || +e.dataset.j === i) {
          e.classList.add("hot");
          const other = +e.dataset.i === i ? +e.dataset.j : +e.dataset.i;
          nodeEls[other] && nodeEls[other].classList.add("hot");
        }
      });
    });
    gg.addEventListener("mouseleave", () => {
      mount.classList.remove("dim");
      nodeEls.forEach((nn) => nn.classList.remove("hot"));
      edgeEls.forEach((e) => e.classList.remove("hot"));
    });
  });

  mount.appendChild(root);
}

/* ── Verdict panel ───────────────────────────────────────────────────────── */
function renderVerdict(v) {
  const rows = [];
  rows.push(v.has_structure
    ? { cls: "good", icon: "✓", t: `Estrutura causal encontrada`,
        s: `O PCMCI identificou ${v.n_links} ligação(ões) causal(is) defasada(s).` }
    : { cls: "warn", icon: "!", t: "Nenhuma estrutura causal",
        s: "O PCMCI não achou ligações — os dados parecem sem causalidade defasada detectável." });

  if (v.graph_informative !== null) {
    rows.push(v.graph_informative
      ? { cls: "good", icon: "✓", t: "O grafo carrega sinal",
          s: "O grafo do PCMCI supera um grafo aleatório de mesma densidade — a estrutura é informativa." }
      : { cls: "bad", icon: "✗", t: "Grafo não supera o aleatório",
          s: "O grafo do PCMCI não bate um grafo aleatório: a estrutura causal pode não estar ajudando." });
  }

  // Automatic common-cause / confounder alarm
  if (v.confounding_suspected) {
    const why = (v.confounding_reasons || []).map((r) => "• " + r).join("<br>");
    const metName = { pcmci: "PCMCI", pcmci_plus: "PCMCI+", lpcmci: "LPCMCI" }[v.method] || v.method;
    const fix = v.method === "lpcmci"
      ? "Já usando LPCMCI. Se o grafo continua denso, tente também <b>dessazonalizar</b> (campo período) para remover o driver comum."
      : `Provável <b>causa comum não observada</b> (ex.: dia-da-semana, campanhas, chuva). Tente <b>dessazonalizar</b> (campo período) e/ou trocar o método para <b>LPCMCI</b>, feito para confundidor latente. Método atual: ${metName}.`;
    rows.push({ cls: "bad", icon: "⚠", t: "Suspeita de causa comum (confundidor)",
      s: `${why}<br><br>${fix}` });
  } else if (v.method && v.method !== "pcmci") {
    const metName = { pcmci_plus: "PCMCI+", lpcmci: "LPCMCI" }[v.method] || v.method;
    rows.push({ cls: "good", icon: "✓", t: `Descoberta com ${metName}`,
      s: v.method === "lpcmci"
        ? "LPCMCI mantém só as ligações direcionadas genuínas; arestas de confundidor latente foram descartadas."
        : "PCMCI+ considerou também ligações contemporâneas." });
  }
  if (v.deseason_period) {
    rows.push({ cls: "good", icon: "✓", t: `Sazonalidade removida (período ${v.deseason_period})`,
      s: "A média sazonal foi estimada só no treino e subtraída — sem vazamento." });
  }

  const causalHelps = v.soft_beats_baseline || v.masked_beats_baseline;
  rows.push(causalHelps
    ? { cls: "good", icon: "✓", t: "A causalidade melhorou a previsão",
        s: `A LSTM regularizada supera a baseline com significância (DM, p<0,05).` }
    : { cls: "warn", icon: "≈", t: "Sem ganho sobre a baseline",
        s: "A LSTM causal não supera a LSTM comum com significância neste conjunto." });

  if (v.graph_f1) {
    const f1 = v.graph_f1.f1;
    rows.push({ cls: f1 >= 0.75 ? "good" : "warn", icon: f1 >= 0.75 ? "✓" : "≈",
      t: `Qualidade do grafo: F1 = ${f1.toFixed(2)}`,
      s: `vs. grafo verdadeiro (quando conhecido). O artigo recomenda usar a regularização quando F1 ≥ 0,75.` });
  }

  rows.push({ cls: "good", icon: "★", t: `Melhor modelo: ${v.best_model}`,
    s: "menor MAE no conjunto de teste." });

  $("verdict").innerHTML = rows.map((r) =>
    `<div class="vrow ${r.cls}"><div class="vicon">${r.icon}</div>
       <div><div class="vtitle">${r.t}</div><div class="vsub">${r.s}</div></div></div>`).join("");
}

/* ── Forecast ────────────────────────────────────────────────────────────── */
let LAST = null;
function renderForecast(fc) {
  const modelNames = Object.keys(fc.models);
  // legend
  const legendItems = [`<span class="item"><span class="swatch truth"></span>Real</span>`]
    .concat(modelNames.map((mname, k) =>
      `<span class="item"><span class="swatch" style="background:${SERIES[k % SERIES.length]}"></span>${mname}</span>`));
  $("forecast-legend").innerHTML = legendItems.join("");

  const mount = $("forecast"); mount.innerHTML = "";
  fc.var_names.forEach((vn) => {
    const series = [{ label: "Real", color: "var(--ink)", values: fc.truth[vn], width: 2 }];
    modelNames.forEach((mname, k) => series.push({
      label: mname, color: SERIES[k % SERIES.length], values: fc.models[mname][vn], dash: true, width: 1.6,
    }));
    miniChart(mount, vn, series, { height: 150, xstart: 0, xend: fc.truth[vn].length - 1 });
  });
}

/* ── Metrics table ───────────────────────────────────────────────────────── */
const ABLATIONS = new Set(["LSTM Causal (Random)", "LSTM Masked (Random)"]);
function renderTable(models) {
  const names = Object.keys(models);
  let bestMae = Infinity, bestName = null;
  names.forEach((n) => { if (models[n].mae < bestMae) { bestMae = models[n].mae; bestName = n; } });
  let html = "<thead><tr><th>Modelo</th><th>MSE</th><th>RMSE</th><th>MAE</th><th>DM</th><th>p</th></tr></thead><tbody>";
  names.forEach((n) => {
    const m = models[n], cls = [];
    if (n === "LSTM Baseline") cls.push("baseline");
    if (n === bestName) cls.push("best");
    let dmCell = "—", pCell = "—";
    if (m.dm_stat !== null && m.dm_stat !== undefined) {
      const stars = m.p_value < 0.05 ? "**" : (m.p_value < 0.10 ? "*" : "");
      dmCell = `<span class="${m.dm_stat > 0 ? "win" : "lose"}">${m.dm_stat > 0 ? "+" : ""}${m.dm_stat.toFixed(2)}${stars}</span>`;
      pCell = m.p_value.toFixed(3);
    }
    const label = ABLATIONS.has(n) ? n + " (aleatório)" : n;
    html += `<tr class="${cls.join(" ")}"><td>${label}</td><td>${m.mse.toExponential(2)}</td>` +
      `<td>${m.rmse.toExponential(2)}</td><td>${m.mae.toExponential(2)}</td><td>${dmCell}</td><td>${pCell}</td></tr>`;
  });
  $("metrics-table").innerHTML = html + "</tbody>";
}

function renderResults(result) {
  LAST = result;
  $("results").classList.remove("hidden");
  $("results-meta").textContent =
    `${result.n_vars} variáveis · ${result.n_obs} obs · ${result.n_links} ligações · teste=${result.splits.test} passos`;
  renderVerdict(result.verdict);
  renderGraph($("graph"), result.links, result.var_names);
  renderForecast(result.forecast);
  renderTable(result.models);
}

/* ── Data preview ────────────────────────────────────────────────────────── */
let PREVIEW = null;
function renderPreview(data) {
  PREVIEW = data;
  $("preview").classList.remove("hidden");
  $("preview-summary").textContent =
    `${data.n_vars} variáveis · ${data.n_obs} observações`;
  const mount = $("preview-chart"); mount.innerHTML = "";
  const grid = document.createElement("div"); grid.className = "multiples"; mount.appendChild(grid);
  data.var_names.forEach((vn) => {
    miniChart(grid, vn, [{ label: vn, color: "var(--s1)", values: data.values[vn], width: 1.8 }],
      { height: 120, xstart: data.index[0], xend: data.index[data.index.length - 1] });
  });
}

async function previewExample(name) {
  try {
    const r = await fetch(`/api/dataset/${name}/preview`);
    if (r.ok) renderPreview(await r.json());
  } catch (e) { /* ignore preview errors */ }
}

/* ── Wiring ──────────────────────────────────────────────────────────────── */
let activeTab = "example";
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    activeTab = btn.dataset.tab;
    $("tab-example").classList.toggle("hidden", activeTab !== "example");
    $("tab-upload").classList.toggle("hidden", activeTab !== "upload");
    $("preview").classList.add("hidden");
    if (activeTab === "example") previewExample($("dataset").value);
  });
});

let META = {};
async function loadDatasets() {
  const r = await fetch("/api/datasets");
  const data = await r.json();
  const sel = $("dataset"); sel.innerHTML = "";
  data.datasets.forEach((d) => {
    META[d.name] = d;
    const o = document.createElement("option");
    o.value = d.name; o.textContent = `${d.name} — ${d.group}`;
    sel.appendChild(o);
  });
  if (data.datasets.length) { applyDefaults(sel.value); previewExample(sel.value); }
}
function applyDefaults(name) {
  const d = META[name]; if (!d) return;
  $("dataset-desc").textContent = d.description;
  const c = d.cfg || {};
  ["window", "tau_max", "alpha", "hidden", "epochs", "num_layers"].forEach((k) => {
    if (c[k] !== undefined && $(k)) $(k).value = c[k];
  });
}
$("dataset").addEventListener("change", (e) => { applyDefaults(e.target.value); previewExample(e.target.value); });
$("csvfile").addEventListener("change", async (e) => {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  try {
    const r = await fetch("/api/preview", { method: "POST", body: fd });
    if (r.ok) renderPreview(await r.json());
    else showStatus("erro no preview: " + ((await r.json()).detail || r.status), 0);
  } catch (err) { showStatus("erro: " + err.message, 0); }
});

function readConfig() {
  return {
    window: +$("window").value, tau_max: +$("tau_max").value, alpha: +$("alpha").value,
    pc_alpha: 0.1, hidden: +$("hidden").value, num_layers: +$("num_layers").value,
    epochs: +$("epochs").value, lr: 1e-3, batch: 64,
    method: $("method").value, deseason_period: +$("deseason_period").value || 0,
  };
}
function showStatus(msg, frac) {
  $("status").classList.remove("hidden");
  $("status-msg").textContent = msg;
  $("bar-fill").style.width = Math.round(frac * 100) + "%";
}

$("run-btn").addEventListener("click", async () => {
  $("run-btn").disabled = true;
  $("results").classList.add("hidden");
  showStatus("enviando…", 0.02);
  try {
    let jobId;
    if (activeTab === "example") {
      const r = await fetch("/api/run/example", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset: $("dataset").value, config: readConfig() }),
      });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      jobId = (await r.json()).job_id;
    } else {
      const f = $("csvfile").files[0];
      if (!f) throw new Error("Escolha um CSV primeiro.");
      const fd = new FormData(); fd.append("file", f);
      Object.entries(readConfig()).forEach(([k, v]) => fd.append(k, v));
      fd.append("transform", $("transform").value);
      const r = await fetch("/api/run/upload", { method: "POST", body: fd });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      jobId = (await r.json()).job_id;
    }
    await poll(jobId);
  } catch (err) {
    showStatus("erro: " + err.message, 0);
  } finally { $("run-btn").disabled = false; }
});

async function poll(jobId) {
  while (true) {
    await new Promise((r) => setTimeout(r, 1200));
    const s = await (await fetch(`/api/jobs/${jobId}`)).json();
    showStatus(s.message, s.progress);
    if (s.status === "done") break;
    if (s.status === "error") throw new Error(s.error || "falhou");
  }
  const payload = await (await fetch(`/api/jobs/${jobId}/result`)).json();
  renderResults(payload.result);
  showStatus("concluído", 1);
}

// re-render charts on resize (SVG width is measured)
let rz;
window.addEventListener("resize", () => {
  clearTimeout(rz);
  rz = setTimeout(() => {
    if (PREVIEW) renderPreview(PREVIEW);
    if (LAST) { renderForecast(LAST.forecast); }
  }, 200);
});

loadDatasets();
