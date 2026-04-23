(() => {
  const $ = (id) => document.getElementById(id);

  const state = {
    catalog: null,
    currentStrategy: null,
    running: false,
    pending: false,
    dataOverview: null,
    selectedTicker: null,
    tickerFilter: "",
  };

  const plotLayout = {
    paper_bgcolor: "#11151c",
    plot_bgcolor: "#11151c",
    font: { color: "#e6ecf2", family: "ui-sans-serif, -apple-system, Segoe UI" },
    margin: { t: 16, r: 16, b: 36, l: 56 },
    xaxis: { gridcolor: "#1c2330", zerolinecolor: "#1c2330" },
    yaxis: { gridcolor: "#1c2330", zerolinecolor: "#1c2330" },
    legend: { orientation: "h", y: -0.18, x: 0 },
    hovermode: "x unified",
  };
  const plotConfig = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

  const fmtPct = (v, d = 2) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
  const fmtNum = (v, d = 2) => (v == null ? "—" : v.toFixed(d));
  const fmtX = (v, d = 2) => (v == null ? "—" : `${v.toFixed(d)}×`);

  function toast(msg, isError = false, ms = 3500) {
    const t = $("toast");
    t.textContent = msg;
    t.classList.toggle("error", isError);
    t.classList.remove("hidden");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => t.classList.add("hidden"), ms);
  }
  function setStatus(text, kind) {
    const el = $("status-bar");
    el.textContent = text;
    el.className = "statusbar" + (kind ? ` ${kind}` : "");
  }

  // ---------- Catalog & tabs ----------
  async function loadCatalog() {
    const res = await fetch("/api/catalog");
    if (!res.ok) throw new Error(`catalog HTTP ${res.status}`);
    return res.json();
  }
  function renderStrategyTabs() {
    const nav = $("strategy-tabs");
    nav.innerHTML = "";
    state.catalog.strategies.forEach((s, i) => {
      const btn = document.createElement("button");
      btn.textContent = s.label;
      btn.dataset.id = s.id;
      if (i === 0) { btn.classList.add("active"); state.currentStrategy = s; }
      btn.addEventListener("click", () => selectStrategy(s.id));
      nav.appendChild(btn);
    });
  }
  function selectStrategy(id) {
    const s = state.catalog.strategies.find((x) => x.id === id);
    if (!s) return;
    state.currentStrategy = s;
    document.querySelectorAll("#strategy-tabs button").forEach((b) => {
      b.classList.toggle("active", b.dataset.id === id);
    });
    $("strategy-summary").textContent = s.summary;
    renderStrategyParams();
    renderEngineParams();
    renderLiveCode();
    scheduleRun();
  }

  // ---------- Params ----------
  function renderStrategyParams() {
    const host = $("strategy-params");
    host.innerHTML = "";
    const s = state.currentStrategy;
    if (!s.params.length) {
      const empty = document.createElement("div");
      empty.style.cssText = "font-size: 11px; color: var(--muted); font-family: ui-monospace, Menlo, monospace;";
      empty.textContent = "# no tunable strategy parameters";
      host.appendChild(empty);
      return;
    }
    for (const p of s.params) host.appendChild(buildParamRow(p, `strat-${p.name}`));
  }
  function renderEngineParams() {
    const host = $("engine-params");
    host.innerHTML = "";
    for (const p of state.catalog.engine_params) host.appendChild(buildParamRow(p, `eng-${p.name}`));
  }
  function buildParamRow(p, inputId) {
    const row = document.createElement("div");
    row.className = "param-row";
    const head = document.createElement("div");
    head.className = "param-head";
    const lbl = document.createElement("label");
    lbl.textContent = p.name;
    lbl.setAttribute("for", inputId);
    const val = document.createElement("div");
    val.className = "param-value";
    val.textContent = formatParamValue(p, p.default);
    head.appendChild(lbl);
    head.appendChild(val);
    row.appendChild(head);

    const inp = document.createElement("input");
    inp.type = "range";
    inp.id = inputId;
    inp.min = p.min; inp.max = p.max; inp.step = p.step;
    inp.value = p.default;
    inp.addEventListener("input", () => {
      val.textContent = formatParamValue(p, +inp.value);
      renderLiveCode();
      scheduleRun();
    });
    row.appendChild(inp);

    if (p.help) {
      const help = document.createElement("p");
      help.className = "param-help";
      help.textContent = p.help;
      row.appendChild(help);
    }
    return row;
  }
  function formatParamValue(p, v) {
    if (p.name === "transaction_cost_bps") return `${v.toFixed(1)} bps`;
    if (p.name === "long_quantile") return `${(v * 100).toFixed(0)}%`;
    if (p.type === "float") return v.toFixed(2);
    return `${v}`;
  }

  function readParams() {
    const s = state.currentStrategy;
    const strategyParams = {};
    for (const p of s.params) {
      const el = document.getElementById(`strat-${p.name}`);
      strategyParams[p.name] = p.type === "int" ? parseInt(el.value, 10) : parseFloat(el.value);
    }
    const engineParams = {};
    for (const p of state.catalog.engine_params) {
      const el = document.getElementById(`eng-${p.name}`);
      engineParams[p.name] = p.type === "int" ? parseInt(el.value, 10) : parseFloat(el.value);
    }
    return { strategyParams, engineParams, start: $("start").value, end: $("end").value };
  }

  // ---------- Live code ----------
  function pyRepr(v) {
    if (typeof v === "string") return `"${v}"`;
    if (typeof v === "boolean") return v ? "True" : "False";
    if (typeof v === "number" && Number.isInteger(v)) return `${v}`;
    return `${(+v).toFixed(2)}`;
  }
  function renderLiveCode() {
    const s = state.currentStrategy;
    const { strategyParams, engineParams } = readParams();
    const stratArgs = Object.entries(strategyParams)
      .map(([k, v]) => `    ${k}=${pyRepr(v)},`)
      .join("\n");
    const stratCall = stratArgs
      ? `strategy = ${s.cls_name}(\n${stratArgs}\n)`
      : `strategy = ${s.cls_name}()`;

    const fixedLines = Object.entries(state.catalog.fixed_engine)
      .map(([k, v]) => `    ${k}=${pyRepr(v)},`)
      .join("\n");
    const liveCfgLines = Object.entries(engineParams)
      .map(([k, v]) => `    ${k}=${pyRepr(v)},`)
      .join("\n");
    const configCall =
`config = BacktestConfig(
${liveCfgLines}
${fixedLines}
)`;

    const full =
`# ── class source — ${s.source_file}
${s.source}

# ── your run (live values below)
${stratCall}

${configCall}

scores  = strategy.generate(dataset).scores
weights = build_target_weights(scores, dataset.returns, config)
result  = run_weight_backtest(dataset.prices, weights, config,
                              benchmark_returns=dataset.benchmark_returns)
`;
    const el = $("code-live");
    el.textContent = full;
    Prism.highlightElement(el);
  }

  // ---------- Run ----------
  let runTimer = null;
  function scheduleRun() {
    clearTimeout(runTimer);
    runTimer = setTimeout(run, 350);
  }
  $("start").addEventListener("change", () => { renderLiveCode(); scheduleRun(); });
  $("end").addEventListener("change", () => { renderLiveCode(); scheduleRun(); });

  async function run() {
    if (state.running) { state.pending = true; return; }
    state.running = true;
    setStatus("running simulation…", "loading");
    try {
      const p = readParams();
      const body = {
        strategy_id: state.currentStrategy.id,
        strategy_params: p.strategyParams,
        engine_params: p.engineParams,
        start: p.start, end: p.end,
      };
      const t0 = performance.now();
      const res = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
        throw new Error(err.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const elapsed = performance.now() - t0;
      renderResults(data);
      setStatus(`ready · last run ${(elapsed / 1000).toFixed(2)}s`, "ready");
    } catch (exc) {
      console.error(exc);
      toast(`Error: ${exc.message}`, true, 6000);
      setStatus(`error: ${exc.message}`, "error");
    } finally {
      state.running = false;
      if (state.pending) { state.pending = false; scheduleRun(); }
    }
  }
  function renderResults(data) {
    const m = data.metrics_net || {};
    const b = data.metrics_benchmark || {};
    const o = data.order_summary || {};
    const setCard = (valId, subId, value, cls, sub) => {
      const el = $(valId);
      el.textContent = value;
      el.className = "m-value" + (cls ? ` ${cls}` : "");
      $(subId).textContent = sub;
    };
    setCard("m-sharpe", "m-sharpe-sub", fmtNum(m.sharpe),
      m.sharpe >= 1 ? "good" : (m.sharpe < 0.3 ? "bad" : ""),
      `bench ${fmtNum(b.sharpe)}`);
    setCard("m-return", "m-return-sub", fmtPct(m.annualized_return),
      (m.annualized_return ?? 0) >= 0 ? "good" : "bad",
      `bench ${fmtPct(b.annualized_return)}`);
    setCard("m-dd", "m-dd-sub", fmtPct(m.max_drawdown), "bad",
      `bench ${fmtPct(b.max_drawdown)}`);
    setCard("m-tcost", "m-tcost-sub", fmtPct(o.tcost_drag_annualized), "bad",
      `turnover ${fmtX(o.turnover_annualized, 1)}/yr`);

    Plotly.react("chart-equity", [
      { x: data.dates, y: data.cumulative_benchmark.map((v) => v * 100),
        name: "Equal-weight universe", type: "scatter", mode: "lines",
        line: { color: "#7a8a9c", width: 1.5, dash: "dot" } },
      { x: data.dates, y: data.cumulative_net.map((v) => v * 100),
        name: "Strategy (net of t-cost)", type: "scatter", mode: "lines",
        line: { color: "#58c1ff", width: 2.3 } },
    ], {
      ...plotLayout,
      yaxis: { ...plotLayout.yaxis, title: "Cumulative return (%)", tickformat: ",.0f" },
    }, plotConfig);

    renderAudit(data.survivorship_audit || {});
  }

  function renderAudit(a) {
    const sub = $("audit-sub");
    const statsEl = $("audit-stats");
    const noteEl = $("audit-note");
    if (!a || !a.universe_size) {
      sub.textContent = "no audit available";
      statsEl.innerHTML = "";
      noteEl.textContent = "";
      Plotly.purge("chart-audit");
      return;
    }
    const bias = a.expected_annual_upward_bias_pct || {};
    const biasTxt = bias.low != null && bias.high != null
      ? `${bias.low.toFixed(1)}–${bias.high.toFixed(1)}% / yr`
      : "—";
    sub.textContent = `${a.universe_size} survivors · ${a.window_start} → ${a.window_end} (${a.window_years}y)`;

    const inceptionCls = a.inception_biased_count > 0 ? "warn" : "";
    const delistCls = a.delisted_within_window_count > 0 ? "bad" : "";

    statsEl.innerHTML = `
      <div class="a-cell"><div class="a-label">Universe size</div><div class="a-value">${a.universe_size}</div></div>
      <div class="a-cell"><div class="a-label">Est. upward bias</div><div class="a-value warn">${biasTxt}</div></div>
      <div class="a-cell"><div class="a-label">Active at start</div><div class="a-value">${a.active_at_start}</div></div>
      <div class="a-cell"><div class="a-label">Active at end</div><div class="a-value">${a.active_at_end}</div></div>
      <div class="a-cell"><div class="a-label">Post-start inceptions</div><div class="a-value ${inceptionCls}">${a.inception_biased_count}</div></div>
      <div class="a-cell"><div class="a-label">Delistings in window</div><div class="a-value ${delistCls}">${a.delisted_within_window_count}</div></div>
    `;
    noteEl.textContent = a.structural_bias_note || "";

    const ts = a.active_timeseries || { dates: [], counts: [] };
    Plotly.react("chart-audit", [
      { x: ts.dates, y: ts.counts, type: "scatter", mode: "lines",
        name: "Active tickers", line: { color: "#58c1ff", width: 1.8 },
        fill: "tozeroy", fillcolor: "rgba(88,193,255,0.08)" },
    ], {
      ...plotLayout,
      margin: { t: 10, r: 10, b: 30, l: 48 },
      height: 220,
      showlegend: false,
      yaxis: { ...plotLayout.yaxis, title: "Active tickers", rangemode: "tozero" },
    }, plotConfig);
  }

  // ---------- Data inspector ----------
  async function loadDataOverview() {
    const res = await fetch("/api/data");
    if (!res.ok) throw new Error(`data HTTP ${res.status}`);
    state.dataOverview = await res.json();
    $("data-sub").textContent = `${state.dataOverview.n_tickers} tickers · ${state.dataOverview.source_file}`;
    renderTickerTable();
  }
  function renderTickerTable() {
    const body = $("ticker-tbody");
    body.innerHTML = "";
    const q = state.tickerFilter.toLowerCase();
    const rows = state.dataOverview.tickers.filter((r) =>
      !q ||
      r.ticker.toLowerCase().includes(q) ||
      (r.name || "").toLowerCase().includes(q) ||
      (r.sector || "").toLowerCase().includes(q)
    );
    for (const r of rows) {
      const tr = document.createElement("tr");
      tr.dataset.ticker = r.ticker;
      if (r.ticker === state.selectedTicker) tr.classList.add("active");
      const retCls = r.total_return >= 0 ? "ret-pos" : "ret-neg";
      tr.innerHTML = `
        <td><span class="tic-sym">${r.ticker}</span></td>
        <td>${escapeHtml(r.name || "")}</td>
        <td>${escapeHtml(r.sector || "")}</td>
        <td style="text-align: right">${r.start}</td>
        <td style="text-align: right">${r.end}</td>
        <td style="text-align: right" class="${retCls}">${fmtPct(r.total_return, 0)}</td>
      `;
      tr.addEventListener("click", () => selectTicker(r.ticker));
      body.appendChild(tr);
    }
    if (!rows.length) {
      body.innerHTML = `<tr><td colspan="6" style="padding: 20px; text-align: center; color: var(--muted);">no matches</td></tr>`;
    }
  }
  function escapeHtml(s) { return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
  async function selectTicker(tic) {
    state.selectedTicker = tic;
    document.querySelectorAll("#ticker-table tbody tr").forEach((tr) => {
      tr.classList.toggle("active", tr.dataset.ticker === tic);
    });
    const right = $("data-right");
    right.innerHTML = `<div class="data-empty">Loading ${tic}…</div>`;
    try {
      const res = await fetch(`/api/data/ticker/${encodeURIComponent(tic)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const d = await res.json();
      renderTickerDetail(d);
    } catch (exc) {
      right.innerHTML = `<div class="data-empty">error: ${exc.message}</div>`;
    }
  }
  function renderTickerDetail(d) {
    const right = $("data-right");
    const metaBits = Object.entries(d.metadata).map(([k, v]) => `${k}=${v}`).join(" · ");
    right.innerHTML = `
      <div class="detail-head">
        <span class="detail-sym">${d.ticker}</span>
        <span class="detail-name">${escapeHtml(d.metadata.conm || "")}</span>
      </div>
      <div class="detail-meta">${escapeHtml(metaBits)}</div>
      <div class="detail-stats">
        <div class="s-cell"><div class="s-label">Period</div><div class="s-value">${d.start} → ${d.end} · ${d.n_days}d</div></div>
        <div class="s-cell"><div class="s-label">Ann. return</div><div class="s-value">${fmtPct(d.annualized_return)}</div></div>
        <div class="s-cell"><div class="s-label">Ann. vol</div><div class="s-value">${fmtPct(d.annualized_volatility)}</div></div>
        <div class="s-cell"><div class="s-label">Max DD</div><div class="s-value">${fmtPct(d.max_drawdown)}</div></div>
        <div class="s-cell"><div class="s-label">First price</div><div class="s-value">$${d.first_price.toFixed(2)}</div></div>
        <div class="s-cell"><div class="s-label">Last price</div><div class="s-value">$${d.last_price.toFixed(2)}</div></div>
      </div>
      <div id="detail-chart" class="detail-chart"></div>
    `;
    Plotly.react("detail-chart", [{
      x: d.dates, y: d.prices, type: "scatter", mode: "lines",
      line: { color: "#58c1ff", width: 1.6 },
      hovertemplate: "%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
    }], {
      ...plotLayout,
      margin: { t: 8, r: 8, b: 28, l: 48 },
      yaxis: { ...plotLayout.yaxis, title: "Price", type: "log", tickformat: ".2f" },
      showlegend: false,
    }, plotConfig);
  }
  $("ticker-search").addEventListener("input", (e) => {
    state.tickerFilter = e.target.value;
    renderTickerTable();
  });

  // ---------- Boot ----------
  async function waitReady() {
    while (true) {
      try {
        const res = await fetch("/api/status");
        const s = await res.json();
        if (s.error) { setStatus(`error: ${s.error}`, "error"); return false; }
        if (s.ready) {
          setStatus(`ready · ${s.tickers} tickers · ${s.date_min} → ${s.date_max}`, "ready");
          $("start").min = s.date_min; $("start").max = s.date_max;
          $("end").min = s.date_min; $("end").max = s.date_max;
          const label = s.universe_label || `${s.tickers} tickers`;
          $("universe-label").textContent = `universe: ${label}`;
          return true;
        }
        setStatus(`loading · ${s.message}`, "loading");
      } catch (exc) {
        setStatus("backend unreachable", "error"); return false;
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
  }

  (async () => {
    setStatus("connecting…", "loading");
    if (!(await waitReady())) return;
    state.catalog = await loadCatalog();
    renderStrategyTabs();
    selectStrategy(state.catalog.strategies[0].id);
    loadDataOverview().catch((e) => console.error(e));
  })();
})();
