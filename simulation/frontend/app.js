(() => {
  const $ = (id) => document.getElementById(id);

  const state = {
    catalog: null,
    currentStrategy: null,
    codeView: "signal",
    running: false,
    pending: false,
  };

  // ---------- Plotly ----------
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

  // ---------- Formatters ----------
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

  // ---------- Catalog ----------
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
    $("strategy-title").textContent = s.label;
    $("strategy-summary").textContent = s.summary;
    renderStrategyParams();
    renderEngineParams();
    renderCodeReference();
    renderLiveCode();
    scheduleRun();
  }

  // ---------- Reference code viewer ----------
  function renderCodeReference() {
    const view = state.codeView;
    const s = state.currentStrategy;
    const cat = state.catalog;
    let src = "", path = "";
    if (view === "signal") { src = s.source; path = s.source_file; }
    else if (view === "orders") { src = cat.engine_sources.build_target_weights; path = cat.engine_source_file; }
    else if (view === "loop") { src = cat.engine_sources.run_weight_backtest; path = cat.engine_source_file; }
    const codeEl = $("code-view");
    codeEl.textContent = src;
    $("panel-path").textContent = path;
    Prism.highlightElement(codeEl);
  }
  document.querySelectorAll("#code-tabs button").forEach((b) => {
    b.addEventListener("click", () => {
      state.codeView = b.dataset.view;
      document.querySelectorAll("#code-tabs button").forEach((x) => x.classList.toggle("active", x === b));
      renderCodeReference();
    });
  });

  // ---------- Live "effective call" block ----------
  function pyRepr(v) {
    if (typeof v === "string") return `"${v}"`;
    if (typeof v === "boolean") return v ? "True" : "False";
    if (typeof v === "number" && Number.isInteger(v)) return `${v}`;
    return `${(+v).toFixed(2)}`;
  }

  function renderLiveCode() {
    const s = state.currentStrategy;
    const { strategyParams, engineParams } = readParams();

    const stratClass = s.cls_name || ({
      momentum: "CrossSectionalMomentumStrategy",
      lowvol: "LowVolatilityStrategy",
      smallcap: "SmallCapTiltStrategy",
    })[s.id];

    const stratArgs = Object.entries(strategyParams)
      .map(([k, v]) => `    ${k}=${pyRepr(v)},`)
      .join("\n");
    const stratCall = stratArgs
      ? `strategy = ${stratClass}(\n${stratArgs}\n)`
      : `strategy = ${stratClass}()`;

    const configArgs = Object.entries(engineParams)
      .map(([k, v]) => `    ${k}=${pyRepr(v)},`)
      .join("\n");
    const configCall = `config = BacktestConfig(\n${configArgs}\n    long_only=True,\n    short_quantile=0.0,\n    min_names=10,\n)`;

    const pipeline =
`scores  = strategy.generate(dataset).scores
weights = build_target_weights(scores, dataset.returns, config)
result  = run_weight_backtest(dataset.prices, weights, config,
                              benchmark_returns=dataset.benchmark_returns)`;

    const full = `${stratCall}\n\n${configCall}\n\n${pipeline}\n`;
    const el = $("code-live");
    el.textContent = full;
    Prism.highlightElement(el);
  }

  // ---------- Params ----------
  function renderStrategyParams() {
    const host = $("strategy-params");
    host.innerHTML = "";
    const s = state.currentStrategy;
    if (!s.params.length) {
      const empty = document.createElement("div");
      empty.style.cssText = "font-size: 11px; color: var(--muted); font-family: ui-monospace, Menlo, monospace;";
      empty.textContent = "# no tunable parameters";
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

    const lbl = document.createElement("label");
    lbl.textContent = p.name;
    lbl.setAttribute("for", inputId);
    row.appendChild(lbl);

    if (p.type === "choice") {
      const sel = document.createElement("select");
      sel.id = inputId;
      p.choices.forEach((c) => {
        const opt = document.createElement("option");
        opt.value = c.value; opt.textContent = c.label;
        if (c.value === p.default) opt.selected = true;
        sel.appendChild(opt);
      });
      sel.addEventListener("change", onParamChange);
      row.appendChild(sel);
    } else {
      const inp = document.createElement("input");
      inp.type = "range";
      inp.id = inputId;
      inp.min = p.min; inp.max = p.max; inp.step = p.step;
      inp.value = p.default;
      const val = document.createElement("div");
      val.className = "param-value";
      val.textContent = formatParamValue(p, p.default);
      inp.addEventListener("input", () => {
        val.textContent = formatParamValue(p, +inp.value);
        onParamChange();
      });
      row.appendChild(inp);
      row.appendChild(val);
    }
    return row;
  }
  function formatParamValue(p, v) {
    if (p.name === "transaction_cost_bps") return `${v.toFixed(1)} bps`;
    if (p.name === "long_quantile" || p.name === "max_position_weight") return `${(v * 100).toFixed(0)}%`;
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
      if (p.type === "choice") engineParams[p.name] = el.value;
      else if (p.type === "int") engineParams[p.name] = parseInt(el.value, 10);
      else engineParams[p.name] = parseFloat(el.value);
    }
    return { strategyParams, engineParams, start: $("start").value, end: $("end").value };
  }

  // ---------- Reactive scheduling ----------
  let runTimer = null;
  function scheduleRun() {
    renderLiveCode();
    clearTimeout(runTimer);
    runTimer = setTimeout(run, 350);
  }
  function onParamChange() {
    scheduleRun();
  }
  $("start").addEventListener("change", onParamChange);
  $("end").addEventListener("change", onParamChange);

  // ---------- Run + render ----------
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
      if (subId) $(subId).textContent = sub;
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

    // Equity: net strategy vs benchmark only. Clean, one decision surface.
    Plotly.react("chart-equity", [
      {
        x: data.dates,
        y: data.cumulative_benchmark.map((v) => v * 100),
        name: "Equal-weight universe",
        type: "scatter", mode: "lines",
        line: { color: "#7a8a9c", width: 1.5, dash: "dot" },
      },
      {
        x: data.dates,
        y: data.cumulative_net.map((v) => v * 100),
        name: "Strategy (net of t-cost)",
        type: "scatter", mode: "lines",
        line: { color: "#58c1ff", width: 2.3 },
      },
    ], {
      ...plotLayout,
      yaxis: { ...plotLayout.yaxis, title: "Cumulative return (%)", tickformat: ",.0f" },
    }, plotConfig);

    // Stats line
    $("stats-line").innerHTML = `
      <div><div class="stat-k">Ann. vol</div><div class="stat-v">${fmtPct(m.annualized_volatility)}</div></div>
      <div><div class="stat-k">Avg positions</div><div class="stat-v">${fmtNum(o.avg_positions, 1)}</div></div>
      <div><div class="stat-k">Turnover / yr</div><div class="stat-v">${fmtX(o.turnover_annualized, 2)}</div></div>
      <div><div class="stat-k">Win rate (daily)</div><div class="stat-v">${fmtPct(m.win_rate, 1)}</div></div>
    `;
  }

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
  })();
})();
