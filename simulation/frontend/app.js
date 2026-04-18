(() => {
  const $ = (id) => document.getElementById(id);

  const state = {
    catalog: null,
    currentStrategy: null,
    codeView: "signal",
    lastResponse: null,
  };

  // ---------- Plotly ----------
  const plotLayout = {
    paper_bgcolor: "#11151c",
    plot_bgcolor: "#11151c",
    font: { color: "#e6ecf2", family: "ui-sans-serif, -apple-system, Segoe UI" },
    margin: { t: 20, r: 16, b: 36, l: 56 },
    xaxis: { gridcolor: "#1c2330", zerolinecolor: "#1c2330" },
    yaxis: { gridcolor: "#1c2330", zerolinecolor: "#1c2330" },
    legend: { orientation: "h", y: -0.22, x: 0 },
    hovermode: "x unified",
  };
  const plotConfig = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

  // ---------- Utilities ----------
  const fmtPct = (v, digits = 2) => (v == null ? "—" : `${(v * 100).toFixed(digits)}%`);
  const fmtNum = (v, digits = 2) => (v == null ? "—" : v.toFixed(digits));
  const fmtInt = (v) => (v == null ? "—" : Math.round(v).toLocaleString());

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

  // ---------- Catalog + strategy selection ----------
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
    renderCode();
  }

  // ---------- Code viewer ----------
  function renderCode() {
    const view = state.codeView;
    const s = state.currentStrategy;
    const cat = state.catalog;
    const codeEl = $("code-view");
    const pathEl = $("panel-path");
    let src = "";
    let path = "";
    if (view === "signal") {
      src = s.source;
      path = s.source_file;
    } else if (view === "orders") {
      src = cat.engine_sources.build_target_weights;
      path = cat.engine_source_file;
    } else if (view === "loop") {
      src = cat.engine_sources.run_weight_backtest;
      path = cat.engine_source_file;
    } else if (view === "config") {
      src = cat.engine_sources.BacktestConfig;
      path = cat.engine_source_file;
    }
    codeEl.textContent = src;
    pathEl.textContent = path;
    Prism.highlightElement(codeEl);
  }

  document.querySelectorAll("#code-tabs button").forEach((b) => {
    b.addEventListener("click", () => {
      state.codeView = b.dataset.view;
      document.querySelectorAll("#code-tabs button").forEach((x) => x.classList.toggle("active", x === b));
      renderCode();
    });
  });

  // ---------- Params ----------
  function renderStrategyParams() {
    const host = $("strategy-params");
    host.innerHTML = "";
    const s = state.currentStrategy;
    if (!s.params.length) {
      host.innerHTML = `<div class="param-desc" style="padding: 0;">No tunable parameters — the strategy only uses dataset.market_caps.</div>`;
      return;
    }
    for (const p of s.params) {
      host.appendChild(buildParamRow(p, `strat-${p.name}`));
    }
  }

  function renderEngineParams() {
    const host = $("engine-params");
    host.innerHTML = "";
    for (const p of state.catalog.engine_params) {
      host.appendChild(buildParamRow(p, `eng-${p.name}`));
    }
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
      row.appendChild(sel);
    } else {
      const isFloat = p.type === "float";
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
      });
      row.appendChild(inp);
      row.appendChild(val);
    }

    const desc = document.createElement("div");
    desc.className = "param-desc";
    desc.textContent = `# ${p.desc}`;
    row.appendChild(desc);
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
    return {
      strategy_id: s.id,
      strategy_params: strategyParams,
      engine_params: engineParams,
      start: $("start").value,
      end: $("end").value,
    };
  }

  // ---------- Run + render ----------
  async function run() {
    const btn = $("run");
    btn.disabled = true;
    btn.textContent = "Running";
    setStatus("running simulation…", "loading");
    try {
      const t0 = performance.now();
      const res = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(readParams()),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
        throw new Error(err.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const elapsed = performance.now() - t0;
      state.lastResponse = data;
      renderResults(data);
      setStatus(`ready · last run ${(elapsed / 1000).toFixed(2)}s`, "ready");
    } catch (exc) {
      console.error(exc);
      toast(`Error: ${exc.message}`, true, 6000);
      setStatus(`error: ${exc.message}`, "error");
    } finally {
      btn.disabled = false;
      btn.textContent = "Run";
    }
  }

  function renderResults(data) {
    const m = data.metrics_net || {};
    const b = data.metrics_benchmark || {};
    const setCard = (id, val, cls, sub) => {
      const el = $(id);
      el.textContent = val;
      el.className = "m-value" + (cls ? ` ${cls}` : "");
      if (sub) {
        const parent = el.parentElement;
        let s = parent.querySelector(".m-sub");
        if (!s) { s = document.createElement("div"); s.className = "m-sub"; parent.appendChild(s); }
        s.textContent = sub;
      }
    };
    setCard("m-sharpe", fmtNum(m.sharpe), m.sharpe >= 1 ? "good" : (m.sharpe < 0.5 ? "bad" : ""), `bench ${fmtNum(b.sharpe)}`);
    setCard("m-return", fmtPct(m.annualized_return), m.annualized_return >= 0 ? "good" : "bad", `bench ${fmtPct(b.annualized_return)}`);
    setCard("m-vol", fmtPct(m.annualized_volatility), "", `bench ${fmtPct(b.annualized_volatility)}`);
    setCard("m-dd", fmtPct(m.max_drawdown), "bad", `bench ${fmtPct(b.max_drawdown)}`);
    setCard("m-sortino", fmtNum(m.sortino), "", `bench ${fmtNum(b.sortino)}`);
    setCard("m-calmar", fmtNum(m.calmar), "", `bench ${fmtNum(b.calmar)}`);

    // Equity curve
    const dates = data.dates;
    Plotly.react("chart-equity", [
      {
        x: dates, y: data.cumulative_benchmark.map((v) => v * 100),
        name: "Equal-weight universe", type: "scatter", mode: "lines",
        line: { color: "#7a8a9c", width: 1.5, dash: "dot" },
      },
      {
        x: dates, y: data.cumulative_gross.map((v) => v * 100),
        name: "Strategy (gross)", type: "scatter", mode: "lines",
        line: { color: "#7aff9e", width: 1.3 }, opacity: 0.7,
      },
      {
        x: dates, y: data.cumulative_net.map((v) => v * 100),
        name: "Strategy (net of t-cost)", type: "scatter", mode: "lines",
        line: { color: "#58c1ff", width: 2.3 },
      },
    ], { ...plotLayout, yaxis: { ...plotLayout.yaxis, title: "Cumulative return (%)", tickformat: ",.0f" } }, plotConfig);

    // Drawdown
    Plotly.react("chart-drawdown", [{
      x: dates, y: data.drawdown.map((v) => v * 100),
      type: "scatter", mode: "lines", fill: "tozeroy",
      line: { color: "#f87171", width: 1.2 },
      fillcolor: "rgba(248, 113, 113, 0.18)", name: "Drawdown",
    }], {
      ...plotLayout, margin: { t: 16, r: 16, b: 32, l: 56 },
      yaxis: { ...plotLayout.yaxis, title: "%", tickformat: ",.0f", rangemode: "nonpositive" },
      showlegend: false,
    }, plotConfig);

    // Orders
    const o = data.order_summary;
    $("order-summary").innerHTML = `
      <div class="o-item"><div class="o-label">Rebalances</div><div class="o-value">${fmtInt(o.n_rebalances)}</div></div>
      <div class="o-item"><div class="o-label">Avg positions</div><div class="o-value">${fmtNum(o.avg_positions, 1)}</div></div>
      <div class="o-item"><div class="o-label">Turnover / year</div><div class="o-value">${fmtNum(o.turnover_annualized, 2)}×</div></div>
      <div class="o-item"><div class="o-label">T-cost drag / year</div><div class="o-value">${fmtPct(o.tcost_drag_annualized, 2)}</div></div>
    `;

    // Holdings
    const host = $("holdings");
    host.innerHTML = "";
    for (const snap of data.recent_holdings) {
      const row = document.createElement("div");
      row.className = "holdings-row";
      const chips = snap.top.map((t) =>
        `<span class="ticker-chip">${t.ticker}<span class="w">${(t.weight * 100).toFixed(1)}%</span></span>`
      ).join("");
      row.innerHTML = `
        <div class="holdings-date">
          <span>${snap.date}</span>
          <span><span class="n-pos">${snap.n_positions}</span> positions · gross ${fmtPct(snap.gross_exposure, 0)}</span>
        </div>
        <div class="holdings-tickers">${chips}</div>
      `;
      host.appendChild(row);
    }
  }

  // ---------- Boot ----------
  $("run").addEventListener("click", run);

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
    run();
  })();
})();
