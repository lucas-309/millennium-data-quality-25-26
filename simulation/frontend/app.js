(() => {
  const $ = (id) => document.getElementById(id);

  const plotLayout = {
    paper_bgcolor: "#121820",
    plot_bgcolor: "#121820",
    font: { color: "#e6ecf2", family: "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto" },
    margin: { t: 30, r: 20, b: 40, l: 55 },
    xaxis: { gridcolor: "#1a2533", zerolinecolor: "#1a2533" },
    yaxis: { gridcolor: "#1a2533", zerolinecolor: "#1a2533" },
    legend: { orientation: "h", y: -0.18, x: 0 },
    hovermode: "x unified",
  };
  const plotConfig = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

  // Wire up slider value readouts
  const sliderMap = {
    target_vol: { el: "tv_v", fmt: (v) => `${(+v * 100).toFixed(0)}%` },
    max_leverage: { el: "ml_v", fmt: (v) => `${(+v).toFixed(2)}×` },
    sma_window: { el: "sma_v", fmt: (v) => `${v}d` },
    risk_free: { el: "rf_v", fmt: (v) => `${(+v * 100).toFixed(1)}%` },
    rolling_window: { el: "rw_v", fmt: (v) => `${v}d` },
  };
  Object.entries(sliderMap).forEach(([id, cfg]) => {
    const inp = $(id);
    const lbl = $(cfg.el);
    const render = () => { lbl.textContent = cfg.fmt(inp.value); };
    inp.addEventListener("input", render);
    render();
  });

  function readParams() {
    const included = [];
    document.querySelectorAll("input[data-name]").forEach((el) => {
      if (el.checked) included.push(el.dataset.name);
    });
    return {
      start: $("start").value,
      end: $("end").value,
      target_vol: +$("target_vol").value,
      max_leverage: +$("max_leverage").value,
      sma_window: +$("sma_window").value,
      risk_free: +$("risk_free").value,
      include_vol_managed: $("inc_vol_managed").checked,
      include_trend_filtered: $("inc_trend_filtered").checked,
      included_selections: included,
      selection_trend_overlay: $("selection_trend_overlay").checked,
      combo_method: $("combo_method").value,
      combined_trend_overlay: $("combined_trend_overlay").checked,
      rolling_window: +$("rolling_window").value,
    };
  }

  function toast(msg, isError = false, ms = 3600) {
    const t = $("toast");
    t.textContent = msg;
    t.classList.toggle("error", isError);
    t.classList.remove("hidden");
    clearTimeout(toast._to);
    toast._to = setTimeout(() => t.classList.add("hidden"), ms);
  }

  function setStatus(text, kind) {
    const el = $("status-pill");
    el.textContent = text;
    el.className = "pill " + (kind ? `pill-${kind}` : "pill-idle");
  }

  async function pollStatus() {
    try {
      const res = await fetch("/api/status");
      const s = await res.json();
      if (s.error) {
        setStatus("error", "error");
        toast(`backend error: ${s.error}`, true, 8000);
        return false;
      }
      if (s.ready) {
        setStatus(`ready · ${s.tickers} tickers · ${s.date_min} → ${s.date_max}`, "ready");
        $("run").disabled = false;
        // Clamp date inputs to available range
        $("start").min = s.date_min; $("start").max = s.date_max;
        $("end").min = s.date_min; $("end").max = s.date_max;
        return true;
      }
      setStatus(`loading · ${s.message}`, "loading");
      $("run").disabled = true;
      return false;
    } catch (exc) {
      setStatus("backend unreachable", "error");
      return false;
    }
  }

  async function waitForReady() {
    while (!(await pollStatus())) {
      await new Promise((r) => setTimeout(r, 1200));
    }
  }

  // ----- Charts -----
  function plotCumulative(resp) {
    const dates = resp.dates;
    const traces = [];
    traces.push({
      x: dates, y: resp.benchmark.cumulative.map((v) => v * 100),
      name: "Equal-Weight Universe", type: "scatter", mode: "lines",
      line: { color: "#8a9cb0", width: 2, dash: "dot" }, opacity: 0.9,
    });
    const palette = ["#58c1ff", "#9b6bff", "#4ade80", "#fbbf24", "#f472b6", "#34d399", "#f87171"];
    let i = 0;
    for (const [name, d] of Object.entries(resp.sleeves)) {
      traces.push({
        x: dates, y: d.cumulative.map((v) => v * 100),
        name, type: "scatter", mode: "lines",
        line: { color: palette[i % palette.length], width: 1.3 }, opacity: 0.7,
      });
      i++;
    }
    traces.push({
      x: dates, y: resp.combined.cumulative.map((v) => v * 100),
      name: resp.combined.name, type: "scatter", mode: "lines",
      line: { color: "#ffffff", width: 3 },
    });
    Plotly.react("chart-cumulative", traces, {
      ...plotLayout,
      yaxis: { ...plotLayout.yaxis, title: "Cumulative return (%)", tickformat: ",.0f" },
    }, plotConfig);
  }

  function plotDrawdown(resp) {
    const dd = resp.combined.drawdown.map((v) => v * 100);
    Plotly.react("chart-drawdown", [{
      x: resp.dates, y: dd, type: "scatter", mode: "lines", fill: "tozeroy",
      line: { color: "#f87171", width: 1.3 }, fillcolor: "rgba(248, 113, 113, 0.18)",
      name: "Drawdown",
    }], {
      ...plotLayout,
      yaxis: { ...plotLayout.yaxis, title: "Drawdown (%)", tickformat: ",.0f", rangemode: "nonpositive" },
      showlegend: false,
    }, plotConfig);
  }

  function plotRollingSharpe(resp) {
    const rs = resp.combined.rolling_sharpe;
    Plotly.react("chart-rolling", [{
      x: resp.dates, y: rs, type: "scatter", mode: "lines",
      line: { color: "#58c1ff", width: 1.5 }, name: "Rolling Sharpe",
    }, {
      x: [resp.dates[0], resp.dates[resp.dates.length - 1]], y: [1.0, 1.0],
      type: "scatter", mode: "lines",
      line: { color: "#8a9cb0", width: 1, dash: "dash" }, name: "Sharpe = 1",
      hoverinfo: "skip",
    }], {
      ...plotLayout,
      yaxis: { ...plotLayout.yaxis, title: "Rolling Sharpe" },
      showlegend: false,
    }, plotConfig);
  }

  function plotCorrelation(resp) {
    const { labels, matrix } = resp.correlation;
    Plotly.react("chart-corr", [{
      z: matrix, x: labels, y: labels, type: "heatmap",
      colorscale: [
        [0, "#1e3a5f"], [0.5, "#121820"], [1, "#9b6bff"],
      ],
      zmin: -1, zmax: 1,
      text: matrix.map((row) => row.map((v) => v.toFixed(2))),
      texttemplate: "%{text}",
      textfont: { size: 11, color: "#e6ecf2" },
      showscale: true,
      colorbar: { thickness: 10, len: 0.7 },
    }], {
      ...plotLayout, margin: { t: 20, r: 20, b: 100, l: 140 },
      xaxis: { ...plotLayout.xaxis, tickangle: -30 },
      yaxis: { ...plotLayout.yaxis, autorange: "reversed" },
    }, plotConfig);
  }

  function plotHeatmap(resp) {
    const { years, months, values } = resp.monthly_heatmap;
    const monthNames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    const text = values.map((row) => row.map((v) => (v == null ? "" : (v * 100).toFixed(1))));
    Plotly.react("chart-heatmap", [{
      z: values.map((row) => row.map((v) => (v == null ? null : v * 100))),
      x: monthNames, y: years, type: "heatmap",
      colorscale: [
        [0, "#7f1d1d"], [0.25, "#b45353"], [0.5, "#121820"],
        [0.75, "#6abf6a"], [1, "#4ade80"],
      ],
      zmid: 0,
      text, texttemplate: "%{text}",
      textfont: { size: 10, color: "#e6ecf2" },
      showscale: true, colorbar: { thickness: 10, ticksuffix: "%" },
    }], {
      ...plotLayout, margin: { t: 20, r: 20, b: 40, l: 60 },
      yaxis: { ...plotLayout.yaxis, autorange: "reversed", type: "category" },
    }, plotConfig);
  }

  // ----- Scorecard + table -----
  const fmtPct = (v) => (v == null ? "—" : `${(v * 100).toFixed(2)}%`);
  const fmtNum = (v) => (v == null ? "—" : v.toFixed(2));

  function renderScorecard(resp) {
    const m = resp.metrics.Combined || {};
    const b = resp.metrics.Benchmark || {};
    const lift = (m.sharpe ?? 0) - (b.sharpe ?? 0);
    const cards = [
      { label: "Sharpe (Combined)", value: fmtNum(m.sharpe), cls: m.sharpe >= 1 ? "good" : (m.sharpe >= 0.5 ? "" : "bad"), sub: `vs benchmark ${fmtNum(b.sharpe)} (${lift >= 0 ? "+" : ""}${lift.toFixed(2)})` },
      { label: "Ann. return", value: fmtPct(m.annualized_return), cls: (m.annualized_return ?? 0) >= 0 ? "good" : "bad", sub: `bench ${fmtPct(b.annualized_return)}` },
      { label: "Ann. vol", value: fmtPct(m.annualized_volatility), cls: "", sub: `bench ${fmtPct(b.annualized_volatility)}` },
      { label: "Max drawdown", value: fmtPct(m.max_drawdown), cls: "bad", sub: `bench ${fmtPct(b.max_drawdown)}` },
      { label: "Sortino", value: fmtNum(m.sortino), cls: "", sub: `bench ${fmtNum(b.sortino)}` },
      { label: "Calmar", value: fmtNum(m.calmar), cls: "", sub: `bench ${fmtNum(b.calmar)}` },
      { label: "Win rate (daily)", value: fmtPct(m.win_rate), cls: "", sub: `bench ${fmtPct(b.win_rate)}` },
      { label: "Sleeves", value: String(resp.params_echo.n_sleeves), cls: "", sub: `method: ${resp.params_echo.combo_method.toUpperCase()}` },
    ];
    $("scorecard").innerHTML = cards.map((c) =>
      `<div class="card">
        <div class="card-label">${c.label}</div>
        <div class="card-value ${c.cls}">${c.value}</div>
        <div class="card-sub">${c.sub}</div>
      </div>`).join("");
  }

  function renderTable(resp) {
    const cols = ["annualized_return", "annualized_volatility", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate"];
    const labels = ["Ann. return", "Ann. vol", "Sharpe", "Sortino", "Max DD", "Calmar", "Win rate"];
    const fmts = [fmtPct, fmtPct, fmtNum, fmtNum, fmtPct, fmtNum, fmtPct];
    const order = ["Benchmark", ...Object.keys(resp.sleeves), "Combined"];
    let html = `<table class="metrics"><thead><tr><th>Strategy</th>${labels.map((l) => `<th>${l}</th>`).join("")}</tr></thead><tbody>`;
    for (const name of order) {
      const m = resp.metrics[name] || {};
      const rowCls = name === "Combined" ? "combined" : (name === "Benchmark" ? "benchmark" : "");
      html += `<tr class="${rowCls}"><td>${name}</td>` +
        cols.map((c, i) => `<td>${fmts[i](m[c])}</td>`).join("") + "</tr>";
    }
    html += "</tbody></table>";
    $("metrics-table").innerHTML = html;
  }

  // ----- Run simulation -----
  async function runSim() {
    const btn = $("run");
    btn.disabled = true;
    btn.textContent = "Running…";
    try {
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
      renderScorecard(data);
      renderTable(data);
      plotCumulative(data);
      plotDrawdown(data);
      plotRollingSharpe(data);
      plotCorrelation(data);
      plotHeatmap(data);
      toast(`Simulation complete — Sharpe ${(data.metrics.Combined?.sharpe ?? 0).toFixed(2)}`);
    } catch (exc) {
      console.error(exc);
      toast(`Error: ${exc.message}`, true, 6000);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run simulation";
    }
  }

  $("run").addEventListener("click", runSim);
  $("reset").addEventListener("click", () => {
    location.reload();
  });

  // Boot: wait until backend is warmed up, then auto-run the first simulation.
  (async () => {
    setStatus("connecting", "loading");
    await waitForReady();
    runSim();
  })();
})();
