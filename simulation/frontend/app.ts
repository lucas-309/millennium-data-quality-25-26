// Source of truth for the strategy-simulator frontend.
// Compile to app.js with `npm run build` (or `npm run watch`).

// ---------- Ambient globals (loaded via <script> tags in index.html) ----------
declare const Plotly: {
  react(
    el: string | HTMLElement,
    data: unknown[],
    layout: Record<string, unknown>,
    config?: Record<string, unknown>,
  ): Promise<unknown>;
  purge(el: string | HTMLElement): void;
};
declare const Prism: {
  highlightElement(el: HTMLElement): void;
};

// ---------- API shapes ----------
interface ParamSpec {
  name: string;
  type: "int" | "float";
  min: number;
  max: number;
  step: number;
  default: number;
  help?: string;
}

interface Strategy {
  id: string;
  label: string;
  cls_name: string;
  source_file: string;
  source: string;
  summary: string;
  params: ParamSpec[];
  engine_overrides?: ParamSpec[];
}

interface Catalog {
  strategies: Strategy[];
  engine_params: ParamSpec[];
  fixed_engine: Record<string, string | number | boolean>;
}

interface Metrics {
  sharpe?: number | null;
  annualized_return?: number | null;
  max_drawdown?: number | null;
}

interface OrderSummary {
  tcost_drag_annualized?: number | null;
  turnover_annualized?: number | null;
}

interface AuditTimeseries {
  dates: string[];
  counts: number[];
}

interface BiasRange {
  low?: number | null;
  high?: number | null;
}

interface SurvivorshipAudit {
  universe_size?: number;
  window_start?: string;
  window_end?: string;
  window_years?: number;
  active_at_start?: number;
  active_at_end?: number;
  inception_biased_count?: number;
  delisted_within_window_count?: number;
  expected_annual_upward_bias_pct?: BiasRange;
  structural_bias_note?: string;
  active_timeseries?: AuditTimeseries;
}

interface UniverseSummary {
  n_tickers: number;
  custom: boolean;
  selected?: string[] | null;
  missing_from_cache?: string[];
}

interface RunConfig {
  long_only: boolean;
  long_quantile: number;
  short_quantile: number;
  leverage: number;
}

interface SimulationResult {
  metrics_net?: Metrics;
  metrics_benchmark?: Metrics;
  order_summary?: OrderSummary;
  dates: string[];
  cumulative_benchmark: number[];
  cumulative_net: number[];
  survivorship_audit?: SurvivorshipAudit;
  universe?: UniverseSummary;
  config?: RunConfig;
}

interface TickerRow {
  ticker: string;
  name?: string;
  sector?: string;
  start: string;
  end: string;
  total_return: number;
}

interface DataOverview {
  n_tickers: number;
  source_file: string;
  tickers: TickerRow[];
}

interface TickerDetail {
  ticker: string;
  metadata: Record<string, string | number>;
  start: string;
  end: string;
  n_days: number;
  annualized_return: number;
  annualized_volatility: number;
  max_drawdown: number;
  first_price: number;
  last_price: number;
  dates: string[];
  prices: number[];
}

interface StatusResponse {
  ready: boolean;
  error?: string;
  tickers?: number;
  date_min?: string;
  date_max?: string;
  message?: string;
  universe_label?: string;
}

interface AppState {
  catalog: Catalog | null;
  currentStrategy: Strategy | null;
  running: boolean;
  pending: boolean;
  dataOverview: DataOverview | null;
  selectedTicker: string | null;
  tickerFilter: string;
  universeSize: number;
  fullUniverseSize: number;
  customTickers: string[];
  // True when the user has explicitly narrowed the universe. Lets us
  // distinguish "empty filter = user deliberately picked 0 tickers" from
  // "empty filter = default full cache", which used to get conflated and
  // run the full universe on what should have been an empty result.
  hasCustomFilter: boolean;
}

interface RunPayload {
  strategyParams: Record<string, number>;
  engineParams: Record<string, number>;
  engineOverrides: Record<string, number>;
  tickers: string[];
  start: string;
  end: string;
}

type StatusKind = "ready" | "loading" | "error" | "" | undefined;

(() => {
  const $ = <T extends HTMLElement = HTMLElement>(id: string): T =>
    document.getElementById(id) as T;

  const state: AppState = {
    catalog: null,
    currentStrategy: null,
    running: false,
    pending: false,
    dataOverview: null,
    selectedTicker: null,
    tickerFilter: "",
    universeSize: 0,
    fullUniverseSize: 0,
    customTickers: [],
    hasCustomFilter: false,
  };

  const plotLayout: Record<string, unknown> = {
    paper_bgcolor: "#11151c",
    plot_bgcolor: "#11151c",
    font: { color: "#e6ecf2", family: "ui-sans-serif, -apple-system, Segoe UI" },
    margin: { t: 16, r: 16, b: 36, l: 56 },
    xaxis: { gridcolor: "#1c2330", zerolinecolor: "#1c2330" },
    yaxis: { gridcolor: "#1c2330", zerolinecolor: "#1c2330" },
    legend: { orientation: "h", y: -0.18, x: 0 },
    hovermode: "x unified",
  };
  const plotConfig: Record<string, unknown> = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };

  const fmtPct = (v: number | null | undefined, d = 2): string =>
    v == null ? "—" : `${(v * 100).toFixed(d)}%`;
  const fmtNum = (v: number | null | undefined, d = 2): string =>
    v == null ? "—" : v.toFixed(d);
  const fmtX = (v: number | null | undefined, d = 2): string =>
    v == null ? "—" : `${v.toFixed(d)}×`;

  let toastTimer: number | undefined;
  function toast(msg: string, isError = false, ms = 3500): void {
    const t = $("toast");
    t.textContent = msg;
    t.classList.toggle("error", isError);
    t.classList.remove("hidden");
    if (toastTimer !== undefined) clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => t.classList.add("hidden"), ms);
  }
  function setStatus(text: string, kind?: StatusKind): void {
    const el = $("status-bar");
    el.textContent = text;
    el.className = "statusbar" + (kind ? ` ${kind}` : "");
  }

  // ---------- Catalog & tabs ----------
  async function loadCatalog(): Promise<Catalog> {
    const res = await fetch("/api/catalog");
    if (!res.ok) throw new Error(`catalog HTTP ${res.status}`);
    return (await res.json()) as Catalog;
  }
  function renderStrategyTabs(): void {
    const nav = $("strategy-tabs");
    nav.innerHTML = "";
    state.catalog!.strategies.forEach((s, i) => {
      const btn = document.createElement("button");
      btn.textContent = s.label;
      btn.dataset.id = s.id;
      if (i === 0) {
        btn.classList.add("active");
        state.currentStrategy = s;
      }
      btn.addEventListener("click", () => selectStrategy(s.id));
      nav.appendChild(btn);
    });
  }
  function selectStrategy(id: string): void {
    const s = state.catalog!.strategies.find((x) => x.id === id);
    if (!s) return;
    state.currentStrategy = s;
    document.querySelectorAll<HTMLButtonElement>("#strategy-tabs button").forEach((b) => {
      b.classList.toggle("active", b.dataset.id === id);
    });
    $("strategy-summary").textContent = s.summary;
    renderStrategyParams();
    renderOverrideParams();
    renderEngineParams();
    // renderLiveCode calls updateSizing at the end, so no explicit call needed here.
    renderLiveCode();
    scheduleRun();
  }

  // ---------- Params ----------
  function renderStrategyParams(): void {
    const host = $("strategy-params");
    host.innerHTML = "";
    const s = state.currentStrategy!;
    if (!s.params.length) {
      const empty = document.createElement("div");
      empty.style.cssText =
        "font-size: 11px; color: var(--muted); font-family: ui-monospace, Menlo, monospace;";
      empty.textContent = "# no tunable strategy parameters";
      host.appendChild(empty);
      return;
    }
    for (const p of s.params) host.appendChild(buildParamRow(p, `strat-${p.name}`));
  }
  function renderEngineParams(): void {
    const host = $("engine-params");
    host.innerHTML = "";
    for (const p of state.catalog!.engine_params) host.appendChild(buildParamRow(p, `eng-${p.name}`));
  }
  function renderOverrideParams(): void {
    const host = $("override-params");
    const group = $("override-params-group");
    host.innerHTML = "";
    const overrides = (state.currentStrategy!.engine_overrides || []);
    if (!overrides.length) {
      (group as HTMLElement).hidden = true;
      return;
    }
    (group as HTMLElement).hidden = false;
    for (const p of overrides) host.appendChild(buildParamRow(p, `ovr-${p.name}`));
  }
  function buildParamRow(p: ParamSpec, inputId: string): HTMLDivElement {
    const row = document.createElement("div");
    row.className = "param-row";
    const head = document.createElement("div");
    head.className = "param-head";
    const lbl = document.createElement("label");
    lbl.textContent = p.name;
    lbl.setAttribute("for", inputId);

    const valWrap = document.createElement("div");
    valWrap.className = "param-value";
    const numInp = document.createElement("input");
    numInp.type = "number";
    numInp.min = String(p.min);
    numInp.max = String(p.max);
    numInp.step = String(p.step);
    numInp.value = String(p.default);
    const unit = paramUnit(p);
    if (unit) {
      const unitEl = document.createElement("span");
      unitEl.className = "param-unit";
      unitEl.textContent = unit;
      valWrap.appendChild(numInp);
      valWrap.appendChild(unitEl);
    } else {
      valWrap.appendChild(numInp);
    }
    head.appendChild(lbl);
    head.appendChild(valWrap);
    row.appendChild(head);

    const slider = document.createElement("input");
    slider.type = "range";
    slider.id = inputId;
    slider.min = String(p.min);
    slider.max = String(p.max);
    slider.step = String(p.step);
    slider.value = String(p.default);
    row.appendChild(slider);

    const syncFromSlider = (): void => {
      numInp.value = slider.value;
      renderLiveCode();
      scheduleRun();
    };
    const syncFromNumber = (): void => {
      // Don't clamp while user is typing — just mirror the value into the
      // slider (slider clips visually) and trigger a debounced run.
      const v = parseFloat(numInp.value);
      if (!Number.isFinite(v)) return;
      slider.value = String(v);
      renderLiveCode();
      scheduleRun();
    };
    slider.addEventListener("input", syncFromSlider);
    numInp.addEventListener("input", syncFromNumber);
    numInp.addEventListener("change", () => {
      // On blur / Enter: clamp to range and snap back if the user was out.
      let v = parseFloat(numInp.value);
      if (!Number.isFinite(v)) {
        numInp.value = slider.value;
        return;
      }
      if (v < +p.min) v = +p.min;
      if (v > +p.max) v = +p.max;
      numInp.value = String(v);
      slider.value = String(v);
      renderLiveCode();
      scheduleRun();
    });

    if (p.help) {
      const help = document.createElement("p");
      help.className = "param-help";
      help.textContent = p.help;
      row.appendChild(help);
    }
    return row;
  }
  function paramUnit(p: ParamSpec): string {
    if (p.name === "transaction_cost_bps") return "bps";
    if (p.name === "long_quantile") return "frac";
    if (p.name.endsWith("_days") || p.name === "signal_lag") return "days";
    return "";
  }

  function readParams(): RunPayload {
    const s = state.currentStrategy!;
    const strategyParams: Record<string, number> = {};
    for (const p of s.params) {
      const el = document.getElementById(`strat-${p.name}`) as HTMLInputElement;
      strategyParams[p.name] = p.type === "int" ? parseInt(el.value, 10) : parseFloat(el.value);
    }
    const engineParams: Record<string, number> = {};
    for (const p of state.catalog!.engine_params) {
      const el = document.getElementById(`eng-${p.name}`) as HTMLInputElement;
      engineParams[p.name] = p.type === "int" ? parseInt(el.value, 10) : parseFloat(el.value);
    }
    const engineOverrides: Record<string, number> = {};
    for (const p of (s.engine_overrides || [])) {
      const el = document.getElementById(`ovr-${p.name}`) as HTMLInputElement | null;
      if (!el) continue;
      engineOverrides[p.name] = p.type === "int" ? parseInt(el.value, 10) : parseFloat(el.value);
    }
    return {
      strategyParams,
      engineParams,
      engineOverrides,
      tickers: state.customTickers.slice(),
      start: ($("start") as HTMLInputElement).value,
      end: ($("end") as HTMLInputElement).value,
    };
  }

  function updateSizing(): void {
    const el = document.getElementById("strategy-sizing");
    if (!el) return;
    const { engineParams, engineOverrides } = readParams();
    const longQ = engineParams.long_quantile ?? 0.2;
    const shortQ = engineOverrides.short_quantile ?? 0;
    const universeSize = state.universeSize || state.fullUniverseSize || 0;
    const longNames = Math.max(1, Math.round(universeSize * longQ));
    const shortNames = shortQ > 0 ? Math.max(1, Math.round(universeSize * shortQ)) : 0;
    const longPct = (longQ * 100).toFixed(0);
    const shortPct = (shortQ * 100).toFixed(0);
    if (shortQ > 0) {
      el.innerHTML =
        `<span class="sizing-tag">LONG</span> top ${longPct}% (~${longNames} of ${universeSize}) · 50% of capital. ` +
        `<span class="sizing-short">SHORT</span> bottom ${shortPct}% (~${shortNames}) · 50% of capital. ` +
        `Net 0%, gross 100%.`;
    } else {
      el.innerHTML =
        `<span class="sizing-tag">LONG</span> top ${longPct}% (~${longNames} of ${universeSize}) · 100% of capital. ` +
        `No short leg.`;
    }
  }

  // ---------- Live code ----------
  function pyRepr(v: unknown): string {
    if (typeof v === "string") return `"${v}"`;
    if (typeof v === "boolean") return v ? "True" : "False";
    if (typeof v === "number" && Number.isInteger(v)) return `${v}`;
    return `${(+(v as number)).toFixed(2)}`;
  }
  function renderLiveCode(): void {
    const s = state.currentStrategy!;
    const { strategyParams, engineParams, engineOverrides } = readParams();
    const stratArgs = Object.entries(strategyParams)
      .map(([k, v]) => `    ${k}=${pyRepr(v)},`)
      .join("\n");
    const stratCall = stratArgs
      ? `strategy = ${s.cls_name}(\n${stratArgs}\n)`
      : `strategy = ${s.cls_name}()`;

    // Merge: live knobs + per-strategy overrides win; fixed defaults fill in.
    const fixedEngine: Record<string, unknown> = { ...state.catalog!.fixed_engine };
    for (const k of Object.keys(engineOverrides)) delete fixedEngine[k];
    if ((engineOverrides.short_quantile ?? 0) > 0) fixedEngine.long_only = false;
    const mergedLive: Record<string, number> = { ...engineParams, ...engineOverrides };

    const liveCfgLines = Object.entries(mergedLive)
      .map(([k, v]) => `    ${k}=${pyRepr(v)},`)
      .join("\n");
    const fixedLines = Object.entries(fixedEngine)
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
    updateSizing();
  }

  // ---------- Run ----------
  let runTimer: number | null = null;
  function scheduleRun(): void {
    if (runTimer !== null) clearTimeout(runTimer);
    runTimer = window.setTimeout(run, 350);
  }
  ($("start") as HTMLInputElement).addEventListener("change", () => {
    renderLiveCode();
    scheduleRun();
  });
  ($("end") as HTMLInputElement).addEventListener("change", () => {
    renderLiveCode();
    scheduleRun();
  });

  async function run(): Promise<void> {
    if (state.running) {
      state.pending = true;
      return;
    }
    state.running = true;
    setStatus("running simulation…", "loading");
    try {
      const p = readParams();
      // Send tickers:null only when NO custom filter is active. An empty
      // array with hasCustomFilter=true means "user explicitly narrowed
      // to nothing" and should bubble up as a backend error, not silently
      // flip back to the full universe.
      const tickers = state.hasCustomFilter ? p.tickers : null;
      const body = {
        strategy_id: state.currentStrategy!.id,
        strategy_params: p.strategyParams,
        engine_params: p.engineParams,
        engine_overrides: p.engineOverrides,
        tickers,
        start: p.start,
        end: p.end,
      };
      const t0 = performance.now();
      const res = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = (await res.json().catch(() => ({ error: `HTTP ${res.status}` }))) as {
          error?: string;
        };
        throw new Error(err.error || `HTTP ${res.status}`);
      }
      const data = (await res.json()) as SimulationResult;
      const elapsed = performance.now() - t0;
      renderResults(data);
      setStatus(`ready · last run ${(elapsed / 1000).toFixed(2)}s`, "ready");
    } catch (exc) {
      console.error(exc);
      const msg = exc instanceof Error ? exc.message : String(exc);
      toast(`Error: ${msg}`, true, 6000);
      setStatus(`error: ${msg}`, "error");
    } finally {
      state.running = false;
      if (state.pending) {
        state.pending = false;
        scheduleRun();
      }
    }
  }
  function renderResults(data: SimulationResult): void {
    const m: Metrics = data.metrics_net || {};
    const b: Metrics = data.metrics_benchmark || {};
    const o: OrderSummary = data.order_summary || {};
    const setCard = (
      valId: string,
      subId: string,
      value: string,
      cls: string,
      sub: string,
    ): void => {
      const el = $(valId);
      el.textContent = value;
      el.className = "m-value" + (cls ? ` ${cls}` : "");
      $(subId).textContent = sub;
    };
    setCard(
      "m-sharpe",
      "m-sharpe-sub",
      fmtNum(m.sharpe),
      (m.sharpe ?? 0) >= 1 ? "good" : ((m.sharpe as number) < 0.3 ? "bad" : ""),
      `bench ${fmtNum(b.sharpe)}`,
    );
    setCard(
      "m-return",
      "m-return-sub",
      fmtPct(m.annualized_return),
      (m.annualized_return ?? 0) >= 0 ? "good" : "bad",
      `bench ${fmtPct(b.annualized_return)}`,
    );
    setCard(
      "m-dd",
      "m-dd-sub",
      fmtPct(m.max_drawdown),
      "bad",
      `bench ${fmtPct(b.max_drawdown)}`,
    );
    setCard(
      "m-tcost",
      "m-tcost-sub",
      fmtPct(o.tcost_drag_annualized),
      "bad",
      `turnover ${fmtX(o.turnover_annualized, 1)}/yr`,
    );

    Plotly.react(
      "chart-equity",
      [
        {
          x: data.dates,
          y: data.cumulative_benchmark.map((v) => v * 100),
          name: "Equal-weight universe",
          type: "scatter",
          mode: "lines",
          line: { color: "#7a8a9c", width: 1.5, dash: "dot" },
        },
        {
          x: data.dates,
          y: data.cumulative_net.map((v) => v * 100),
          name: "Strategy (net of t-cost)",
          type: "scatter",
          mode: "lines",
          line: { color: "#58c1ff", width: 2.3 },
        },
      ],
      {
        ...plotLayout,
        yaxis: {
          ...(plotLayout.yaxis as Record<string, unknown>),
          title: "Cumulative return (%)",
          tickformat: ",.0f",
        },
      },
      plotConfig,
    );

    renderAudit(data.survivorship_audit || {});
    if (data.universe && typeof data.universe.n_tickers === "number") {
      state.universeSize = data.universe.n_tickers;
      updateSizing();
    }
  }

  function renderAudit(a: SurvivorshipAudit): void {
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
    const bias: BiasRange = a.expected_annual_upward_bias_pct || {};
    const biasTxt =
      bias.low != null && bias.high != null
        ? `${bias.low.toFixed(1)}–${bias.high.toFixed(1)}% / yr`
        : "—";
    sub.textContent = `${a.universe_size} survivors · ${a.window_start} → ${a.window_end} (${a.window_years}y)`;

    const inceptionCls = (a.inception_biased_count ?? 0) > 0 ? "warn" : "";
    const delistCls = (a.delisted_within_window_count ?? 0) > 0 ? "bad" : "";

    statsEl.innerHTML = `
      <div class="a-cell"><div class="a-label">Universe size</div><div class="a-value">${a.universe_size}</div></div>
      <div class="a-cell"><div class="a-label">Est. upward bias</div><div class="a-value warn">${biasTxt}</div></div>
      <div class="a-cell"><div class="a-label">Active at start</div><div class="a-value">${a.active_at_start}</div></div>
      <div class="a-cell"><div class="a-label">Active at end</div><div class="a-value">${a.active_at_end}</div></div>
      <div class="a-cell"><div class="a-label">Post-start inceptions</div><div class="a-value ${inceptionCls}">${a.inception_biased_count}</div></div>
      <div class="a-cell"><div class="a-label">Delistings in window</div><div class="a-value ${delistCls}">${a.delisted_within_window_count}</div></div>
    `;
    noteEl.textContent = a.structural_bias_note || "";

    const ts: AuditTimeseries = a.active_timeseries || { dates: [], counts: [] };
    Plotly.react(
      "chart-audit",
      [
        {
          x: ts.dates,
          y: ts.counts,
          type: "scatter",
          mode: "lines",
          name: "Active tickers",
          line: { color: "#58c1ff", width: 1.8 },
          fill: "tozeroy",
          fillcolor: "rgba(88,193,255,0.08)",
        },
      ],
      {
        ...plotLayout,
        margin: { t: 10, r: 10, b: 30, l: 48 },
        height: 220,
        showlegend: false,
        yaxis: {
          ...(plotLayout.yaxis as Record<string, unknown>),
          title: "Active tickers",
          rangemode: "tozero",
        },
      },
      plotConfig,
    );
  }

  // ---------- Data inspector ----------
  async function loadDataOverview(): Promise<void> {
    const res = await fetch("/api/data");
    if (!res.ok) throw new Error(`data HTTP ${res.status}`);
    state.dataOverview = (await res.json()) as DataOverview;
    $("data-sub").textContent = `${state.dataOverview.n_tickers} tickers · ${state.dataOverview.source_file}`;
    renderTickerTable();
  }
  function renderTickerTable(): void {
    const body = $("ticker-tbody");
    body.innerHTML = "";
    const q = state.tickerFilter.toLowerCase();
    const rows = state.dataOverview!.tickers.filter(
      (r) =>
        !q ||
        r.ticker.toLowerCase().includes(q) ||
        (r.name || "").toLowerCase().includes(q) ||
        (r.sector || "").toLowerCase().includes(q),
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
  function escapeHtml(s: string): string {
    return s.replace(
      /[&<>"']/g,
      (c) =>
        (({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }) as Record<
          string,
          string
        >)[c],
    );
  }
  async function selectTicker(tic: string): Promise<void> {
    state.selectedTicker = tic;
    document.querySelectorAll<HTMLTableRowElement>("#ticker-table tbody tr").forEach((tr) => {
      tr.classList.toggle("active", tr.dataset.ticker === tic);
    });
    const right = $("data-right");
    right.innerHTML = `<div class="data-empty">Loading ${tic}…</div>`;
    try {
      const res = await fetch(`/api/data/ticker/${encodeURIComponent(tic)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const d = (await res.json()) as TickerDetail;
      renderTickerDetail(d);
    } catch (exc) {
      const msg = exc instanceof Error ? exc.message : String(exc);
      right.innerHTML = `<div class="data-empty">error: ${msg}</div>`;
    }
  }
  function renderTickerDetail(d: TickerDetail): void {
    const right = $("data-right");
    const metaBits = Object.entries(d.metadata)
      .map(([k, v]) => `${k}=${v}`)
      .join(" · ");
    right.innerHTML = `
      <div class="detail-head">
        <span class="detail-sym">${d.ticker}</span>
        <span class="detail-name">${escapeHtml((d.metadata.conm as string) || "")}</span>
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
    Plotly.react(
      "detail-chart",
      [
        {
          x: d.dates,
          y: d.prices,
          type: "scatter",
          mode: "lines",
          line: { color: "#58c1ff", width: 1.6 },
          hovertemplate: "%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
        },
      ],
      {
        ...plotLayout,
        margin: { t: 8, r: 8, b: 28, l: 48 },
        yaxis: {
          ...(plotLayout.yaxis as Record<string, unknown>),
          title: "Price",
          type: "log",
          tickformat: ".2f",
        },
        showlegend: false,
      },
      plotConfig,
    );
  }
  ($("ticker-search") as HTMLInputElement).addEventListener("input", (e) => {
    state.tickerFilter = (e.target as HTMLInputElement).value;
    renderTickerTable();
  });

  // ---------- Universe editor ----------
  interface UniverseAddResponse {
    requested: string[];
    already_cached: string[];
    fetched: string[];
    failed: { ticker: string; error?: string }[];
    universe_size_after: number;
    error?: string;
  }
  function parseUniverseText(raw: string): string[] {
    return (raw || "")
      .replace(/\n/g, ",")
      .split(",")
      .map((t) => t.trim().toUpperCase().replace(/\./g, "-"))
      .filter(Boolean);
  }
  function refreshUniverseStatus(): void {
    const status = document.getElementById("universe-status");
    if (!status) return;
    if (!state.hasCustomFilter) {
      status.textContent = `all cached (${state.fullUniverseSize || "?"})`;
    } else {
      const n = state.customTickers.length;
      status.textContent = `custom: ${n} ticker${n === 1 ? "" : "s"}`;
    }
  }
  const universeInput = document.getElementById("universe-input") as HTMLTextAreaElement | null;
  const applyBtn = document.getElementById("universe-apply");
  const fetchBtn = document.getElementById("universe-fetch") as HTMLButtonElement | null;
  const resetBtn = document.getElementById("universe-reset");
  if (applyBtn && universeInput) {
    applyBtn.addEventListener("click", () => {
      const tickers = parseUniverseText(universeInput.value);
      if (!tickers.length) {
        toast("paste tickers to restrict to", true);
        return;
      }
      state.customTickers = tickers;
      state.hasCustomFilter = true;
      refreshUniverseStatus();
      updateSizing();
      scheduleRun();
    });
  }
  if (resetBtn && universeInput) {
    resetBtn.addEventListener("click", () => {
      universeInput.value = "";
      state.customTickers = [];
      state.hasCustomFilter = false;
      refreshUniverseStatus();
      updateSizing();
      scheduleRun();
    });
  }
  if (fetchBtn && universeInput) {
    // "Exclude" — take the typed list out of the current cached universe.
    // Does NOT call /api/universe/add, because that endpoint rebuilds the
    // global dataset from every cached pickle and would permanently add
    // typed non-SP500 names to the baseline (Codex adversarial finding).
    fetchBtn.addEventListener("click", () => {
      const excluded = parseUniverseText(universeInput.value);
      if (!excluded.length) {
        toast("paste tickers to exclude", true);
        return;
      }
      const allCached = (state.dataOverview?.tickers || []).map((r) => r.ticker);
      if (!allCached.length) {
        toast("data inspector hasn't loaded yet — try again in a second", true);
        return;
      }
      const excludeSet = new Set(excluded);
      const complement = allCached.filter((t) => !excludeSet.has(t));
      if (!complement.length) {
        toast("exclusion list covers every cached ticker — no universe left to run", true, 6000);
        return;
      }
      const matched = allCached.filter((t) => excludeSet.has(t)).length;
      const missing = excluded.length - matched;
      state.customTickers = complement;
      state.hasCustomFilter = true;
      const suffix = missing ? ` · ${missing} not in cache (ignored)` : "";
      toast(
        `excluded ${matched} · running on ${complement.length} ticker${complement.length === 1 ? "" : "s"}${suffix}`,
        false,
        4000,
      );
      refreshUniverseStatus();
      updateSizing();
      scheduleRun();
    });
  }

  // ---------- Boot ----------
  async function waitReady(): Promise<boolean> {
    while (true) {
      try {
        const res = await fetch("/api/status");
        const s = (await res.json()) as StatusResponse;
        if (s.error) {
          setStatus(`error: ${s.error}`, "error");
          return false;
        }
        if (s.ready) {
          setStatus(`ready · ${s.tickers} tickers · ${s.date_min} → ${s.date_max}`, "ready");
          const startEl = $("start") as HTMLInputElement;
          const endEl = $("end") as HTMLInputElement;
          startEl.min = s.date_min ?? "";
          startEl.max = s.date_max ?? "";
          endEl.min = s.date_min ?? "";
          endEl.max = s.date_max ?? "";
          const label = s.universe_label || `${s.tickers} tickers`;
          $("universe-label").textContent = `universe: ${label}`;
          state.universeSize = s.tickers || 0;
          state.fullUniverseSize = s.tickers || 0;
          refreshUniverseStatus();
          return true;
        }
        setStatus(`loading · ${s.message}`, "loading");
      } catch (exc) {
        setStatus("backend unreachable", "error");
        return false;
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
  }

  (async (): Promise<void> => {
    setStatus("connecting…", "loading");
    if (!(await waitReady())) return;
    state.catalog = await loadCatalog();
    renderStrategyTabs();
    selectStrategy(state.catalog.strategies[0].id);
    loadDataOverview().catch((e) => console.error(e));
  })();
})();
