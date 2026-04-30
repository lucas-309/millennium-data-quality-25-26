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
  data_source?: string;
  available_sources?: string[];
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
  loading?: boolean;
  error?: string;
  tickers?: number;
  date_min?: string;
  date_max?: string;
  message?: string;
  universe_label?: string;
  data_source?: string;
  available_sources?: string[];
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
  pinnedRuns: PinnedRun[];
  dataSource: string;
  availableSources: string[];
}

interface SimRequest {
  strategy_id: string;
  strategy_params: Record<string, number>;
  engine_params: Record<string, number>;
  engine_overrides: Record<string, number>;
  tickers: string[] | null;
  start: string;
  end: string;
}

interface PinnedRun {
  id: string;
  label: string;          // user-editable; auto-generated from strategy + params
  strategyLabel: string;  // for display in chip subtext
  colorIdx: number;       // index into series-color rotation
  dates: string[];
  cumulativeNet: number[]; // already in percent (× 100), like the live trace
  payload: SimRequest;     // snapshot of the request that produced this curve
  metricsNet?: Metrics;
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

  const PINNED_STORAGE_KEY = "sim.pinned.v1";
  const MAX_PINNED = 6;
  // Series-color rotation for pinned runs. The current run keeps its own
  // signature color (--q, cinnabar) so the "live" curve always reads the
  // same; pinned runs cycle through the editorial palette below.
  const SERIES_VARS = [
    "--ser-a",
    "--ser-b",
    "--ser-c",
    "--ser-d",
    "--ser-e",
    "--ser-f",
  ];

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
    pinnedRuns: loadPinnedFromStorage(),
    dataSource: "yfinance",
    availableSources: ["yfinance", "wharton"],
  };

  function loadPinnedFromStorage(): PinnedRun[] {
    try {
      const raw = localStorage.getItem(PINNED_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw) as PinnedRun[];
      if (!Array.isArray(parsed)) return [];
      // Quick shape guard. Anything malformed is dropped silently.
      return parsed.filter(
        (p) =>
          typeof p === "object" &&
          p !== null &&
          typeof p.id === "string" &&
          Array.isArray(p.dates) &&
          Array.isArray(p.cumulativeNet),
      );
    } catch (_) {
      return [];
    }
  }
  function savePinnedToStorage(): void {
    try {
      localStorage.setItem(
        PINNED_STORAGE_KEY,
        JSON.stringify(state.pinnedRuns),
      );
    } catch (_) {
      /* localStorage full / disabled — silently drop persistence */
    }
  }
  function nextFreeColorIdx(): number {
    const used = new Set(state.pinnedRuns.map((r) => r.colorIdx));
    for (let i = 0; i < SERIES_VARS.length; i++) {
      if (!used.has(i)) return i;
    }
    // All slots claimed — reuse the oldest position.
    return state.pinnedRuns.length % SERIES_VARS.length;
  }

  // Charts read their palette from CSS variables so the theme toggle can
  // repaint them. Keep this in sync with the :root[data-theme=…] block in
  // style.css — q is the strategy line, k is the benchmark.
  const cssVar = (name: string): string =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function chartPalette(): {
    bg: string; ink: string; inkSoft: string; muted: string;
    rule: string; ruleSoft: string;
    q: string; qGhost: string; k: string; kGhost: string;
    accent: string; accentTint: string;
    series: string[];
  } {
    return {
      bg:         cssVar("--bg-tint"),
      ink:        cssVar("--ink"),
      inkSoft:    cssVar("--ink-soft"),
      muted:      cssVar("--muted"),
      rule:       cssVar("--rule"),
      ruleSoft:   cssVar("--rule-soft"),
      q:          cssVar("--q"),
      qGhost:     cssVar("--q-ghost"),
      k:          cssVar("--k"),
      kGhost:     cssVar("--k-ghost"),
      accent:     cssVar("--accent"),
      accentTint: cssVar("--accent-tint"),
      series:     SERIES_VARS.map((v) => cssVar(v)),
    };
  }

  function plotLayout(): Record<string, unknown> {
    const c = chartPalette();
    return {
      paper_bgcolor: c.bg,
      plot_bgcolor: c.bg,
      font: {
        color: c.inkSoft,
        family: "-apple-system, Inter, 'Segoe UI', Roboto, sans-serif",
        size: 12,
      },
      margin: { t: 16, r: 16, b: 36, l: 56 },
      xaxis: {
        gridcolor: c.ruleSoft, zerolinecolor: c.rule,
        linecolor: c.rule, tickcolor: c.rule,
        tickfont: { color: c.inkSoft },
      },
      yaxis: {
        gridcolor: c.ruleSoft, zerolinecolor: c.rule,
        linecolor: c.rule, tickcolor: c.rule,
        tickfont: { color: c.inkSoft },
      },
      legend: { orientation: "h", y: -0.2, x: 0, font: { color: c.inkSoft } },
      hovermode: "x unified",
      hoverlabel: { bgcolor: c.bg, bordercolor: c.rule, font: { color: c.ink } },
    };
  }
  const plotConfig: Record<string, unknown> = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };

  // Stashed copies of the most recent payloads so the theme toggle can
  // re-render charts with the new palette without re-fetching, and so
  // "Pin to overlay" can capture the request that produced the curve.
  const lastRender: {
    sim?: SimulationResult;
    simPayload?: SimRequest;
    ticker?: TickerDetail;
  } = {};

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
    const resetBtn = document.createElement("button");
    resetBtn.type = "button";
    resetBtn.className = "param-reset";
    resetBtn.title = `reset to default ${p.default}`;
    resetBtn.setAttribute("aria-label", `reset ${p.name} to ${p.default}`);
    resetBtn.textContent = "↺";
    valWrap.appendChild(resetBtn);

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

    const updateResetState = (): void => {
      const cur = parseFloat(numInp.value);
      resetBtn.classList.toggle(
        "active",
        Number.isFinite(cur) && cur !== +p.default,
      );
    };

    const syncFromSlider = (): void => {
      numInp.value = slider.value;
      updateResetState();
      renderLiveCode();
      scheduleRun();
    };
    const syncFromNumber = (): void => {
      // Don't clamp while user is typing — just mirror the value into the
      // slider (slider clips visually) and trigger a debounced run.
      const v = parseFloat(numInp.value);
      if (!Number.isFinite(v)) return;
      slider.value = String(v);
      updateResetState();
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
      updateResetState();
      renderLiveCode();
      scheduleRun();
    });
    resetBtn.addEventListener("click", () => {
      numInp.value = String(p.default);
      slider.value = String(p.default);
      updateResetState();
      renderLiveCode();
      scheduleRun();
    });
    updateResetState();

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
    setProgress(20);
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
    setProgress(60);
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
      lastRender.simPayload = body;
      renderResults(data);
      setStatus(`ready · last run ${(elapsed / 1000).toFixed(2)}s`, "ready");
    } catch (exc) {
      console.error(exc);
      const msg = exc instanceof Error ? exc.message : String(exc);
      toast(`Error: ${msg}`, true, 6000);
      setStatus(`error: ${msg}`, "error");
    } finally {
      state.running = false;
      progressDone();
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

    lastRender.sim = data;
    const eqPal = chartPalette();
    const eqLayout = plotLayout();
    const liveLabel = state.currentStrategy
      ? autoRunLabel(state.currentStrategy, lastRender.simPayload)
      : "Strategy";
    const traces: Record<string, unknown>[] = [
      {
        x: data.dates,
        y: data.cumulative_benchmark.map((v) => v * 100),
        name: "Equal-weight universe",
        type: "scatter",
        mode: "lines",
        line: { color: eqPal.k, width: 1.5, dash: "dot" },
        hovertemplate: "%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>Benchmark</extra>",
      },
    ];
    for (const pinned of state.pinnedRuns) {
      const color = eqPal.series[pinned.colorIdx % eqPal.series.length];
      traces.push({
        x: pinned.dates,
        y: pinned.cumulativeNet,
        name: pinned.label,
        type: "scatter",
        mode: "lines",
        line: { color, width: 1.6 },
        opacity: 0.85,
        hovertemplate: `%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>${escapeHtml(pinned.label)}</extra>`,
      });
    }
    traces.push({
      x: data.dates,
      y: data.cumulative_net.map((v) => v * 100),
      name: `${liveLabel} · live`,
      type: "scatter",
      mode: "lines",
      line: { color: eqPal.q, width: 2.2 },
      hovertemplate: `%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>${escapeHtml(liveLabel)}</extra>`,
    });
    Plotly.react(
      "chart-equity",
      traces,
      {
        ...eqLayout,
        yaxis: {
          ...(eqLayout.yaxis as Record<string, unknown>),
          title: "Cumulative return (%)",
          tickformat: ",.0f",
          ticksuffix: "%",
          automargin: true,
        },
        xaxis: {
          ...(eqLayout.xaxis as Record<string, unknown>),
          automargin: true,
        },
      },
      plotConfig,
    );

    renderAudit(data.survivorship_audit || {});
    renderPinnedList();
    if (data.universe && typeof data.universe.n_tickers === "number") {
      state.universeSize = data.universe.n_tickers;
      updateSizing();
    }
  }

  // ---------- Pinned-run overlay ----------
  function autoRunLabel(strategy: Strategy, payload?: SimRequest): string {
    if (!payload) return strategy.label;
    const sp = payload.strategy_params || {};
    const eo = payload.engine_overrides || {};
    const bits: string[] = [];
    for (const p of strategy.params) {
      const v = sp[p.name];
      if (v == null) continue;
      bits.push(`${shortName(p.name)}=${pyRepr(v)}`);
    }
    for (const p of strategy.engine_overrides || []) {
      const v = eo[p.name];
      if (v == null) continue;
      bits.push(`${shortName(p.name)}=${pyRepr(v)}`);
    }
    return bits.length ? `${strategy.label} · ${bits.join(", ")}` : strategy.label;
  }
  function shortName(name: string): string {
    // Compact param names for chip labels: "lookback_days" → "lb",
    // "skip_days" → "skip", etc.
    if (name === "lookback_days") return "lb";
    if (name === "skip_days") return "skip";
    if (name === "transaction_cost_bps") return "tc";
    if (name === "long_quantile") return "lq";
    if (name === "short_quantile") return "sq";
    if (name === "signal_lag") return "lag";
    return name;
  }
  function payloadFingerprint(p: SimRequest): string {
    const tickerHash = p.tickers ? `[${p.tickers.length}]` : "all";
    return [
      p.strategy_id,
      JSON.stringify(p.strategy_params),
      JSON.stringify(p.engine_params),
      JSON.stringify(p.engine_overrides),
      p.start,
      p.end,
      tickerHash,
    ].join("|");
  }
  function pinCurrentRun(): void {
    if (!lastRender.sim || !lastRender.simPayload || !state.currentStrategy) {
      toast("nothing to pin yet — run a backtest first", true);
      return;
    }
    if (state.pinnedRuns.length >= MAX_PINNED) {
      toast(`overlay holds at most ${MAX_PINNED} runs — remove one first`, true);
      return;
    }
    const fp = payloadFingerprint(lastRender.simPayload);
    const dupe = state.pinnedRuns.some(
      (r) => payloadFingerprint(r.payload) === fp,
    );
    if (dupe) {
      toast("this run is already pinned", false, 2500);
      return;
    }
    const colorIdx = nextFreeColorIdx();
    const pinned: PinnedRun = {
      id:
        typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      label: autoRunLabel(state.currentStrategy, lastRender.simPayload),
      strategyLabel: state.currentStrategy.label,
      colorIdx,
      dates: lastRender.sim.dates.slice(),
      cumulativeNet: lastRender.sim.cumulative_net.map((v) => v * 100),
      payload: JSON.parse(JSON.stringify(lastRender.simPayload)) as SimRequest,
      metricsNet: lastRender.sim.metrics_net,
    };
    state.pinnedRuns.push(pinned);
    savePinnedToStorage();
    if (lastRender.sim) renderResults(lastRender.sim);
    else renderPinnedList();
  }
  function removePinnedRun(id: string): void {
    const before = state.pinnedRuns.length;
    state.pinnedRuns = state.pinnedRuns.filter((r) => r.id !== id);
    if (state.pinnedRuns.length === before) return;
    savePinnedToStorage();
    if (lastRender.sim) renderResults(lastRender.sim);
    else renderPinnedList();
  }
  function clearAllPinnedRuns(): void {
    if (!state.pinnedRuns.length) return;
    state.pinnedRuns = [];
    savePinnedToStorage();
    if (lastRender.sim) renderResults(lastRender.sim);
    else renderPinnedList();
  }
  function restorePinnedConfig(id: string): void {
    const pinned = state.pinnedRuns.find((r) => r.id === id);
    if (!pinned) return;
    if (!state.catalog) return;
    const strat = state.catalog.strategies.find(
      (s) => s.id === pinned.payload.strategy_id,
    );
    if (!strat) {
      toast(`strategy ${pinned.payload.strategy_id} not in catalog`, true);
      return;
    }
    // Switch to the right strategy first, then push the pinned values
    // into the input controls. selectStrategy already calls scheduleRun;
    // we delay our writes until *after* the new param rows mount.
    selectStrategy(strat.id);
    requestAnimationFrame(() => {
      const writeInput = (id: string, v: number): void => {
        const el = document.getElementById(id) as HTMLInputElement | null;
        if (!el) return;
        el.value = String(v);
        el.dispatchEvent(new Event("input", { bubbles: true }));
      };
      for (const [k, v] of Object.entries(pinned.payload.strategy_params)) {
        writeInput(`strat-${k}`, v);
      }
      for (const [k, v] of Object.entries(pinned.payload.engine_params)) {
        writeInput(`eng-${k}`, v);
      }
      for (const [k, v] of Object.entries(pinned.payload.engine_overrides)) {
        writeInput(`ovr-${k}`, v);
      }
      const startEl = document.getElementById("start") as HTMLInputElement | null;
      const endEl = document.getElementById("end") as HTMLInputElement | null;
      if (startEl) startEl.value = pinned.payload.start;
      if (endEl) endEl.value = pinned.payload.end;
      renderLiveCode();
      scheduleRun();
    });
  }
  function renderPinnedList(): void {
    const host = document.getElementById("pinned-list");
    const empty = document.getElementById("pinned-empty");
    const clearBtn = document.getElementById("pinned-clear") as HTMLButtonElement | null;
    if (!host) return;
    host.innerHTML = "";
    const pal = chartPalette();
    if (!state.pinnedRuns.length) {
      if (empty) (empty as HTMLElement).hidden = false;
      if (clearBtn) clearBtn.disabled = true;
      return;
    }
    if (empty) (empty as HTMLElement).hidden = true;
    if (clearBtn) clearBtn.disabled = false;
    for (const pinned of state.pinnedRuns) {
      const color = pal.series[pinned.colorIdx % pal.series.length];
      const sharpe = pinned.metricsNet?.sharpe;
      const ret = pinned.metricsNet?.annualized_return;
      const chip = document.createElement("div");
      chip.className = "pinned-chip";
      chip.innerHTML = `
        <span class="pinned-swatch" style="background: ${color}"></span>
        <div class="pinned-body">
          <div class="pinned-label" title="click to rename">${escapeHtml(pinned.label)}</div>
          <div class="pinned-meta">
            ${sharpe != null ? `Sharpe ${sharpe.toFixed(2)}` : ""}
            ${ret != null ? ` · ann ${(ret * 100).toFixed(1)}%` : ""}
            ${pinned.payload.start} → ${pinned.payload.end}
          </div>
        </div>
        <button type="button" class="pinned-restore" title="reload these params into the controls" aria-label="reload params">↺</button>
        <button type="button" class="pinned-remove" title="remove from overlay" aria-label="remove">×</button>
      `;
      const labelEl = chip.querySelector(".pinned-label") as HTMLDivElement;
      labelEl.addEventListener("click", () => {
        const next = window.prompt("Rename pinned run:", pinned.label);
        if (next != null && next.trim() && next.trim() !== pinned.label) {
          pinned.label = next.trim();
          savePinnedToStorage();
          if (lastRender.sim) renderResults(lastRender.sim);
          else renderPinnedList();
        }
      });
      (chip.querySelector(".pinned-restore") as HTMLButtonElement).addEventListener(
        "click",
        () => restorePinnedConfig(pinned.id),
      );
      (chip.querySelector(".pinned-remove") as HTMLButtonElement).addEventListener(
        "click",
        () => removePinnedRun(pinned.id),
      );
      host.appendChild(chip);
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
    const auPal = chartPalette();
    const auLayout = plotLayout();
    Plotly.react(
      "chart-audit",
      [
        {
          x: ts.dates,
          y: ts.counts,
          type: "scatter",
          mode: "lines",
          name: "Active tickers",
          line: { color: auPal.accent, width: 1.6 },
          fill: "tozeroy",
          fillcolor: auPal.accentTint,
        },
      ],
      {
        ...auLayout,
        margin: { t: 10, r: 10, b: 30, l: 12 },
        height: 220,
        showlegend: false,
        yaxis: {
          ...(auLayout.yaxis as Record<string, unknown>),
          title: "Active tickers",
          rangemode: "tozero",
          automargin: true,
          nticks: 4,
        },
        xaxis: {
          ...(auLayout.xaxis as Record<string, unknown>),
          automargin: true,
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
    lastRender.ticker = d;
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
    const tdPal = chartPalette();
    const tdLayout = plotLayout();
    // Plotly's `nticks` is only a soft hint on log axes — its default
    // subdivision (D2: 1/2/5 per decade) puts 6–7 labels on a 2-decade
    // chart, and they collide vertically in this pane. On linear axes
    // `nticks` is a hard cap and never bunches. We just use linear for
    // every stock; the chart still reads fine for wide ranges.
    const baseYAxis = tdLayout.yaxis as Record<string, unknown>;
    const yaxis = {
      ...baseYAxis,
      title: "Price",
      tickmode: "auto",
      nticks: 5,
      tickformat: "$,.2~f",
      automargin: true,
      tickfont: { size: 11, color: tdPal.inkSoft },
    };
    Plotly.react(
      "detail-chart",
      [
        {
          x: d.dates,
          y: d.prices,
          type: "scatter",
          mode: "lines",
          line: { color: tdPal.ink, width: 1.4 },
          hovertemplate: "%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
        },
      ],
      {
        ...tdLayout,
        // Let automargin claim the left margin so long $-formatted
        // labels (e.g. "$1,234.56") never get clipped.
        margin: { t: 8, r: 12, b: 32, l: 12 },
        yaxis,
        xaxis: {
          ...(tdLayout.xaxis as Record<string, unknown>),
          automargin: true,
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

  // ---------- Pin / overlay controls ----------
  const pinBtn = document.getElementById("pin-current");
  if (pinBtn) pinBtn.addEventListener("click", () => pinCurrentRun());
  const clearAllBtn = document.getElementById("pinned-clear");
  if (clearAllBtn) clearAllBtn.addEventListener("click", () => clearAllPinnedRuns());

  // ---------- Theme toggle ----------
  type Theme = "light" | "dark";
  function applyTheme(t: Theme): void {
    document.documentElement.setAttribute("data-theme", t);
    try { localStorage.setItem("sim-theme", t); } catch (_) { /* ignore */ }
    // Repaint any charts that already have data so their backgrounds and
    // line colors track the new palette.
    if (lastRender.sim) renderResults(lastRender.sim);
    if (lastRender.ticker) renderTickerDetail(lastRender.ticker);
  }
  const themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const cur = (document.documentElement.getAttribute("data-theme") as Theme) || "light";
      applyTheme(cur === "light" ? "dark" : "light");
    });
  }

  // ---------- Progress bar ----------
  let progressResetTimer: number | null = null;
  function setProgress(pct: number, idle = false): void {
    const el = document.getElementById("progress");
    if (!el) return;
    // A non-idle update means a new run is starting — cancel any pending
    // reset from a previous run so it can't clobber us back to 0.
    if (!idle && progressResetTimer !== null) {
      clearTimeout(progressResetTimer);
      progressResetTimer = null;
    }
    el.style.setProperty("--p", `${pct}%`);
    el.classList.toggle("idle", idle);
  }
  function progressDone(): void {
    setProgress(100);
    if (progressResetTimer !== null) clearTimeout(progressResetTimer);
    progressResetTimer = window.setTimeout(() => {
      progressResetTimer = null;
      setProgress(0, true);
    }, 320);
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
        if (s.data_source) state.dataSource = s.data_source;
        if (s.available_sources) state.availableSources = s.available_sources;
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
          updateSourceSelector();
          updateUniverseEditorMode();
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

  // ---------- Data source selector ----------
  function sourceLabel(src: string): string {
    if (src === "wharton") return "Wharton WRDS · ~865 tickers";
    return "yfinance · S&P 500";
  }
  function updateSourceSelector(): void {
    const sel = document.getElementById("source-select") as HTMLSelectElement | null;
    if (!sel) return;
    // Rebuild the option list to match the backend's reported sources;
    // mark the active one as selected.
    const sources = state.availableSources.length
      ? state.availableSources
      : ["yfinance", "wharton"];
    sel.innerHTML = "";
    for (const s of sources) {
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = sourceLabel(s);
      if (s === state.dataSource) opt.selected = true;
      sel.appendChild(opt);
    }
  }
  function updateUniverseEditorMode(): void {
    const note = document.getElementById("universe-source-note");
    const fetchBtn = document.getElementById("universe-fetch") as HTMLButtonElement | null;
    const wharton = state.dataSource === "wharton";
    if (note) {
      (note as HTMLElement).hidden = !wharton;
    }
    if (fetchBtn) {
      // The "Exclude" button doesn't actually fetch — it operates on what's
      // already in cache, which works for Wharton. So leave it enabled.
      fetchBtn.disabled = false;
    }
  }
  async function switchDataSource(target: string): Promise<void> {
    if (target === state.dataSource) return;
    state.dataSource = target;
    setStatus(`switching to ${sourceLabel(target)}…`, "loading");
    setProgress(15);
    try {
      const res = await fetch("/api/warmup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: target }),
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      // Clear pinned runs — their tickers / dates may not exist in the
      // new universe, and the chart palette is shared. Toast so the user
      // knows why their overlay just emptied.
      if (state.pinnedRuns.length) {
        const n = state.pinnedRuns.length;
        state.pinnedRuns = [];
        savePinnedToStorage();
        renderPinnedList();
        toast(`cleared ${n} pinned run${n === 1 ? "" : "s"} — new data source`, false, 4000);
      }
      // Now wait for the backend to finish reloading. Reuse waitReady's
      // polling logic so the rest of the boot flow stays consistent.
      const ok = await waitReady();
      if (!ok) return;
      state.catalog = await loadCatalog();
      const stillSelectable = state.currentStrategy
        ? state.catalog.strategies.find((s) => s.id === state.currentStrategy!.id)
        : null;
      if (!stillSelectable) {
        renderStrategyTabs();
        selectStrategy(state.catalog.strategies[0].id);
      } else {
        // Strategy still exists — just refresh the live code in case the
        // catalog reflowed.
        renderLiveCode();
        scheduleRun();
      }
      // Re-fetch the data overview so the inspector reflects the new universe.
      state.dataOverview = null;
      state.selectedTicker = null;
      const right = document.getElementById("data-right");
      if (right) right.innerHTML = `<div class="data-empty">Pick a ticker to preview its price history.</div>`;
      loadDataOverview().catch((e) => console.error(e));
    } catch (exc) {
      const msg = exc instanceof Error ? exc.message : String(exc);
      toast(`source switch failed: ${msg}`, true, 6000);
      setStatus(`error: ${msg}`, "error");
    } finally {
      progressDone();
    }
  }
  const sourceSel = document.getElementById("source-select") as HTMLSelectElement | null;
  if (sourceSel) {
    sourceSel.addEventListener("change", () => {
      const v = sourceSel.value;
      if (v) switchDataSource(v);
    });
  }

  (async (): Promise<void> => {
    setStatus("connecting…", "loading");
    renderPinnedList();
    if (!(await waitReady())) return;
    state.catalog = await loadCatalog();
    renderStrategyTabs();
    selectStrategy(state.catalog.strategies[0].id);
    loadDataOverview().catch((e) => console.error(e));
  })();
})();
