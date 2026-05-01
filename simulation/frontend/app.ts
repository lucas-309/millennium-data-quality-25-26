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
  formula?: string;
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
  annualized_volatility?: number | null;
  max_drawdown?: number | null;
  win_rate?: number | null;
  hit_rate?: number | null;
  sortino?: number | null;
  calmar?: number | null;
  profit_factor?: number | null;
  info_ratio?: number | null;
  beta?: number | null;
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

interface MonthlyReturn {
  year: number;
  month: number;
  ret: number;
}

interface SimulationResult {
  metrics_net?: Metrics;
  metrics_benchmark?: Metrics;
  order_summary?: OrderSummary;
  dates: string[];
  cumulative_benchmark: number[];
  cumulative_net: number[];
  cumulative_drawdown?: number[];
  cumulative_drawdown_benchmark?: number[];
  monthly_returns?: MonthlyReturn[];
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
  universeMissingCount: number;
  // True when the user has explicitly narrowed the universe. Lets us
  // distinguish "empty filter = user deliberately picked 0 tickers" from
  // "empty filter = default full cache", which used to get conflated and
  // run the full universe on what should have been an empty result.
  hasCustomFilter: boolean;
  pinnedRuns: PinnedRun[];
  dataSource: string;
  availableSources: string[];
  // Live code editor: per-strategy edited source, keyed by strategy id.
  // null means "no edit active" — the catalog source is rendered.
  editedSourceByStrategy: Record<string, string>;
  // Whether the editor pane is currently visible (textarea up, pre hidden).
  editing: boolean;
}

interface SimRequest {
  strategy_id: string;
  strategy_params: Record<string, number>;
  engine_params: Record<string, number>;
  engine_overrides: Record<string, number>;
  tickers: string[] | null;
  start: string;
  end: string;
  data_source?: string;
}

interface PinnedRun {
  id: string;
  label: string;          // user-editable; auto-generated from strategy + params
  strategyLabel: string;  // for display in chip subtext
  colorIdx: number;       // index into series-color rotation
  dates: string[];
  cumulativeNet: number[];      // already in percent (× 100), like the live trace
  cumulativeDrawdown?: number[]; // fraction underwater, percent (× 100)
  payload: SimRequest;          // snapshot of the request that produced this curve
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
    universeMissingCount: 0,
    hasCustomFilter: false,
    pinnedRuns: loadPinnedFromStorage(),
    dataSource: "yfinance",
    availableSources: ["yfinance", "wharton"],
    editedSourceByStrategy: {},
    editing: false,
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
  function assertValidDateRange(start: string, end: string): void {
    if (start && end && start > end) {
      throw new Error(`start date must be on or before end date (${start} > ${end})`);
    }
  }
  const sleep = (ms: number): Promise<void> =>
    new Promise((resolve) => window.setTimeout(resolve, ms));
  async function fetchJson<T>(url: string, init?: RequestInit, retries = 2): Promise<T> {
    let lastErr: unknown;
    for (let attempt = 0; attempt <= retries; attempt += 1) {
      try {
        const res = await fetch(url, init);
        if (!res.ok) {
          const err = (await res.json().catch(() => ({ error: `HTTP ${res.status}` }))) as {
            error?: string;
          };
          throw new Error(err.error || `HTTP ${res.status}`);
        }
        return (await res.json()) as T;
      } catch (exc) {
        lastErr = exc;
        if (attempt === retries) break;
        await sleep(350 * (attempt + 1));
      }
    }
    throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
  }
  function sourceQuery(src = state.dataSource): string {
    return `source=${encodeURIComponent(src)}`;
  }

  // ---------- Catalog & tabs ----------
  async function loadCatalog(): Promise<Catalog> {
    return fetchJson<Catalog>("/api/catalog");
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
  function renderStrategyHeader(s: Strategy): void {
    const host = $("strategy-summary");
    const formula = (s.formula || "").trim();
    const summary = (s.summary || "").trim();
    if (!formula && !summary) {
      host.textContent = "";
      return;
    }
    const formulaHtml = formula
      ? `<span class="formula">${escapeHtml(formula)}</span>`
      : "";
    const captionHtml = summary
      ? `<span class="formula-caption">${escapeHtml(summary)}</span>`
      : "";
    host.innerHTML = `${formulaHtml}${captionHtml}`;
  }

  function selectStrategy(id: string): void {
    const s = state.catalog!.strategies.find((x) => x.id === id);
    if (!s) return;
    state.currentStrategy = s;
    // Switching tabs always exits editor mode — but per-strategy edits are
    // preserved in editedSourceByStrategy and re-applied on return.
    state.editing = false;
    document.querySelectorAll<HTMLButtonElement>("#strategy-tabs button").forEach((b) => {
      b.classList.toggle("active", b.dataset.id === id);
    });
    renderStrategyHeader(s);
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
    // If the user edited the class source, parse the new class name so the
    // "your run" snippet reflects what actually executes.
    const editedSrc = state.editedSourceByStrategy[s.id];
    const editedClsMatch = editedSrc ? editedSrc.match(/class\s+(\w+)\s*\(/) : null;
    const clsName = editedClsMatch ? editedClsMatch[1] : s.cls_name;
    const stratCall = stratArgs
      ? `strategy = ${clsName}(\n${stratArgs}\n)`
      : `strategy = ${clsName}()`;

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

    const classSource = state.editedSourceByStrategy[s.id] ?? s.source;
    const full =
`# ── class source — ${s.source_file}
${classSource}

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
    syncEditButtons();
    updateSizing();
  }

  // ---------- Live code editor ----------
  function syncEditButtons(): void {
    const s = state.currentStrategy;
    const hasEdit = !!(s && state.editedSourceByStrategy[s.id] != null);
    const status = $("code-edit-status");
    const toggleBtn = $("code-edit-toggle") as HTMLButtonElement;
    const runBtn = $("code-edit-run") as HTMLButtonElement;
    const revertBtn = $("code-edit-revert") as HTMLButtonElement;
    const editor = $("code-editor") as HTMLTextAreaElement;
    const pre = document.querySelector(".code-block") as HTMLElement | null;

    if (state.editing) {
      toggleBtn.textContent = "Cancel";
      runBtn.hidden = false;
      revertBtn.hidden = !hasEdit;
      editor.hidden = false;
      if (pre) pre.hidden = true;
      status.textContent = "editing — Cmd+Enter to run";
    } else {
      toggleBtn.textContent = hasEdit ? "Edit (modified)" : "Edit";
      runBtn.hidden = true;
      revertBtn.hidden = !hasEdit;
      editor.hidden = true;
      if (pre) pre.hidden = false;
      status.textContent = hasEdit ? "running edited code" : "";
    }
  }
  function classSourceForCurrent(): string {
    const s = state.currentStrategy!;
    return state.editedSourceByStrategy[s.id] ?? s.source;
  }
  function enterEditMode(): void {
    const s = state.currentStrategy;
    if (!s) return;
    const editor = $("code-editor") as HTMLTextAreaElement;
    editor.value = classSourceForCurrent();
    state.editing = true;
    syncEditButtons();
    editor.focus();
  }
  function exitEditMode(): void {
    state.editing = false;
    syncEditButtons();
  }
  function runEditedCode(): void {
    const s = state.currentStrategy;
    if (!s) return;
    const editor = $("code-editor") as HTMLTextAreaElement;
    const src = editor.value;
    if (!src.trim()) {
      toast("editor is empty — nothing to run", true);
      return;
    }
    state.editedSourceByStrategy[s.id] = src;
    state.editing = false;
    renderLiveCode();
    scheduleRun();
  }
  function revertEdit(): void {
    const s = state.currentStrategy;
    if (!s) return;
    delete state.editedSourceByStrategy[s.id];
    state.editing = false;
    renderLiveCode();
    scheduleRun();
  }
  ($("code-edit-toggle") as HTMLButtonElement).addEventListener("click", () => {
    if (state.editing) exitEditMode();
    else enterEditMode();
  });
  ($("code-edit-run") as HTMLButtonElement).addEventListener("click", runEditedCode);
  ($("code-edit-revert") as HTMLButtonElement).addEventListener("click", revertEdit);
  ($("code-editor") as HTMLTextAreaElement).addEventListener("keydown", (ev) => {
    if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") {
      ev.preventDefault();
      runEditedCode();
      return;
    }
    if (ev.key === "Escape") {
      ev.preventDefault();
      exitEditMode();
      return;
    }
    // Tab inserts indent rather than moving focus.
    if (ev.key === "Tab") {
      ev.preventDefault();
      const ta = ev.currentTarget as HTMLTextAreaElement;
      const s = ta.selectionStart;
      const e = ta.selectionEnd;
      ta.value = ta.value.slice(0, s) + "    " + ta.value.slice(e);
      ta.selectionStart = ta.selectionEnd = s + 4;
    }
  });

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
      assertValidDateRange(p.start, p.end);
      // Send tickers:null only when NO custom filter is active. An empty
      // array with hasCustomFilter=true means "user explicitly narrowed
      // to nothing" and should bubble up as a backend error, not silently
      // flip back to the full universe.
      const tickers = state.hasCustomFilter ? p.tickers : null;
      const sid = state.currentStrategy!.id;
      const editedSrc = state.editedSourceByStrategy[sid];
      const body: Record<string, unknown> = {
        strategy_id: sid,
        strategy_params: p.strategyParams,
        engine_params: p.engineParams,
        engine_overrides: p.engineOverrides,
        tickers,
        start: p.start,
        end: p.end,
        data_source: state.dataSource,
      };
      if (editedSrc != null) body.strategy_source_override = editedSrc;
      const t0 = performance.now();
      const data = await fetchJson<SimulationResult>("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }, 1);
      const elapsed = performance.now() - t0;
      lastRender.simPayload = body as unknown as SimRequest;
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
    // Secondary row — Sortino, Calmar, Info Ratio, Hit Rate
    setCard(
      "m-sortino",
      "m-sortino-sub",
      fmtNum(m.sortino),
      (m.sortino ?? 0) >= 1 ? "good" : ((m.sortino as number) < 0.3 ? "bad" : ""),
      `downside-only Sharpe`,
    );
    setCard(
      "m-calmar",
      "m-calmar-sub",
      fmtNum(m.calmar),
      (m.calmar ?? 0) >= 0.5 ? "good" : ((m.calmar as number) < 0 ? "bad" : ""),
      `ann ret / |max DD|`,
    );
    setCard(
      "m-ir",
      "m-ir-sub",
      fmtNum(m.info_ratio),
      (m.info_ratio ?? 0) >= 0.5 ? "good" : ((m.info_ratio as number) < 0 ? "bad" : ""),
      `vs benchmark`,
    );
    setCard(
      "m-hit",
      "m-hit-sub",
      fmtPct(m.hit_rate, 1),
      (m.hit_rate ?? 0) >= 0.52 ? "good" : ((m.hit_rate as number) < 0.48 ? "bad" : ""),
      `pos days / total`,
    );

    lastRender.sim = data;
    const eqPal = chartPalette();
    const eqLayout = plotLayout();
    const liveLabel = state.currentStrategy
      ? autoRunLabel(state.currentStrategy, lastRender.simPayload)
      : "Strategy";
    // Two-pane layout: equity on top (y), drawdown below (y2).
    // Plotly handles the shared x-axis automatically when traces share xaxis: "x".
    const traces: Record<string, unknown>[] = [
      {
        x: data.dates,
        y: data.cumulative_benchmark.map((v) => v * 100),
        name: "Equal-weight universe",
        type: "scatter",
        mode: "lines",
        xaxis: "x",
        yaxis: "y",
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
        xaxis: "x",
        yaxis: "y",
        line: { color, width: 1.6 },
        opacity: 0.85,
        hovertemplate: `%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>${escapeHtml(pinned.label)}</extra>`,
      });
      // Pinned drawdown trace on the bottom pane, dimmer matching color.
      if (pinned.cumulativeDrawdown && pinned.cumulativeDrawdown.length) {
        traces.push({
          x: pinned.dates,
          y: pinned.cumulativeDrawdown,
          name: `${pinned.label} · DD`,
          type: "scatter",
          mode: "lines",
          xaxis: "x",
          yaxis: "y2",
          line: { color, width: 1.0 },
          opacity: 0.5,
          showlegend: false,
          hovertemplate: `%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>${escapeHtml(pinned.label)} DD</extra>`,
        });
      }
    }
    traces.push({
      x: data.dates,
      y: data.cumulative_net.map((v) => v * 100),
      name: `${liveLabel} · live`,
      type: "scatter",
      mode: "lines",
      xaxis: "x",
      yaxis: "y",
      line: { color: eqPal.q, width: 2.2 },
      hovertemplate: `%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>${escapeHtml(liveLabel)}</extra>`,
    });
    // Live drawdown — filled to zero in loss-red.
    const liveDD = (data.cumulative_drawdown || []).map((v) => v * 100);
    if (liveDD.length) {
      traces.push({
        x: data.dates,
        y: liveDD,
        name: "Drawdown",
        type: "scatter",
        mode: "lines",
        xaxis: "x",
        yaxis: "y2",
        line: { color: cssVar("--loss"), width: 1.4 },
        fill: "tozeroy",
        fillcolor: "rgba(224,112,80,0.18)",
        showlegend: false,
        hovertemplate: "%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>Drawdown</extra>",
      });
    }
    Plotly.react(
      "chart-equity",
      traces,
      {
        ...eqLayout,
        // Top pane: equity curve, ~70% of vertical space.
        yaxis: {
          ...(eqLayout.yaxis as Record<string, unknown>),
          title: "Cumulative return (%)",
          tickformat: ",.0f",
          ticksuffix: "%",
          automargin: true,
          domain: [0.32, 1.0],
        },
        // Bottom pane: drawdown, ~28%, tied to the same x-axis.
        yaxis2: {
          ...(eqLayout.yaxis as Record<string, unknown>),
          title: "Drawdown (%)",
          tickformat: ",.0f",
          ticksuffix: "%",
          automargin: true,
          domain: [0, 0.24],
          rangemode: "tozero",
          // Drawdown is always ≤0; flip the autorange so 0 is at top, deepest at bottom.
          autorange: "reversed",
          zeroline: true,
          zerolinecolor: cssVar("--border-strong"),
          zerolinewidth: 1,
        },
        xaxis: {
          ...(eqLayout.xaxis as Record<string, unknown>),
          automargin: true,
          anchor: "y2",
        },
      },
      plotConfig,
    );

    renderMonthlyHeatmap(data.monthly_returns || []);
    renderAudit(data.survivorship_audit || {});
    renderPinnedList();
    if (data.universe && typeof data.universe.n_tickers === "number") {
      state.universeSize = data.universe.n_tickers;
      state.universeMissingCount = data.universe.missing_from_cache?.length || 0;
      if (data.universe.custom) {
        state.hasCustomFilter = true;
        state.customTickers = (data.universe.selected || []).slice();
      } else {
        state.hasCustomFilter = false;
        state.customTickers = [];
      }
      refreshUniverseStatus();
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
      p.data_source || "yfinance",
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
      cumulativeDrawdown: (lastRender.sim.cumulative_drawdown || []).map((v) => v * 100),
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

  function renderMonthlyHeatmap(monthly: MonthlyReturn[]): void {
    const host = document.getElementById("chart-monthly");
    if (!host) return;
    if (!monthly || monthly.length === 0) {
      Plotly.purge("chart-monthly");
      return;
    }
    // Pivot the long-form payload into a year × month grid (most recent year on top).
    const years = Array.from(new Set(monthly.map((m) => m.year))).sort((a, b) => a - b);
    const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    const grid: (number | null)[][] = years.map(() => Array(12).fill(null));
    for (const cell of monthly) {
      const yi = years.indexOf(cell.year);
      if (yi >= 0) grid[yi][cell.month - 1] = cell.ret;
    }
    // Cell text: "+3.4%", "-1.8%", or "" for missing.
    const text: string[][] = grid.map((row) =>
      row.map((v) =>
        v == null ? "" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`,
      ),
    );
    // Custom red→black→green colorscale anchored to 0; aligns with the
    // amber-on-warm-black palette without using cyan/blue elsewhere on screen.
    const palette = chartPalette();
    const layout = plotLayout();
    Plotly.react(
      "chart-monthly",
      [
        {
          z: grid.map((row) => row.map((v) => (v == null ? null : v * 100))),
          x: months,
          y: years.map((y) => String(y)),
          type: "heatmap",
          colorscale: [
            [0,    "#7a1f1f"],
            [0.25, "#3a1010"],
            [0.5,  palette.bg],
            [0.75, "#1f4a25"],
            [1,    "#3a8a4a"],
          ],
          zmid: 0,
          hovertemplate: "%{y} %{x}: %{z:.1f}%<extra></extra>",
          showscale: false,
          xgap: 2,
          ygap: 2,
        },
        {
          // Cell-text overlay using a transparent scattergl trick:
          // Plotly's heatmap doesn't support `text=` natively for cells,
          // so we render annotations via layout.annotations below instead.
          type: "scatter",
          x: [],
          y: [],
          mode: "markers",
          showlegend: false,
        },
      ],
      {
        ...layout,
        margin: { t: 8, r: 8, b: 28, l: 60 },
        height: Math.max(160, years.length * 28 + 60),
        xaxis: {
          ...(layout.xaxis as Record<string, unknown>),
          side: "top",
          tickfont: { size: 11, color: palette.inkSoft },
          fixedrange: true,
        },
        yaxis: {
          ...(layout.yaxis as Record<string, unknown>),
          autorange: "reversed",
          tickfont: { size: 11, color: palette.inkSoft },
          fixedrange: true,
          automargin: true,
        },
        annotations: years.flatMap((y, yi) =>
          months.map((mo, mi) => {
            const v = grid[yi][mi];
            if (v == null) return null;
            const txt = `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;
            return {
              x: mo,
              y: String(y),
              text: txt,
              showarrow: false,
              font: {
                family: "IBM Plex Mono, JetBrains Mono, monospace",
                size: 10,
                color:
                  Math.abs(v) < 0.005
                    ? palette.inkSoft
                    : v > 0
                      ? "#bdf3c0"
                      : "#f5c8be",
              },
            };
          }).filter((a) => a !== null),
        ).filter((a) => a !== null),
      },
      { displayModeBar: false, responsive: true },
    );
    void text;
  }

  function renderAudit(a: SurvivorshipAudit): void {
    const sub = $("audit-sub");
    const statsEl = $("audit-stats");
    if (!a || !a.universe_size) {
      sub.textContent = "no audit available";
      statsEl.innerHTML = "";
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
    state.dataOverview = await fetchJson<DataOverview>(`/api/data?${sourceQuery()}`);
    $("data-sub").textContent = `${state.dataOverview.n_tickers} tickers · ${state.dataOverview.source_file}`;
    renderTickerTable();
  }
  function renderTickerTable(): void {
    const body = $("ticker-tbody");
    body.innerHTML = "";
    if (!state.dataOverview) return;
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
      const d = await fetchJson<TickerDetail>(
        `/api/data/ticker/${encodeURIComponent(tic)}?${sourceQuery()}`,
      );
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
      const missing = state.universeMissingCount
        ? ` · ${state.universeMissingCount} ignored`
        : "";
      status.textContent = `custom: ${n} ticker${n === 1 ? "" : "s"}${missing}`;
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
      state.universeMissingCount = 0;
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
      state.universeMissingCount = 0;
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
      state.universeMissingCount = missing;
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
  async function waitReady(targetSource?: string, maxWaitMs = 0): Promise<boolean> {
    const started = performance.now();
    let misses = 0;
    while (true) {
      if (maxWaitMs && performance.now() - started > maxWaitMs) {
        const label = targetSource ? sourceLabel(targetSource) : "backend";
        setStatus(`error: timed out loading ${label}`, "error");
        return false;
      }
      try {
        const url = targetSource ? `/api/status?${sourceQuery(targetSource)}` : "/api/status";
        const s = await fetchJson<StatusResponse>(url, undefined, 0);
        misses = 0;
        const sourceMatches = !targetSource || s.data_source === targetSource;
        if (s.error && sourceMatches) {
          setStatus(`error: ${s.error}`, "error");
          return false;
        }
        if (s.data_source && sourceMatches) state.dataSource = s.data_source;
        if (s.available_sources) state.availableSources = s.available_sources;
        if (s.ready && sourceMatches) {
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
        if (s.ready && targetSource) {
          setStatus(`loading · waiting for ${sourceLabel(targetSource)}`, "loading");
        } else {
          setStatus(`loading · ${s.message || "dataset"}`, "loading");
        }
      } catch (exc) {
        misses += 1;
        setStatus(`backend unreachable · retrying (${misses})`, misses > 5 ? "error" : "loading");
      }
      await sleep(Math.min(1000 + misses * 250, 2500));
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
  function setSourceSelectorBusy(busy: boolean): void {
    const sel = document.getElementById("source-select") as HTMLSelectElement | null;
    if (sel) sel.disabled = busy;
  }
  function showSourceLoadingState(target: string): void {
    const label = sourceLabel(target);
    $("universe-label").textContent = `universe: loading ${label}`;
    $("universe-status").textContent = "loading…";
    $("strategy-sizing").textContent = "loading universe…";
    $("data-sub").textContent = `loading ${label} data…`;
    for (const [valueId, subId] of [
      ["m-sharpe", "m-sharpe-sub"],
      ["m-return", "m-return-sub"],
      ["m-dd", "m-dd-sub"],
      ["m-tcost", "m-tcost-sub"],
      ["m-sortino", "m-sortino-sub"],
      ["m-calmar", "m-calmar-sub"],
      ["m-ir", "m-ir-sub"],
      ["m-hit", "m-hit-sub"],
    ]) {
      const valueEl = $(valueId);
      valueEl.textContent = "—";
      valueEl.className = "m-value";
      $(subId).textContent = "";
    }
    $("audit-sub").textContent = `loading ${label} universe`;
    $("audit-stats").innerHTML = "";
    Plotly.purge("chart-equity");
    Plotly.purge("chart-monthly");
    Plotly.purge("chart-audit");
    state.dataOverview = null;
    state.selectedTicker = null;
    renderTickerTable();
    const right = document.getElementById("data-right");
    if (right) {
      right.innerHTML = `<div class="data-empty">Loading ${escapeHtml(label)} data…</div>`;
    }
  }
  async function switchDataSource(target: string): Promise<void> {
    if (target === state.dataSource) return;
    const previousSource = state.dataSource;
    state.dataSource = target;
    updateSourceSelector();
    updateUniverseEditorMode();
    setSourceSelectorBusy(true);
    showSourceLoadingState(target);
    setStatus(`switching to ${sourceLabel(target)}…`, "loading");
    setProgress(15);
    try {
      await fetchJson<StatusResponse>("/api/warmup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: target }),
      }, 1);
      // Now wait for a worker that is actually ready on the requested
      // source. Production can have more than one backend worker, so a
      // generic "ready" status is not enough after a source change.
      const ok = await waitReady(target, 120000);
      if (!ok) {
        state.dataSource = previousSource;
        updateSourceSelector();
        updateUniverseEditorMode();
        return;
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
      state.dataSource = previousSource;
      updateSourceSelector();
      updateUniverseEditorMode();
      toast(`source switch failed: ${msg}`, true, 6000);
      setStatus(`error: ${msg}`, "error");
    } finally {
      setSourceSelectorBusy(false);
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
    try {
      if (!(await waitReady())) return;
      state.catalog = await loadCatalog();
      renderStrategyTabs();
      selectStrategy(state.catalog.strategies[0].id);
      loadDataOverview().catch((e) => console.error(e));
    } catch (exc) {
      const msg = exc instanceof Error ? exc.message : String(exc);
      toast(`startup failed: ${msg}`, true, 6000);
      setStatus(`error: ${msg}`, "error");
    }
  })();
})();
