// Entry point: loads runs/batches, renders the shell, and routes state changes to the views.

import { api } from "./api.js";
import { attentionLayers, renderAttention } from "./attention.js";
import { embeddingTables, renderEmbeddings, representationLayers } from "./embeddings.js";
import { camLayers, renderGradcam } from "./gradcam.js";
import { renderInspector } from "./inspector.js";
import { renderLayers } from "./layers.js";
import { renderLoss } from "./loss.js";
import { renderOverview } from "./overview.js";
import { renderReplay, replayKey, replayPosition, stopReplay } from "./replay.js";
import { renderScene3d, scene3dKey, scene3dPosition, stopScene3d } from "./scene3d.js";
import { renderSamples } from "./samples.js";
import { renderSource } from "./source.js";
import { renderStep } from "./step.js";
import { loadHidden, saveHidden, store } from "./store.js";
import { renderTimeline } from "./timeline.js";
import { renderKindFilters, renderTree } from "./tree.js";
import { clear, debounce, fmtMs, h, icon, phaseColor } from "./util.js";
import { renderWeights } from "./weights.js";

// Pages follow the order of a training step: watch it (Replay), what the model sees on the way
// forward (Insights), how wrong it was (Loss), what the update did (Weights) — then the raw call
// trace underneath everything.
const PAGES = {
  replay: { label: "Replay", what: "Watch training happen: data → forward → loss → backward → update" },
  insights: { label: "Insights", what: "What the model attends to and how it represents samples", available: (s) => availableInsights(s).length > 0 },
  loss: { label: "Loss", what: "What the model predicted and how wrong it was, per sample", render: renderLoss },
  weights: { label: "Weights", what: "Each parameter, its gradient and the update the optimizer applied", render: renderWeights },
  explore: { label: "Trace", what: "Every recorded call with its timing, inputs, outputs and source line" },
};
const REPLAY_VIEWS = {
  "2d": { label: "Layer stack", render: renderReplay },
  "3d": { label: "3D", render: renderScene3d },
};
const INSIGHTS = {
  attention: { label: "Attention", what: "Where each token looks: rows are queries, columns are keys", render: renderAttention, available: (s) => s.allBatches?.some((b) => attentionLayers(b).length) },
  gradcam: { label: "Grad-CAM", what: "Which regions of each image drove the class score", render: renderGradcam, available: (s) => s.allBatches?.some((b) => camLayers(b).length) },
  embeddings: {
    label: "Embeddings",
    what: "2-D projections: points that sit together are represented alike",
    render: renderEmbeddings,
    available: (s) => s.allBatches?.[0] && (representationLayers(s.allBatches[0]).length || embeddingTables(s).length),
  },
};
// old page names (bookmarks, links) → where they live now
const LEGACY_PAGES = { scene3d: { page: "replay", replayView: "3d" }, attention: { page: "insights", insight: "attention" }, gradcam: { page: "insights", insight: "gradcam" }, embeddings: { page: "insights", insight: "embeddings" } };

const availableInsights = (s) => Object.keys(INSIGHTS).filter((k) => INSIGHTS[k].available(s));
const currentInsight = (s) => {
  const available = availableInsights(s);
  return available.includes(s.insight) ? s.insight : available[0];
};
const visiblePages = (s) => Object.keys(PAGES).filter((k) => !PAGES[k].available || PAGES[k].available(s));

const $ = (id) => document.getElementById(id);
const PHASES = ["data", "forward", "loss", "backward", "post_backward", "optimizer"];
const PHASE_LABEL = { post_backward: "after backward" };
const BATCH_TABS = [
  ["overview", "Overview"],
  ["samples", "Samples"],
  ["timeline", "Timeline"],
  ["source", "Source"],
  ["layers", "Layers"],
];
const STEP_TABS = [
  ["step", "Step"],
  ["overview", "Overview"],
];

// ------------------------------------------------------------------------------------------
// data loading
// ------------------------------------------------------------------------------------------
async function boot() {
  initTheme();
  bindStatic();
  try {
    const runs = await api("runs");
    store.set({ runs });
    if (!runs.length) {
      clear($("center-body"), h("div", { class: "empty" }, "No runs found here. Record one with ", h("span", { class: "mono" }, "tvis run train.py"), "."));
      return;
    }
    const wanted = readHash();
    const run = runs.find((r) => r.run_id === wanted.run) ? wanted.run : runs[0].run_id;
    await selectRun(run, wanted);
  } catch (err) {
    showError(err);
  }
}

async function selectRun(runId, initial = {}) {
  const { meta, steps } = await api("run", { run: runId });
  const hidden = loadHidden(meta.project_dir);
  const mode = initial.mode === "step" ? "step" : "batch";
  store.set({
    runId,
    meta,
    steps,
    hidden,
    page: PAGES[initial.page] ? initial.page : store.get().page || "replay",
    replayView: REPLAY_VIEWS[initial.view] ? initial.view : store.get().replayView,
    insight: INSIGHTS[initial.insight] ? initial.insight : store.get().insight,
    mode,
    batchIndex: clampIndex(initial.b, meta.batches_captured),
    stepIndex: clampIndex(initial.s, steps.length),
    selection: initial.call != null ? { call: initial.call } : null,
    sample: initial.sample ?? null,
    tab: initial.tab || (mode === "step" ? "step" : "overview"),
    batch: null,
    stepBatches: [],
    allBatches: null,
    expanded: new Map(),
  });
  await Promise.all([loadBatch(), loadStep(), loadAllBatches()]);
}

const clampIndex = (value, count) => (Number.isInteger(value) && value >= 0 && value < count ? value : 0);

async function loadBatch() {
  const { runId, batchIndex, meta } = store.get();
  if (!meta?.batches_captured) return;
  const batch = await api("batch", { run: runId, batch: batchIndex });
  if (store.get().runId === runId && store.get().batchIndex === batchIndex) store.set({ batch });
}

async function loadStep() {
  const { runId, steps, stepIndex } = store.get();
  const step = steps[stepIndex];
  if (!step) return;
  const stepBatches = await Promise.all(step.batches.map((b) => api("batch", { run: runId, batch: b })));
  if (store.get().runId === runId && store.get().stepIndex === stepIndex) store.set({ stepBatches });
}

async function loadAllBatches() {
  const { runId, meta } = store.get();
  const docs = await Promise.all(Array.from({ length: meta.batches_captured }, (_, i) => api("batch", { run: runId, batch: i })));
  if (store.get().runId === runId) store.set({ allBatches: docs });
}

// ------------------------------------------------------------------------------------------
// rendering
// ------------------------------------------------------------------------------------------
function renderRunSelect() {
  const s = store.get();
  clear(
    $("run-select"),
    s.runs.map((r) => h("option", { value: r.run_id, selected: r.run_id === s.runId }, runLabel(r))),
  );
  renderNotices();
}

function runLabel(run) {
  const id = run.run_id;
  const when = /^\d{8}-\d{6}/.test(id) ? `${id.slice(4, 6)}/${id.slice(6, 8)} ${id.slice(9, 11)}:${id.slice(11, 13)}` : id;
  return `${run.script?.split("/").pop() || id}  ·  ${when}${run.status === "complete" ? "" : `  ·  ${run.status}`}`;
}

function renderNotices() {
  const s = store.get();
  const notices = s.meta?.notices?.length || 0;
  const failed = s.meta?.status === "error";
  const button = $("notices-btn");
  button.classList.toggle("hidden", !notices && !failed);
  button.title = failed ? "The script failed: see the run overview" : `${notices} capture notice${notices === 1 ? "" : "s"}: see the run overview`;
  clear(button, icon("alert"), failed ? "failed" : String(notices));
}

function renderHeader() {
  const s = store.get();
  for (const button of $("mode-seg").querySelectorAll("button")) button.classList.toggle("on", button.dataset.mode === s.mode);
  const count = s.mode === "batch" ? s.meta?.batches_captured || 0 : s.steps.length;
  const current = s.mode === "batch" ? s.batchIndex : s.stepIndex;
  clear(
    $("index-chips"),
    Array.from({ length: count }, (_, i) =>
      h("button", { class: `chip ${i === current ? "on" : ""}`, onclick: () => goTo(i), title: s.mode === "batch" ? `batch ${i}` : `step ${i}` }, s.mode === "batch" ? `b${i}` : `s${i}`),
    ),
  );
  const sample = s.sample != null ? s.batch?.samples?.[s.sample] : null;
  clear(
    $("follow-slot"),
    s.sample != null
      ? h("span", { class: "follow" }, `following sample #${s.sample}${sample?.index != null ? ` (dataset index ${sample.index})` : ""}`, h("button", { title: "Stop following (Esc)", onclick: () => store.set({ sample: null }) }, icon("close")))
      : null,
  );
  renderNotices();
}

function renderNav() {
  const s = store.get();
  const pages = visiblePages(s);
  clear(
    $("nav"),
    pages.flatMap((key, i) => [
      key === "explore" ? h("span", { class: "gap" }) : null,
      h("button", { class: s.page === key ? "on" : "", title: `${PAGES[key].what}  (${i + 1})`, "aria-current": s.page === key ? "page" : null, onclick: () => openPage(key) }, PAGES[key].label),
    ]),
  );
}

function openPage(key) {
  if (key !== store.get().page) store.set({ page: key });
}

/** Second level: the views of the current page, what the page answers, and its keys. */
function renderSubnav() {
  const s = store.get();
  const node = $("subnav");
  if (s.page === "explore") return clear(node);
  const views = (entries, current, pick) =>
    h(
      "div",
      { class: "views", role: "tablist" },
      entries.map(([key, label]) => h("button", { class: key === current ? "on" : "", role: "tab", "aria-selected": key === current, onclick: () => key !== current && pick(key) }, label)),
    );
  const keys = (...pairs) => h("div", { class: "keys" }, pairs.map(([k, what]) => h("span", {}, k.split(" ").map((x) => h("kbd", {}, x)), what)));
  if (s.page === "replay") {
    return clear(
      node,
      views(Object.entries(REPLAY_VIEWS).map(([k, v]) => [k, v.label]), s.replayView, switchReplayView),
      keys(["space", "play"], ["← →", "step"], ["[ ]", "sample"]),
    );
  }
  if (s.page === "insights") {
    const current = currentInsight(s);
    return clear(
      node,
      views(availableInsights(s).map((k) => [k, INSIGHTS[k].label]), current, (k) => store.set({ insight: k })),
      current ? h("span", { class: "what" }, INSIGHTS[current].what) : null,
    );
  }
  clear(node, h("span", { class: "what", style: { marginLeft: 0 } }, PAGES[s.page]?.what));
}

/** Switch 2D ⇄ 3D at the same moment of the recording. */
function switchReplayView(view) {
  const event = store.get().replayView === "3d" ? scene3dPosition() : replayPosition();
  store.set({ replayView: view, replayFocus: event ? { batch: event.batch, kind: event.kind, call: event.call?.id } : null });
}

function renderPage() {
  const s = store.get();
  const explore = s.page === "explore";
  $("explore").classList.toggle("hidden", !explore);
  $("page").classList.toggle("hidden", explore);
  if (s.page !== "replay" || s.replayView !== "2d") stopReplay();
  if (s.page !== "replay" || s.replayView !== "3d") stopScene3d();
  if (explore) {
    clear($("page"));
    for (const view of ["header", "phasebar", "left", "tabs", "center", "inspector"]) RENDER[view]();
    return;
  }
  const page = PAGES[s.page] ? s.page : "replay";
  if (page === "replay") return (REPLAY_VIEWS[s.replayView] || REPLAY_VIEWS["2d"]).render($("page"));
  if (page === "insights") {
    const insight = currentInsight(s);
    return insight ? INSIGHTS[insight].render($("page")) : clear($("page"), h("div", { class: "loading" }, "loading…"));
  }
  PAGES[page].render($("page"));
}

function renderPhasebar() {
  const s = store.get();
  const batches = s.mode === "batch" ? (s.batch ? [s.batch] : []) : s.stepBatches;
  const phases = {};
  let total = 0;
  for (const b of batches) {
    for (const [p, ms] of Object.entries(b.timing?.phases_ms || {})) phases[p] = (phases[p] || 0) + ms;
    total += b.timing?.total_ms || 0;
  }
  if (!total) return clear($("phasebar"), h("span", { class: "faint" }, "—"));
  clear(
    $("phasebar"),
    h("div", { class: "bar" }, PHASES.map((p) => (phases[p] ? h("span", { style: { width: `${(phases[p] / total) * 100}%`, background: phaseColor(p) }, title: `${p} ${fmtMs(phases[p])}` }) : null))),
    h("div", { class: "legend" }, PHASES.filter((p) => phases[p]).map((p) => h("span", {}, h("i", { class: "dot", style: { background: phaseColor(p) } }), `${PHASE_LABEL[p] || p} `, h("b", {}, fmtMs(phases[p]))))),
    h("span", { class: "total", title: `clock: ${s.meta?.clock}; capture overhead removed` }, `${fmtMs(total)} ${s.mode === "step" ? "step" : "batch"}`),
  );
}

function renderTabs() {
  const s = store.get();
  const tabs = s.mode === "step" ? STEP_TABS : BATCH_TABS;
  if (!tabs.some(([key]) => key === s.tab)) {
    store.set({ tab: tabs[0][0] });
    return;
  }
  clear($("tabs"), tabs.map(([key, label]) => h("button", { class: key === s.tab ? "on" : "", onclick: () => store.set({ tab: key }) }, label)));
}

function renderCenter() {
  const s = store.get();
  const body = $("center-body");
  const views = { overview: renderOverview, samples: renderSamples, timeline: renderTimeline, source: renderSource, layers: renderLayers, step: renderStep };
  (views[s.tab] || renderOverview)(body);
}

function renderLeft() {
  const s = store.get();
  $("left-title").textContent = s.mode === "step" ? "Step" : "Calls";
  $("kind-filters").classList.toggle("hidden", s.mode === "step");
  $("tree-search").classList.toggle("hidden", s.mode === "step");
  renderKindFilters($("kind-filters"));
  renderTree($("tree"));
  $("hidden-btn").title = `Hidden (${s.hidden.size})`;
  $("hidden-btn").classList.toggle("hidden", s.mode === "step");
}

// which views depend on which state keys (explore views only render while Explore is open)
const EXPLORE_VIEWS = new Set(["header", "phasebar", "left", "tabs", "center", "inspector"]);
const DEPENDS = {
  nav: ["page", "allBatches", "steps", "meta", "runs"],
  subnav: ["page", "replayView", "insight", "allBatches", "meta"],
  runselect: ["runs", "runId", "meta"],
  page: ["page", "replayView", "insight", "allBatches", "meta", "steps", "sample", "batchIndex", "runId"],
  header: ["runs", "runId", "mode", "batchIndex", "stepIndex", "steps", "meta", "sample", "batch"],
  phasebar: ["batch", "mode", "stepBatches"],
  left: ["batch", "mode", "filter", "kinds", "hidden", "expanded", "selection", "stepBatches", "steps", "stepIndex"],
  tabs: ["tab", "mode"],
  center: ["tab", "batch", "mode", "selection", "sample", "meta", "stepBatches", "allBatches", "sourceFile", "sourceLine", "hidden"],
  inspector: ["selection", "batch", "sample", "stepBatches", "mode"],
};
const RENDER = {
  nav: renderNav,
  subnav: renderSubnav,
  runselect: renderRunSelect,
  page: () => store.get().page !== "explore" && renderPage(),
  header: renderHeader,
  phasebar: renderPhasebar,
  left: renderLeft,
  tabs: renderTabs,
  center: renderCenter,
  inspector: () => renderInspector($("insp-title"), $("inspector")),
};

store.subscribe((state, patch) => {
  const keys = Object.keys(patch);
  if (keys.includes("page")) renderPage();
  for (const [view, deps] of Object.entries(DEPENDS)) {
    if (view === "page" && keys.includes("page")) continue;
    if (EXPLORE_VIEWS.has(view) && state.page !== "explore") continue;
    if (keys.some((k) => deps.includes(k))) {
      try {
        RENDER[view]();
      } catch (err) {
        console.error(view, err);
      }
    }
  }
  if (["batchIndex", "runId"].some((k) => keys.includes(k)) && !keys.includes("batch")) {
    store.set({ batch: null, selection: keys.includes("selection") ? state.selection : null });
    loadBatch().catch(showError);
  }
  if (["stepIndex"].some((k) => keys.includes(k)) || (keys.includes("mode") && state.mode === "step")) loadStep().catch(showError);
  writeHash();
});

// ------------------------------------------------------------------------------------------
// interaction
// ------------------------------------------------------------------------------------------
function goTo(i) {
  const s = store.get();
  if (s.mode === "batch") store.set({ batchIndex: i, sample: null });
  else store.set({ stepIndex: i, selection: null });
}

function bindStatic() {
  $("run-select").addEventListener("change", (e) => selectRun(e.target.value).catch(showError));
  for (const button of $("mode-seg").querySelectorAll("button")) {
    button.addEventListener("click", () => {
      const s = store.get();
      const mode = button.dataset.mode;
      if (mode === s.mode) return;
      const step = s.batch?.step ?? s.stepIndex;
      store.set(mode === "step" ? { mode, stepIndex: step, tab: "step", selection: null } : { mode, tab: "overview", selection: null });
    });
  }
  $("tree-search").addEventListener("input", debounce((e) => store.set({ filter: e.target.value.trim() }), 120));
  $("theme-btn").append(icon("theme"));
  $("theme-btn").addEventListener("click", toggleTheme);
  $("help-btn").append(icon("keys"));
  $("help-btn").addEventListener("click", showShortcuts);
  $("notices-btn").addEventListener("click", () => store.set({ page: "explore", mode: "batch", tab: "overview" }));
  $("hidden-btn").append(icon("hide"));
  $("hidden-btn").addEventListener("click", showHiddenDialog);
  document.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.metaKey || e.ctrlKey || e.altKey) return;
    if (document.querySelector(".modal-back")) {
      if (e.key === "Escape") document.querySelector(".modal-back").remove();
      return;
    }
    const s = store.get();
    const pages = visiblePages(s);
    if (/^[1-9]$/.test(e.key) && pages[Number(e.key) - 1]) return openPage(pages[Number(e.key) - 1]);
    if (e.key === "?") return showShortcuts();
    if (s.page === "replay" && s.replayView === "3d" && scene3dKey(e)) return;
    if (s.page === "replay" && s.replayView !== "3d" && replayKey(e)) return;
    if (s.page !== "explore") return;
    const count = s.mode === "batch" ? s.meta?.batches_captured || 0 : s.steps.length;
    const current = s.mode === "batch" ? s.batchIndex : s.stepIndex;
    if (e.key === "j" && current + 1 < count) goTo(current + 1);
    else if (e.key === "k" && current > 0) goTo(current - 1);
    else if (e.key === "Escape") store.set({ sample: null });
    else if (e.key === "/") {
      e.preventDefault();
      $("tree-search").focus();
    }
  });
}

function showHiddenDialog() {
  const s = store.get();
  const close = () => back.remove();
  const back = h("div", { class: "modal-back", onclick: (e) => e.target === back && close() });
  const list = [...s.hidden].sort().map((key) =>
    h(
      "div",
      { class: "row", style: { gridTemplateColumns: "1fr auto", paddingLeft: "4px" } },
      h("span", { class: "mono" }, key),
      h(
        "button",
        {
          class: "chip small",
          onclick: () => {
            const hidden = new Set(store.get().hidden);
            hidden.delete(key);
            saveHidden(s.meta?.project_dir, hidden);
            store.set({ hidden });
            close();
            showHiddenDialog();
          },
        },
        "show",
      ),
    ),
  );
  back.append(
    h(
      "div",
      { class: "modal" },
      h("h3", {}, "Hidden calls"),
      list.length ? list : h("div", { class: "faint" }, "Nothing hidden. Hover a row in the call tree and use its hide button to hide calls like it; their children stay visible."),
      h("div", { class: "actions" }, h("button", { class: "chip", onclick: close }, "Close")),
    ),
  );
  document.body.append(back);
}

function showShortcuts() {
  if (document.querySelector(".modal-back")) return;
  const s = store.get();
  const back = h("div", { class: "modal-back", onclick: (e) => e.target === back && back.remove() });
  const row = (keys, what) => [h("dt", {}, keys.split(" ").map((k) => h("kbd", {}, k))), h("dd", {}, what)];
  back.append(
    h(
      "div",
      { class: "modal" },
      h("h3", {}, "Keyboard"),
      h(
        "dl",
        { class: "shortcuts" },
        h("h4", {}, "Anywhere"),
        visiblePages(s).map((key, i) => row(String(i + 1), PAGES[key].label)),
        row("?", "this list"),
        h("h4", {}, "Replay"),
        row("space", "play / pause"),
        row("← →", "previous / next stop"),
        row("shift ← →", "previous / next phase"),
        row("[ ]", "previous / next sample"),
        h("h4", {}, "Trace"),
        row("j k", "next / previous batch or step"),
        row("/", "filter the call tree"),
        row("esc", "stop following a sample"),
      ),
      h("div", { class: "actions" }, h("button", { class: "chip", onclick: () => back.remove() }, "Close")),
    ),
  );
  document.body.append(back);
}

function showError(err) {
  console.error(err);
  clear($("center-body"), h("div", { class: "empty" }, h("div", { class: "err" }, String(err.message || err))));
}

// ------------------------------------------------------------------------------------------
// theme + URL state
// ------------------------------------------------------------------------------------------
function initTheme() {
  try {
    const theme = localStorage.getItem("tvis.theme");
    if (theme) document.documentElement.dataset.theme = theme;
  } catch {
    /* no storage: follow the system preference */
  }
}

function toggleTheme() {
  const current = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  const next = current === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem("tvis.theme", next);
  } catch {
    /* ignore */
  }
  store.set({ batch: store.get().batch }); // redraw canvases with the new palette
}

function readHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  const int = (k) => (params.has(k) ? Number(params.get(k)) : undefined);
  const legacy = LEGACY_PAGES[params.get("page")] || {};
  return {
    run: params.get("run"),
    page: legacy.page || params.get("page"),
    view: legacy.replayView || params.get("view"),
    insight: legacy.insight || params.get("insight"),
    mode: params.get("mode"),
    b: int("b"),
    s: int("s"),
    call: int("call"),
    sample: int("sample"),
    tab: params.get("tab"),
  };
}

const writeHash = debounce(() => {
  const s = store.get();
  if (!s.runId) return;
  const params = new URLSearchParams({ run: s.runId, page: s.page });
  if (s.page === "replay") params.set("view", s.replayView);
  if (s.page === "insights" && currentInsight(s)) params.set("insight", currentInsight(s));
  params.set("mode", s.mode);
  params.set("tab", s.tab);
  if (s.mode === "batch") params.set("b", s.batchIndex);
  else params.set("s", s.stepIndex);
  if (s.selection?.call != null) params.set("call", s.selection.call);
  if (s.sample != null) params.set("sample", s.sample);
  history.replaceState(null, "", `#${params}`);
}, 100);

window.addEventListener("resize", debounce(() => store.get().page === "explore" && store.get().tab === "timeline" && renderCenter(), 150));
boot();
