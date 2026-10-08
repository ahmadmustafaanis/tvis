// Entry point: loads runs/batches, renders the shell, and routes state changes to the views.

import { api } from "./api.js";
import { attentionLayers, renderAttention } from "./attention.js";
import { embeddingTables, renderEmbeddings, representationLayers } from "./embeddings.js";
import { camLayers, renderGradcam } from "./gradcam.js";
import { renderInspector } from "./inspector.js";
import { renderLayers } from "./layers.js";
import { renderLoss } from "./loss.js";
import { renderOverview } from "./overview.js";
import { renderReplay, replayKey, stopReplay } from "./replay.js";
import { renderSamples } from "./samples.js";
import { renderSource } from "./source.js";
import { renderStep } from "./step.js";
import { loadHidden, saveHidden, store } from "./store.js";
import { renderTimeline } from "./timeline.js";
import { renderKindFilters, renderTree } from "./tree.js";
import { clear, debounce, fmtMs, h, phaseColor } from "./util.js";
import { renderWeights } from "./weights.js";

const PAGES = {
  replay: { label: "▶ Replay", render: renderReplay },
  loss: { label: "Loss", render: renderLoss },
  weights: { label: "Weights", render: renderWeights },
  attention: { label: "Attention", render: renderAttention, insight: true, available: (s) => s.allBatches?.some((b) => attentionLayers(b).length) },
  gradcam: { label: "Grad-CAM", render: renderGradcam, insight: true, available: (s) => s.allBatches?.some((b) => camLayers(b).length) },
  embeddings: {
    label: "Embeddings",
    render: renderEmbeddings,
    insight: true,
    available: (s) => s.allBatches?.[0] && (representationLayers(s.allBatches[0]).length || embeddingTables(s).length),
  },
  explore: { label: "Explore" },
};

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
    s.runs.map((r) => h("option", { value: r.run_id, selected: r.run_id === s.runId }, `${r.script?.split("/").pop() || r.run_id} · ${r.run_id.slice(0, 15)} · ${r.status}`)),
  );
  const notices = s.meta?.notices?.length || 0;
  $("notices-btn").classList.toggle("hidden", !notices && s.meta?.status !== "error");
  $("notices-btn").title = `${notices} notice(s)`;
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
      ? h("span", { class: "follow" }, `following sample #${s.sample}${sample?.index != null ? ` (idx ${sample.index})` : ""}`, h("button", { title: "Stop following (Esc)", onclick: () => store.set({ sample: null }) }, "✕"))
      : null,
  );
  const notices = s.meta?.notices?.length || 0;
  $("notices-btn").classList.toggle("hidden", !notices && s.meta?.status !== "error");
  $("notices-btn").title = `${notices} notice(s)`;
}

function renderNav() {
  const s = store.get();
  const items = [];
  let insights = false;
  for (const [key, page] of Object.entries(PAGES)) {
    if (page.available && !page.available(s)) continue;
    if (page.insight && !insights) {
      items.push(h("span", { class: "sep" }), h("span", { class: "faint", style: { fontSize: "11px", marginRight: "2px" } }, "Insights"));
      insights = true;
    }
    if (key === "explore") items.push(h("span", { class: "sep" }));
    items.push(h("button", { class: `${s.page === key ? "on" : ""} ${page.insight ? "sub" : ""}`, onclick: () => store.set({ page: key }) }, page.label));
  }
  clear($("nav"), items);
}

function renderPage() {
  const s = store.get();
  const explore = s.page === "explore";
  $("explore").classList.toggle("hidden", !explore);
  $("page").classList.toggle("hidden", explore);
  if (s.page !== "replay") stopReplay();
  if (explore) {
    clear($("page"));
    for (const view of ["header", "phasebar", "left", "tabs", "center", "inspector"]) RENDER[view]();
    return;
  }
  const page = PAGES[s.page] || PAGES.replay;
  page.render($("page"));
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
  runselect: ["runs", "runId", "meta"],
  page: ["page", "allBatches", "meta", "steps", "sample", "batchIndex", "runId"],
  header: ["runs", "runId", "mode", "batchIndex", "stepIndex", "steps", "meta", "sample", "batch"],
  phasebar: ["batch", "mode", "stepBatches"],
  left: ["batch", "mode", "filter", "kinds", "hidden", "expanded", "selection", "stepBatches", "steps", "stepIndex"],
  tabs: ["tab", "mode"],
  center: ["tab", "batch", "mode", "selection", "sample", "meta", "stepBatches", "allBatches", "sourceFile", "sourceLine", "hidden"],
  inspector: ["selection", "batch", "sample", "stepBatches", "mode"],
};
const RENDER = {
  nav: renderNav,
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
  $("theme-btn").addEventListener("click", toggleTheme);
  $("notices-btn").addEventListener("click", () => store.set({ page: "explore", tab: "overview" }));
  $("hidden-btn").addEventListener("click", showHiddenDialog);
  document.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.metaKey || e.ctrlKey) return;
    const s = store.get();
    if (s.page === "replay" && replayKey(e)) return;
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
  back.append(h("div", { class: "modal" }, h("h3", {}, "Hidden calls"), list.length ? list : h("div", { class: "faint" }, "Nothing hidden. Use ⊘ on a row in the call tree to hide calls like it; their children stay visible."), h("div", { style: { marginTop: "14px", textAlign: "right" } }, h("button", { class: "chip", onclick: close }, "Close"))));
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
  return { run: params.get("run"), page: params.get("page"), mode: params.get("mode"), b: int("b"), s: int("s"), call: int("call"), sample: int("sample"), tab: params.get("tab") };
}

const writeHash = debounce(() => {
  const s = store.get();
  if (!s.runId) return;
  const params = new URLSearchParams({ run: s.runId, page: s.page, mode: s.mode, tab: s.tab });
  if (s.mode === "batch") params.set("b", s.batchIndex);
  else params.set("s", s.stepIndex);
  if (s.selection?.call != null) params.set("call", s.selection.call);
  if (s.sample != null) params.set("sample", s.sample);
  history.replaceState(null, "", `#${params}`);
}, 100);

window.addEventListener("resize", debounce(() => store.get().page === "explore" && store.get().tab === "timeline" && renderCenter(), 150));
boot();
