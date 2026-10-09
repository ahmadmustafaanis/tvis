// Weights page: one layer at a time (or every layer as thumbnails), weight / gradient / update,
// across the captured steps.

import { api } from "./api.js";
import { store } from "./store.js";
import { colorFor, drawHeat, drawHistogram, tensorCard } from "./tensorview.js";
import { clear, cssVar, decodeFloat32, fmtNum, fmtShape, h } from "./util.js";

const VIEWS = [
  ["weight", "Weight"],
  ["grad", "Gradient"],
  ["update", "Update Δw"],
];
const W = { param: null, view: "weight", step: 0, all: false };

function ratioColor(r) {
  if (r == null) return "var(--text-faint)";
  if (r < 1e-5 || r > 1e-1) return "var(--bad)";
  if (r < 1e-4 || r > 1e-2) return "var(--warn)";
  return "var(--good)";
}

export function renderWeights(container) {
  const state = store.get();
  const steps = state.steps;
  if (!steps.length || !state.allBatches) return clear(container, h("div", { class: "empty" }, steps.length ? "loading…" : "No optimizer steps were captured, so there are no weight updates to show."));
  W.step = Math.min(W.step, steps.length - 1);
  const step = steps[W.step];
  if (!step.params.some((p) => p.name === W.param)) W.param = step.params[0]?.name;
  const batchOf = (ref) => state.allBatches.find((b) => b.index === ref?.batch);
  const list = h(
    "div",
    { class: "plist" },
    h("div", { class: "grp" }, `${step.params.length} parameters · ‖Δw‖/‖w‖ · shape`),
    step.params.map((p) =>
      h(
        "div",
        { class: `p ${!W.all && p.name === W.param ? "on" : ""}`, onclick: () => ((W.param = p.name), (W.all = false), renderWeights(container)) },
        h("span", { class: "n" }, p.name),
        h("span", { class: "s", style: { color: ratioColor(p.update_ratio) }, title: "‖Δw‖/‖w‖" }, p.update_ratio != null ? p.update_ratio.toExponential(0) : ""),
        h("span", { class: "s" }, fmtShape(p.shape)),
      ),
    ),
  );
  const controls = h(
    "div",
    { class: "controls" },
    h("div", { class: "seg" }, VIEWS.map(([k, l]) => h("button", { class: W.view === k ? "on" : "", onclick: () => ((W.view = k), renderWeights(container)) }, l))),
    h(
      "label",
      {},
      "step",
      h("input", { type: "range", min: 0, max: steps.length - 1, value: W.step, oninput: (e) => ((W.step = Number(e.target.value)), renderWeights(container)) }),
      h("b", {}, `${W.step}`),
      h("span", { class: "faint" }, `/ ${steps.length - 1}`),
    ),
    h("span", { class: "faint" }, W.view === "weight" ? `values before optimizer step ${W.step}` : W.view === "grad" ? `gradient used by step ${W.step}` : `change applied by step ${W.step}`),
    h("div", { style: { flex: 1 } }),
    h("div", { class: "seg" }, [[false, "One parameter"], [true, "All"]].map(([all, label]) => h("button", { class: W.all === all ? "on" : "", onclick: () => ((W.all = all), renderWeights(container)) }, label))),
  );
  const main = h("div", { class: "page-wrap", style: { maxWidth: "none" } });
  clear(container, h("div", { class: "weights" }, list, main));
  main.append(controls);
  if (W.all) return main.append(allLayers(step, batchOf, state));
  const p = step.params.find((x) => x.name === W.param);
  if (!p) return;
  const ref = p[W.view];
  const batch = batchOf(ref);
  const meta = ref && batch ? batch.tensors[ref.tid] : null;
  main.append(
    h(
      "div",
      { class: "kpis", style: { marginBottom: "24px" } },
      [
        ["‖w‖", fmtNum(p.weight_norm)],
        ["‖grad‖", fmtNum(p.grad_norm)],
        ["‖Δw‖", fmtNum(p.update_norm)],
        ["‖Δw‖ / ‖w‖", p.update_ratio != null ? p.update_ratio.toExponential(2) : "—"],
      ].map(([label, value]) => h("div", { class: "kpi" }, h("div", { class: "label" }, label), h("div", { class: "value", style: label.includes("/") ? { color: ratioColor(p.update_ratio) } : {} }, value))),
    ),
  );
  if (!meta) return main.append(h("div", { class: "faint" }, "not recorded for this step"));
  const viewer = h("div", { class: "card" });
  main.append(viewer);
  if (p.shape.length === 4 && meta.stored === "full") kernelGrid(viewer, ref, p, state);
  else viewer.append(tensorCard({ run: state.runId, ref, meta, name: `${p.name} · ${VIEWS.find((v) => v[0] === W.view)[1]}`, open: true }));
  if (p.shape.length === 4) {
    main.append(h("div", { style: { height: "12px" } }), tensorCard({ run: state.runId, ref, meta, name: "statistics & raw slices" }));
  }
}

async function kernelGrid(card, ref, p, state) {
  const [out, inp, kh, kw] = p.shape;
  const maxOut = 48;
  const maxIn = 16;
  card.append(
    h("h3", {}, `${p.name}: ${out} filters × ${inp} input channels, ${kh}×${kw} kernels`),
    h("div", { class: "hint" }, `rows = output filters${out > maxOut ? ` (first ${maxOut})` : ""}, columns = input channels${inp > maxIn ? ` (first ${maxIn})` : ""}; one shared colour scale (blue < 0 < red)`),
  );
  const slot = h("div", { class: "loading" }, "loading…");
  card.append(slot);
  let data;
  try {
    data = await api("array", { run: state.runId, batch: ref.batch, tid: ref.tid });
  } catch (err) {
    return clear(slot, h("div", { class: "err" }, err.message));
  }
  const values = decodeFloat32(data.data);
  const cmap = colorFor(data.min, data.max);
  const rows = Math.min(out, maxOut);
  const cols = Math.min(inp, maxIn);
  const scale = Math.max(4, Math.floor(28 / Math.max(kh, kw)));
  const grid = h("div", { class: "kernels", style: { gridTemplateColumns: `repeat(${cols}, ${kw * scale}px)` } });
  for (let o = 0; o < rows; o++) {
    for (let i = 0; i < cols; i++) {
      const canvas = h("canvas", { width: kw, height: kh, style: { width: `${kw * scale}px`, height: `${kh * scale}px` }, title: `filter ${o}, input channel ${i}` });
      const ctx = canvas.getContext("2d");
      const img = ctx.createImageData(kw, kh);
      for (let y = 0; y < kh; y++) {
        for (let x = 0; x < kw; x++) {
          const v = values[((o * inp + i) * kh + y) * kw + x];
          img.data.set([...cmap.fn(v), 255], (y * kw + x) * 4);
        }
      }
      ctx.putImageData(img, 0, 0);
      grid.append(canvas);
    }
  }
  clear(slot, grid);
  slot.classList.remove("loading");
}

function allLayers(step, batchOf, state) {
  const grid = h("div", { class: "allw" });
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      observer.unobserve(entry.target);
      entry.target.dispatchEvent(new Event("visible"));
    }
  });
  for (const p of step.params) {
    const ref = p[W.view];
    const batch = batchOf(ref);
    const meta = ref && batch ? batch.tensors[ref.tid] : null;
    const thumb = h("canvas", { class: "thumbc" });
    const hist = h("canvas", { class: "hist" });
    const card = h(
      "div",
      { class: "card", onclick: () => ((W.param = p.name), (W.all = false), store.set({ page: "weights" })) },
      h("div", { style: { display: "flex", justifyContent: "space-between", gap: "6px" } }, h("span", { class: "mono", style: { overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" } }, p.name), h("span", { class: "mono faint" }, fmtShape(p.shape))),
      thumb,
      hist,
      h("div", { class: "faint", style: { fontSize: "11px" } }, `σ ${fmtNum(meta?.stats?.std, 3)} · ‖Δw‖/‖w‖ `, h("span", { style: { color: ratioColor(p.update_ratio) } }, p.update_ratio != null ? p.update_ratio.toExponential(1) : "—")),
    );
    card.addEventListener("visible", () => {
      drawHistogram(hist, meta?.stats?.hist, cssVar("--accent"));
      if (meta && meta.stored !== "none") {
        api("thumb", { run: state.runId, batch: ref.batch, tid: ref.tid, size: 96, layout: "matrix" })
          .then((view) => drawHeat(thumb, view))
          .catch(() => thumb.remove());
      }
    });
    grid.append(card);
    observer.observe(card);
  }
  return grid;
}
