// Grad-CAM page: where in each image the model looked for its predicted (or the true) class.

import { api } from "./api.js";
import { paintOverlay } from "./overlay.js";
import { openInReplay, store } from "./store.js";
import { clear, decodeFloat32, fmtNum, h } from "./util.js";

const G = { layer: null, which: "pred", opacity: 0.55 };

export function camLayers(batch) {
  return batch.calls.filter((c) => c.extra?.gradcam);
}

export function renderGradcam(container) {
  const state = store.get();
  const batches = (state.allBatches || []).filter((b) => camLayers(b).length);
  if (!batches.length) return clear(container, h("div", { class: "empty" }, "No Grad-CAM was recorded. It needs conv layers, image inputs and a cross-entropy / NLL loss (and wasn't disabled with --no-gradcam)."));
  const batch = batches.find((b) => b.index === state.batchIndex) || batches[0];
  const layers = camLayers(batch);
  if (!layers.some((c) => c.name === G.layer)) G.layer = layers[layers.length - 1].name;
  const call = layers.find((c) => c.name === G.layer);
  const grid = h("div", { class: "cams" });
  clear(
    container,
    h(
      "div",
      { class: "page-wrap" },
      h("div", { class: "page-head" }, h("h1", {}, "Grad-CAM"), h("span", { class: "lede" }, "Which regions of each image drove the class score, at a chosen conv layer. Red = strong evidence.")),
      h(
        "div",
        { class: "controls" },
        h("label", {}, "batch", h("select", { class: "select", onchange: (e) => store.set({ batchIndex: Number(e.target.value) }) }, batches.map((b) => h("option", { value: b.index, selected: b.index === batch.index }, `b${b.index}`)))),
        h("label", {}, "layer", h("select", { class: "select", onchange: (e) => ((G.layer = e.target.value), renderGradcam(container)) }, layers.map((c) => h("option", { value: c.name, selected: c.name === G.layer }, `${c.name} (${c.cls})`)))),
        h("div", { class: "seg" }, [["pred", "for predicted class"], ["target", "for true class"]].map(([k, l]) => h("button", { class: G.which === k ? "on" : "", onclick: () => ((G.which = k), renderGradcam(container)) }, l))),
        h("label", {}, "overlay", h("input", { type: "range", min: 0, max: 100, value: G.opacity * 100, onchange: (e) => ((G.opacity = Number(e.target.value) / 100), renderGradcam(container)) })),
      ),
      h("div", { class: "hint" }, "Computed during capture with an extra gradient pass of the class score (training numerics untouched). Deeper layers are coarser but more semantic."),
      grid,
    ),
  );
  for (const s of batch.samples || []) grid.append(camCard(batch, call, s, state));
}

function camCard(batch, call, sample, state) {
  const canvas = h("canvas", { width: 160, height: 160 });
  const p = sample.prediction || {};
  const card = h(
    "div",
    { class: "cam", onclick: () => openInReplay(batch.index, sample.position), style: { cursor: "pointer" }, title: "open this sample in Replay" },
    canvas,
    h("div", { class: "lab" }, h("span", { class: "faint" }, `#${sample.position}`), h("span", {}, "true ", h("b", {}, sample.target?.name ?? sample.target?.id))),
    h("div", { class: "lab" }, h("span", { style: { color: p.correct ? "var(--good)" : "var(--bad)" } }, `${p.correct ? "✓" : "✗"} ${p.name ?? p.id ?? "?"}`), h("span", { class: "faint" }, p.p != null ? fmtNum(p.p, 2) : "")),
  );
  const camRef = call.extra.gradcam[G.which];
  const ref = sample.input; // the model's input, so the CAM aligns with it
  Promise.all([
    api("image", { run: state.runId, batch: ref.batch, tid: ref.tid, sample: sample.position }),
    camRef ? api("array", { run: state.runId, batch: camRef.batch, tid: camRef.tid, sample: sample.position }) : null,
  ])
    .then(([image, cam]) => paint(canvas, image, cam))
    .catch(() => card.append(h("div", { class: "err" }, "no image")));
  return card;
}

function paint(canvas, image, cam) {
  paintOverlay(canvas, image, cam ? decodeFloat32(cam.data) : null, cam?.shape[0], cam?.shape[1], G.opacity);
}

