// Attention page: where each token looks, per layer and head, plus attention rollout.

import { api } from "./api.js";
import { paintOverlay } from "./gradcam.js";
import { followableSamples, index } from "./model.js";
import { store } from "./store.js";
import { colorFor } from "./tensorview.js";
import { clear, cssVar, decodeFloat32, fmtNum, h } from "./util.js";

const A = { layer: 0, head: "mean", view: "layer", query: 0 };

export function attentionLayers(batch) {
  const idx = index(batch);
  const layers = [];
  for (const op of batch.ops || []) {
    if (!op.attention || !op.value) continue;
    const call = idx.byId.get(op.call);
    let owner = call;
    while (owner && owner.kind !== "module") owner = idx.byId.get(owner.parent);
    const where = owner?.module_path || owner?.name || "";
    layers.push({ op, label: `${where}${where ? " · " : ""}${call?.name ?? "softmax"} (${op.file}:${op.line})`, shape: op.shapes[0] });
  }
  return layers;
}

export function renderAttention(container) {
  const state = store.get();
  const batches = (state.allBatches || []).filter((b) => attentionLayers(b).length);
  if (!batches.length) return clear(container, h("div", { class: "empty" }, "No attention weights were recorded. tvis stores softmax outputs shaped [B, H, T, T] or [B, T, T] computed in your own code."));
  const batch = batches.find((b) => b.index === state.batchIndex) || batches[0];
  const layers = attentionLayers(batch);
  A.layer = Math.min(A.layer, layers.length - 1);
  const { total, limit: n } = followableSamples(batch);
  const sample = Math.min(state.sample ?? 0, n - 1);
  const layer = layers[A.layer];
  const heads = layer.shape.length === 4 ? layer.shape[1] : 1;
  if (A.head !== "mean" && A.head >= heads) A.head = "mean";
  const plot = h("div", { class: "attn-wrap" }, h("div", { class: "loading" }, "loading…"));
  const readout = h("div", { class: "scatter-tip" });
  clear(
    container,
    h(
      "div",
      { class: "page-wrap" },
      h("div", { class: "page-head" }, h("h1", {}, "Attention"), h("span", { class: "lede" }, "Row = the token doing the looking (query), column = the token it looks at (key). Brighter = more attention.")),
      h(
        "div",
        { class: "controls" },
        h("label", {}, "batch", h("select", { class: "select", onchange: (e) => store.set({ batchIndex: Number(e.target.value) }) }, batches.map((b) => h("option", { value: b.index, selected: b.index === batch.index }, `b${b.index}`)))),
        h("label", {}, "sample", h("select", { class: "select", onchange: (e) => store.set({ sample: Number(e.target.value) }) }, Array.from({ length: n }, (_, i) => h("option", { value: i, selected: i === sample }, `#${i}`)))),
        n < total ? h("span", { class: "faint", title: "raise --max-elems to store more samples" }, `first ${n} of ${total} samples stored`) : null,
        h("div", { class: "seg" }, [["layer", "One layer"], ["rollout", "Rollout (all layers)"]].map(([k, l]) => h("button", { class: A.view === k ? "on" : "", onclick: () => ((A.view = k), renderAttention(container)) }, l))),
        A.view === "layer"
          ? h("label", {}, "layer", h("select", { class: "select", onchange: (e) => ((A.layer = Number(e.target.value)), renderAttention(container)) }, layers.map((l, i) => h("option", { value: i, selected: i === A.layer }, l.label))))
          : null,
        A.view === "layer" && heads > 1
          ? h(
              "label",
              {},
              "head",
              h(
                "select",
                { class: "select", onchange: (e) => ((A.head = e.target.value === "mean" ? "mean" : Number(e.target.value)), renderAttention(container)) },
                h("option", { value: "mean", selected: A.head === "mean" }, "mean of heads"),
                Array.from({ length: heads }, (_, i) => h("option", { value: i, selected: A.head === i }, `head ${i}`)),
              ),
            )
          : null,
      ),
      h("div", { class: "card" }, plot, readout),
      h("div", { id: "attn-image" }),
      A.view === "rollout" ? h("div", { class: "hint", style: { marginTop: "8px" } }, "Rollout multiplies (½·A + ½·I) across layers (heads averaged) to estimate how much each output position draws on each input token overall.") : null,
    ),
  );
  draw(plot, readout, batch, layers, sample, state).catch((err) => clear(plot, h("div", { class: "err" }, err.message)));
}

/** For image models (ViT): how many prefix tokens (cls/registers) precede a square patch grid. */
function patchLayout(batch, sample, T) {
  const s = batch.samples?.[sample];
  const meta = s?.input ? batch.tensors[s.input.tid] : null;
  if (!meta || meta.shape.length !== 4 || ![1, 3].includes(meta.shape[1])) return null;
  for (let prefix = 0; prefix <= 8 && prefix < T; prefix++) {
    const grid = Math.round(Math.sqrt(T - prefix));
    if (grid > 1 && grid * grid === T - prefix) return { prefix, grid, input: s.input };
  }
  return null;
}

async function tokenLabels(batch, sample, state, T) {
  const layout = patchLayout(batch, sample, T);
  if (layout) {
    const names = Array.from({ length: layout.prefix }, (_, i) => (i === 0 ? "CLS" : `reg${i}`));
    for (let r = 0; r < layout.grid; r++) for (let c = 0; c < layout.grid; c++) names.push(`r${r}c${c}`);
    return names;
  }
  const s = batch.samples?.[sample];
  const vocab = state.meta?.data?.vocab;
  if (s?.input && vocab) {
    try {
      const ids = await api("array", { run: state.runId, batch: s.input.batch, tid: s.input.tid, sample });
      const values = Array.from(decodeFloat32(ids.data));
      if (values.length === T) return values.map((id) => vocab[Math.round(id)] ?? String(id));
    } catch {
      /* fall through to positions */
    }
  }
  return Array.from({ length: T }, (_, i) => String(i));
}

async function matrixFor(layer, sample, state, head) {
  const data = await api("array", { run: state.runId, batch: layer.op.value.batch, tid: layer.op.value.tid, sample });
  const values = decodeFloat32(data.data);
  const shape = data.shape; // [H, T, T] or [T, T]
  const T = shape[shape.length - 1];
  const Tq = shape[shape.length - 2];
  const H = shape.length === 3 ? shape[0] : 1;
  const m = Array.from({ length: Tq }, () => new Float32Array(T));
  for (let hh = 0; hh < H; hh++) {
    if (head !== "mean" && hh !== head) continue;
    for (let q = 0; q < Tq; q++) for (let k = 0; k < T; k++) m[q][k] += values[(hh * Tq + q) * T + k] / (head === "mean" ? H : 1);
  }
  return m;
}

function rollout(mats) {
  const T = mats[0].length;
  let result = Array.from({ length: T }, (_, i) => Float32Array.from({ length: T }, (_, j) => (i === j ? 1 : 0)));
  for (const a of mats) {
    const aug = a.map((row, i) => {
      const r = Float32Array.from(row, (v, j) => 0.5 * v + (i === j ? 0.5 : 0));
      const sum = r.reduce((x, y) => x + y, 0) || 1;
      return r.map((v) => v / sum);
    });
    result = aug.map((row) => Float32Array.from({ length: T }, (_, j) => row.reduce((acc, v, k) => acc + v * result[k][j], 0)));
  }
  return result;
}

async function draw(plot, readout, batch, layers, sample, state) {
  let matrix;
  let title;
  if (A.view === "rollout") {
    const mats = [];
    for (const layer of layers) mats.push(await matrixFor(layer, sample, state, "mean"));
    if (mats.some((m) => m.length !== mats[0].length || m[0].length !== mats[0].length)) throw new Error("layers have different sequence lengths; rollout needs square, equal-size maps");
    matrix = rollout(mats);
    title = `rollout over ${layers.length} layers`;
  } else {
    matrix = await matrixFor(layers[A.layer], sample, state, A.head);
    title = layers[A.layer].label;
  }
  const T = matrix[0].length;
  const labels = await tokenLabels(batch, sample, state, T);
  const qLabels = matrix.length === T ? labels : Array.from({ length: matrix.length }, (_, i) => String(i));
  const cell = Math.max(14, Math.min(34, Math.floor(560 / T)));
  const left = 90;
  const top = 80;
  const canvas = h("canvas", {});
  const dpr = window.devicePixelRatio || 1;
  const width = left + cell * T + 10;
  const height = top + cell * matrix.length + 10;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const max = Math.max(...matrix.map((r) => Math.max(...r)), 1e-9);
  const cmap = colorFor(0, max);
  matrix.forEach((row, q) => {
    row.forEach((v, k) => {
      const [r, g, b] = cmap.fn(v);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(left + k * cell, top + q * cell, cell - 1, cell - 1);
    });
  });
  ctx.fillStyle = cssVar("--text-muted");
  ctx.font = `${Math.min(12, cell - 3)}px ${cssVar("--mono")}`;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  qLabels.forEach((label, q) => ctx.fillText(label.slice(0, 12), left - 6, top + q * cell + cell / 2));
  labels.forEach((label, k) => {
    ctx.save();
    ctx.translate(left + k * cell + cell / 2, top - 6);
    ctx.rotate(-Math.PI / 3);
    ctx.textAlign = "left";
    ctx.fillText(label.slice(0, 12), 0, 0);
    ctx.restore();
  });
  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const q = Math.floor((e.clientY - rect.top - top) / cell);
    if (q >= 0 && q < matrix.length) {
      A.query = q;
      drawPatchOverlay(batch, sample, state, matrix, qLabels);
    }
  });
  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const k = Math.floor((e.clientX - rect.left - left) / cell);
    const q = Math.floor((e.clientY - rect.top - top) / cell);
    if (q < 0 || k < 0 || q >= matrix.length || k >= T) return (readout.textContent = "");
    readout.textContent = `"${qLabels[q]}" (query ${q}) → "${labels[k]}" (key ${k}): ${fmtNum(matrix[q][k], 4)}`;
  });
  clear(plot, h("div", { class: "faint", style: { marginBottom: "8px" } }, title), canvas);
  drawPatchOverlay(batch, sample, state, matrix, qLabels);
}

async function drawPatchOverlay(batch, sample, state, matrix, qLabels) {
  const slot = document.getElementById("attn-image");
  const layout = patchLayout(batch, sample, matrix[0].length);
  if (!slot || !layout) return;
  const q = Math.min(A.query, matrix.length - 1);
  const row = matrix[q].slice(layout.prefix);
  const peak = Math.max(...row, 1e-12);
  const canvas = h("canvas", { width: 280, height: 280, style: { borderRadius: "8px", width: "280px" } });
  clear(
    slot,
    h(
      "div",
      { class: "card", style: { marginTop: "14px" } },
      h("h3", {}, `Where "${qLabels[q]}" looks in the image`),
      h("div", { class: "hint" }, `Attention from query ${q} to the ${layout.grid}×${layout.grid} patch tokens${layout.prefix ? ` (${layout.prefix} prefix token${layout.prefix > 1 ? "s" : ""} — CLS/registers — left out)` : ""}. Click a row of the map above to pick another query.`),
      canvas,
    ),
  );
  try {
    const image = await api("image", { run: state.runId, batch: layout.input.batch, tid: layout.input.tid, sample });
    paintOverlay(canvas, image, row.map((v) => v / peak), layout.grid, layout.grid, 0.6);
  } catch (err) {
    canvas.replaceWith(h("div", { class: "err" }, err.message));
  }
}
