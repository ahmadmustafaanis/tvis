// One layer's output (or its gradient) for the followed sample, three ways:
//   On image — activation strength / attention / gradient painted over the input picture
//   Neurons  — every unit as its own map (conv channels, ViT hidden dims) or bars (vectors),
//              click one for its raw numbers
//   Stats    — histogram + heatmap slices (the generic tensor card)

import { api } from "./api.js";
import { firstOutput, imageInput, isDescendant, patchGrid, sampleShape } from "./model.js";
import { paintOverlay } from "./overlay.js";
import { colorFor, tensorCard } from "./tensorview.js";
import { clear, decodeFloat32, fmtNum, fmtShape, h } from "./util.js";

const MAX_TILES = 64;
const memory = new Map(); // per call name: {tab, unit, sort, location, query}

export function layerView({ run, batch, call, sample, grad = false, state }) {
  const out = firstOutput(call);
  const meta = out ? batch.tensors[out.tid] : null;
  if (!meta) return h("div", { class: "faint" }, "This call has no tensor output.");
  const shape = meta.batch_dim ? sampleShape(meta) : meta.shape;
  const img = imageInput(batch, sample);
  const kind = layoutOf(shape, img);
  const attention = grad ? [] : (batch.ops || []).filter((o) => o.attention && o.value && isDescendant(batch, o.call, call.id));
  const tabs = [];
  if (img && (kind.type === "chw" || kind.type === "tokens" || attention.length)) tabs.push(["image", "On image"]);
  if (kind.type !== "other") tabs.push(["neurons", "Neurons"]);
  tabs.push(["stats", "Stats"]);
  const mem = memory.get(call.name) || { tab: tabs[0][0], unit: null, sort: "active", location: null, query: null };
  memory.set(call.name, mem);
  if (!tabs.some(([k]) => k === mem.tab)) mem.tab = tabs[0][0];

  const body = h("div", {});
  const wrap = h(
    "div",
    { class: "layerview" },
    h(
      "div",
      { class: "toolbar" },
      h("div", { class: "seg" }, tabs.map(([k, label]) => h("button", { class: mem.tab === k ? "on" : "", onclick: () => ((mem.tab = k), render()) }, label))),
      h("span", { class: "faint" }, `${grad ? "dL/d(output)" : "output"} ${fmtShape(shape)} · sample #${sample}`),
    ),
    body,
  );
  const stored = grad ? meta.grad_stored : meta.stored;
  const rows = grad ? meta.grad_stored_rows : meta.stored_rows;
  const ctx = { run, batch, call, out, meta, shape, img, kind, attention, sample, grad, state, mem, body };

  function render() {
    for (const b of wrap.querySelectorAll(".toolbar .seg button")) b.classList.toggle("on", b.textContent === tabs.find(([k]) => k === mem.tab)[1]);
    if (mem.tab === "stats") return clear(body, tensorCard({ run, ref: out, meta, name: grad ? "dL/d(output)" : "output", sample, grad, open: true }));
    if (!stored || stored === "none" || (meta.batch_dim && stored === "rows" && rows != null && sample >= rows / (meta.batch_dim.block || 1))) {
      return clear(body, h("div", { class: "faint" }, "Values for this sample weren't stored (only statistics). Use Stats, or re-record with a larger --sample-rows / --max-elems."));
    }
    clear(body, h("div", { class: "loading" }, "loading…"));
    loadValues(ctx)
      .then((values) => (mem.tab === "image" ? renderOnImage(ctx, values, render) : renderNeurons(ctx, values, render)))
      .catch((err) => clear(body, h("div", { class: "err" }, err.message)));
  }
  render();
  return wrap;
}

function layoutOf(shape, img) {
  if (shape.length === 3 && shape[1] > 1 && shape[2] > 1) return { type: "chw", units: shape[0], rows: shape[1], cols: shape[2] };
  if (shape.length === 2) {
    const grid = patchGrid(shape[0]);
    if (grid && img) return { type: "tokens", units: shape[1], rows: grid.grid, cols: grid.grid, prefix: grid.prefix, tokens: shape[0] };
    return { type: "seq", tokens: shape[0], units: shape[1] };
  }
  if (shape.length === 1) return { type: "vector", units: shape[0] };
  return { type: "other" };
}

async function loadValues(ctx) {
  const params = { run: ctx.run, batch: ctx.out.batch, tid: ctx.out.tid, grad: ctx.grad };
  if (ctx.meta.batch_dim) params.sample = ctx.sample;
  const data = await api("array", params);
  return decodeFloat32(data.data);
}

/** Unit u's spatial map (rows × cols) for conv channels or ViT hidden dimensions. */
function unitMap(ctx, values, u) {
  const { kind } = ctx;
  const n = kind.rows * kind.cols;
  const map = new Float32Array(n);
  if (kind.type === "chw") map.set(values.subarray(u * n, (u + 1) * n));
  else for (let p = 0; p < n; p++) map[p] = values[(kind.prefix + p) * kind.units + u];
  return map;
}

/** Strength per location: mean |x| over channels (conv) or the L2 norm of each patch token (ViT). */
function strengthMap(ctx, values) {
  const { kind } = ctx;
  const n = kind.rows * kind.cols;
  const map = new Float32Array(n);
  if (kind.type === "chw") {
    for (let c = 0; c < kind.units; c++) for (let p = 0; p < n; p++) map[p] += Math.abs(values[c * n + p]) / kind.units;
  } else {
    for (let p = 0; p < n; p++) {
      let sum = 0;
      for (let d = 0; d < kind.units; d++) sum += values[(kind.prefix + p) * kind.units + d] ** 2;
      map[p] = Math.sqrt(sum);
    }
  }
  return map;
}

function normalise(map) {
  let peak = 0;
  for (const v of map) peak = Math.max(peak, Math.abs(v));
  return Float32Array.from(map, (v) => Math.abs(v) / (peak || 1));
}

async function overlayCanvas(ctx, map, rows, cols, onPick, size = 260) {
  const canvas = h("canvas", { width: size, height: size, style: { width: `${size}px`, height: `${size}px`, borderRadius: "8px", cursor: onPick ? "crosshair" : "default" } });
  try {
    const image = await api("image", { run: ctx.run, batch: ctx.img.batch, tid: ctx.img.tid, sample: ctx.sample });
    paintOverlay(canvas, image, normalise(map), rows, cols, 0.6);
  } catch {
    /* image unavailable: leave blank */
  }
  if (onPick) {
    canvas.addEventListener("click", (e) => {
      const r = canvas.getBoundingClientRect();
      onPick(Math.floor(((e.clientY - r.top) / r.height) * rows), Math.floor(((e.clientX - r.left) / r.width) * cols));
    });
  }
  return canvas;
}

const locName = (ctx, row, col) => (ctx.kind.type === "tokens" ? `patch r${row}c${col}` : `pixel (${row}, ${col})`);

// ---------------------------------------------------------------------------------------------
// On image
// ---------------------------------------------------------------------------------------------
async function renderOnImage(ctx, values, rerender) {
  const { mem, kind, body } = ctx;
  const modes = [];
  if (kind.type === "chw" || kind.type === "tokens") modes.push(["strength", ctx.grad ? "Gradient strength" : "Activation strength"]);
  if (ctx.attention.length) modes.push(["attention", "Attention"]);
  if (!modes.some(([k]) => k === mem.mode)) mem.mode = modes[0][0];
  const controls = h(
    "div",
    { class: "toolbar" },
    modes.length > 1 ? h("div", { class: "seg" }, modes.map(([k, l]) => h("button", { class: mem.mode === k ? "on" : "", onclick: () => ((mem.mode = k), rerender()) }, l))) : null,
  );
  const side = h("div", { class: "lv-side" });
  let canvas;
  let caption;
  if (mem.mode === "attention") {
    const op = ctx.attention[0];
    const data = await api("array", { run: ctx.run, batch: op.value.batch, tid: op.value.tid, sample: ctx.sample });
    const att = decodeFloat32(data.data);
    const [heads, T] = data.shape.length === 3 ? [data.shape[0], data.shape[2]] : [1, data.shape[1]];
    const layout = patchGrid(T);
    if (!layout) return clear(body, controls, h("div", { class: "faint" }, "Attention isn't over an image patch grid here."));
    const query = mem.query ?? (layout.prefix ? 0 : layout.prefix);
    const map = new Float32Array(layout.grid * layout.grid);
    for (let hh = 0; hh < heads; hh++) for (let p = 0; p < map.length; p++) map[p] += att[(hh * T + query) * T + layout.prefix + p] / heads;
    canvas = await overlayCanvas(ctx, map, layout.grid, layout.grid, (r, c) => ((mem.query = layout.prefix + r * layout.grid + c), rerender()));
    const qName = query < layout.prefix ? (query === 0 ? "[CLS]" : `register ${query}`) : `patch r${Math.floor((query - layout.prefix) / layout.grid)}c${(query - layout.prefix) % layout.grid}`;
    caption = h(
      "div",
      {},
      h("div", {}, h("b", {}, `Where ${qName} looks`), ` · mean of ${heads} head${heads > 1 ? "s" : ""}`),
      h("div", { class: "faint" }, "Click a patch to see where that patch looks instead."),
      mem.query != null ? h("a", { class: "link", onclick: () => ((mem.query = null), rerender()) }, "back to [CLS]") : null,
    );
  } else {
    const map = strengthMap(ctx, values);
    let peak = 0;
    map.forEach((v, i) => (v > map[peak] ? (peak = i) : null));
    canvas = await overlayCanvas(ctx, map, kind.rows, kind.cols, (r, c) => ((mem.location = [r, c]), rerender()));
    caption = h(
      "div",
      {},
      h("div", {}, h("b", {}, ctx.grad ? "Where the loss pushes this layer" : "Where this layer responds"), kind.type === "chw" ? ` · mean |x| over ${kind.units} channels` : ` · ‖token‖ over ${kind.units} dims`),
      h("div", { class: "faint" }, `strongest at ${locName(ctx, Math.floor(peak / kind.cols), peak % kind.cols)} · click any location for its raw neuron values`),
    );
    if (mem.location) side.append(locationVector(ctx, values, ...mem.location, rerender));
  }
  clear(body, controls, h("div", { class: "lv-row" }, canvas, h("div", { class: "lv-caption" }, caption)), side);
}

/** (b) every neuron's value at one location: a patch token's vector, or the channels at a pixel. */
function locationVector(ctx, values, row, col, rerender) {
  const { kind } = ctx;
  const n = kind.rows * kind.cols;
  const p = row * kind.cols + col;
  const vec = Float32Array.from({ length: kind.units }, (_, u) => (kind.type === "chw" ? values[u * n + p] : values[(kind.prefix + p) * kind.units + u]));
  return h("div", { class: "card", style: { marginTop: "12px" } }, h("h3", {}, `${locName(ctx, row, col)}: all ${kind.units} neurons`), bars(vec, null, (u) => ((ctx.mem.unit = u), (ctx.mem.tab = "neurons"), rerender())));
}

// ---------------------------------------------------------------------------------------------
// Neurons
// ---------------------------------------------------------------------------------------------
function renderNeurons(ctx, values, rerender) {
  const { kind, mem, body } = ctx;
  if (kind.type === "vector") {
    const labels = vectorLabels(ctx);
    return clear(body, h("div", { class: "hint" }, `${kind.units} neurons${labels ? " (labelled by class)" : ""}. Exact values below.`), bars(values, labels), rawTable(values, 1, kind.units, null, labels));
  }
  if (kind.type === "seq") return clear(body, seqTable(ctx, values));
  const units = Array.from({ length: kind.units }, (_, u) => {
    const map = unitMap(ctx, values, u);
    let act = 0;
    for (const v of map) act += Math.abs(v);
    return { u, map, act: act / map.length };
  });
  if (mem.sort === "active") units.sort((a, b) => b.act - a.act);
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) if (Number.isFinite(v)) (lo = Math.min(lo, v)), (hi = Math.max(hi, v));
  const cmap = colorFor(lo, hi);
  const shown = mem.showAll ? units : units.slice(0, MAX_TILES);
  const grid = h(
    "div",
    { class: "neuron-grid" },
    shown.map(({ u, map, act }) => {
      const tile = h("canvas", { width: kind.cols, height: kind.rows, title: `neuron ${u} · mean |x| ${fmtNum(act, 3)}` });
      const c2 = tile.getContext("2d");
      const img = c2.createImageData(kind.cols, kind.rows);
      map.forEach((v, i) => img.data.set([...cmap.fn(v), 255], i * 4));
      c2.putImageData(img, 0, 0);
      return h("div", { class: `neuron ${mem.unit === u ? "on" : ""}`, onclick: () => ((mem.unit = u), rerender()) }, tile, h("span", {}, `#${u}`));
    }),
  );
  const unitWord = kind.type === "chw" ? "channels" : "hidden dims";
  const header = h(
    "div",
    { class: "toolbar" },
    h("span", { class: "faint" }, `${kind.units} ${unitWord}, each as a ${kind.rows}×${kind.cols} map${kind.type === "tokens" ? " over patches" : ""}; one colour scale (blue < 0 < red)`),
    h("div", { class: "seg" }, [["active", "most active"], ["index", "by index"]].map(([k, l]) => h("button", { class: mem.sort === k ? "on" : "", onclick: () => ((mem.sort = k), rerender()) }, l))),
    units.length > MAX_TILES ? h("button", { class: "chip small", onclick: () => ((mem.showAll = !mem.showAll), rerender()) }, mem.showAll ? `top ${MAX_TILES}` : `all ${units.length}`) : null,
  );
  const detail = h("div", {});
  clear(body, header, grid, detail);
  if (mem.unit != null && mem.unit < kind.units) renderUnit(ctx, values, mem.unit, detail);
}

async function renderUnit(ctx, values, u, slot) {
  const { kind } = ctx;
  const map = unitMap(ctx, values, u);
  const parts = [h("h3", {}, `Neuron #${u}${ctx.grad ? " (gradient)" : ""}`)];
  if (ctx.img) parts.push(h("div", { class: "lv-row" }, await overlayCanvas(ctx, map, kind.rows, kind.cols, null, 200), h("div", { class: "lv-caption faint" }, "|value| painted over the input")));
  const rowLabels = Array.from({ length: kind.rows }, (_, i) => (kind.type === "tokens" ? `r${i}` : String(i)));
  const colLabels = Array.from({ length: kind.cols }, (_, i) => (kind.type === "tokens" ? `c${i}` : String(i)));
  parts.push(h("div", { class: "subhead" }, "raw values"), rawTable(map, kind.rows, kind.cols, rowLabels, colLabels));
  if (kind.type === "tokens" && kind.prefix) parts.push(h("div", { class: "faint" }, `[CLS]${kind.prefix > 1 ? " and registers" : ""}: ${Array.from({ length: kind.prefix }, (_, i) => fmtNum(values[i * kind.units + u], 4)).join(", ")}`));
  clear(slot, h("div", { class: "card", style: { marginTop: "12px" } }, parts));
}

function vectorLabels(ctx) {
  const classes = ctx.state.meta?.data?.dataset?.classes;
  return classes && classes.length === ctx.kind.units ? classes : null;
}

function seqTable(ctx, values) {
  const { kind } = ctx;
  const cols = Math.min(kind.units, 48);
  const sub = new Float32Array(kind.tokens * cols);
  for (let t = 0; t < kind.tokens; t++) for (let d = 0; d < cols; d++) sub[t * cols + d] = values[t * kind.units + d];
  return h("div", {}, h("div", { class: "hint" }, `rows = positions, columns = the first ${cols} of ${kind.units} neurons`), rawTable(sub, kind.tokens, cols));
}

// ---------------------------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------------------------
function bars(values, labels, onPick) {
  const n = values.length;
  let peak = 0;
  for (const v of values) peak = Math.max(peak, Math.abs(v));
  const W = 640;
  const H = 120;
  const canvas = h("canvas", { width: W, height: H, class: "bars", title: "click a bar to open that neuron" });
  const c2 = canvas.getContext("2d");
  const bw = W / n;
  const mid = H / 2;
  for (let i = 0; i < n; i++) {
    const v = values[i] / (peak || 1);
    c2.fillStyle = v >= 0 ? "rgb(214,64,69)" : "rgb(59,110,214)";
    c2.fillRect(i * bw, v >= 0 ? mid - v * (mid - 4) : mid, Math.max(1, bw - (bw > 3 ? 1 : 0)), Math.abs(v) * (mid - 4));
  }
  const readout = h("div", { class: "readout" }, `${n} values · max |x| ${fmtNum(peak, 4)}`);
  canvas.addEventListener("mousemove", (e) => {
    const r = canvas.getBoundingClientRect();
    const i = Math.min(n - 1, Math.floor(((e.clientX - r.left) / r.width) * n));
    readout.textContent = `${labels ? labels[i] : `neuron #${i}`} = ${fmtNum(values[i], 6)}`;
  });
  if (onPick) canvas.addEventListener("click", (e) => {
    const r = canvas.getBoundingClientRect();
    onPick(Math.min(n - 1, Math.floor(((e.clientX - r.left) / r.width) * n)));
  });
  return h("div", {}, canvas, readout);
}

/** Exact numbers, coloured by value. Capped at 64×64 cells. */
function rawTable(values, rows, cols, rowLabels = null, colLabels = null) {
  const R = Math.min(rows, 64);
  const C = Math.min(cols, 64);
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) if (Number.isFinite(v)) (lo = Math.min(lo, v)), (hi = Math.max(hi, v));
  const cmap = colorFor(lo, hi);
  const table = h("table", { class: "raw" });
  if (colLabels) table.append(h("tr", {}, h("th", {}), Array.from({ length: C }, (_, c) => h("th", {}, colLabels[c]))));
  for (let r = 0; r < R; r++) {
    table.append(
      h(
        "tr",
        {},
        h("th", {}, rowLabels ? rowLabels[r] : String(r)),
        Array.from({ length: C }, (_, c) => {
          const v = values[r * cols + c];
          const [cr, cg, cb] = cmap.fn(v);
          return h("td", { style: { background: `rgba(${cr | 0},${cg | 0},${cb | 0},0.55)` }, title: String(v) }, fmtNum(v, 3));
        }),
      ),
    );
  }
  const note = rows > R || cols > C ? h("div", { class: "faint" }, `showing ${R}×${C} of ${rows}×${cols}`) : null;
  return h("div", { class: "raw-wrap" }, table, note);
}
