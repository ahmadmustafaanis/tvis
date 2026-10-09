// Replay page: play the recorded steps back as if training were happening — data → each layer
// forward → loss → gradients flowing back → optimizer update → next batch.

import { api } from "./api.js";
import { firstOutput, followableSamples, index, lossValue } from "./model.js";
import { samplePipeline } from "./samples.js";
import { store } from "./store.js";
import { layerView } from "./layerview.js";
import { drawHeat, drawImage, tensorCard } from "./tensorview.js";
import { buildTransport, phaseJump } from "./transport.js";
import { clear, fmtMs, fmtNum, fmtShape, h } from "./util.js";

const GRANULARITIES = [
  ["layer", "Layer"],
  ["op", "Every call"],
  ["phase", "Phase"],
  ["batch", "Batch"],
];

const R = { key: null, events: [], i: 0, playing: false, timer: null, speed: 1, gran: "layer", dom: null, stackBatch: null };

export function renderReplay(container) {
  const state = store.get();
  if (!state.allBatches) return clear(container, h("div", { class: "loading" }, "loading the recording…"));
  const key = `${state.runId}:${listGran()}`;
  if (R.key !== key) {
    const previous = R.events[R.i];
    R.events = buildEvents(state.allBatches, state.steps, listGran());
    R.i = previous ? Math.max(0, R.events.findIndex((e) => e.batch === previous.batch)) : 0;
    R.key = key;
    R.stackBatch = null;
  }
  R.stops = stopsFor(R.events, R.gran);
  const focus = state.replayFocus;
  if (focus) {
    state.replayFocus = null; // consumed (set directly: no re-render needed for this bookkeeping)
    const exact = focus.call != null ? R.events.findIndex((e) => e.batch === focus.batch && e.kind === focus.kind && e.call?.id === focus.call) : -1;
    const target = exact >= 0 ? exact : R.events.findIndex((e) => e.batch === focus.batch && e.kind === focus.kind);
    const fallback = R.events.findIndex((e) => e.batch === focus.batch);
    pause();
    R.i = Math.max(0, target >= 0 ? target : fallback);
  }
  if (!R.dom || !container.contains(R.dom.root)) buildSkeleton(container);
  R.dom.transport.setEvents(R.events);
  R.dom.transport.setGran(R.gran);
  update();
}

export function stopReplay() {
  pause();
}

/** The event Replay is showing (so the 3D view can open at the same moment). */
export const replayPosition = () => R.events[R.i] || null;

// ------------------------------------------------------------------------------------------
// events
// ------------------------------------------------------------------------------------------
export function layerCalls(batch, gran) {
  const idx = index(batch);
  const isLeafModule = (c) => !(idx.children.get(c.id) || []).some((k) => k.kind === "module");
  const attentionOwners = new Set((batch.ops || []).filter((o) => o.attention).map((o) => o.call));
  return batch.calls.filter(
    (c) =>
      c.phase === "forward" &&
      (c.kind === "function" || (c.kind === "module" && (gran === "op" || isLeafModule(c) || attentionOwners.has(c.id)))),
  );
}

/** The left list always shows every layer ("op": every call); only the step size changes. */
const listGran = () => (R.gran === "op" ? "op" : "layer");

/** Indices the transport stops at. Phase and batch stop at the *end* of each, so everything that
 * happened in it (thumbnails, gradients) is visible when you land. */
export function stopsFor(events, gran) {
  if (gran === "layer" || gran === "op") return events.map((_, i) => i);
  const stops = [];
  events.forEach((e, i) => {
    const next = events[i + 1];
    const boundary = !next || next.batch !== e.batch || (gran === "phase" && next.phase !== e.phase);
    if (boundary) stops.push(i);
  });
  return stops;
}

function nextStop(direction) {
  if (direction > 0) return R.stops.find((i) => i > R.i) ?? R.events.length - 1;
  return [...R.stops].reverse().find((i) => i < R.i) ?? 0;
}

export function buildEvents(batches, steps, gran) {
  const events = [];
  const lastOfStep = new Set(steps.map((s) => s.batches[s.batches.length - 1]));
  for (const batch of batches) {
    const base = { batch: batch.index, step: batch.step };
    const layers = layerCalls(batch, gran);
    const loss = [...batch.calls].reverse().find((c) => c.kind === "loss");
    const opt = batch.calls.find((c) => c.kind === "optimizer");
    events.push({ ...base, kind: "data", phase: "data", label: `load batch ${batch.index}` });
    for (const call of layers) events.push({ ...base, kind: "forward", phase: "forward", call, label: call.name });
    if (loss) events.push({ ...base, kind: "loss", phase: "loss", call: loss, label: loss.name });
    const back = layers.filter((c) => c.bwd_start_ms != null).sort((a, b) => a.bwd_start_ms - b.bwd_start_ms);
    for (const call of back) events.push({ ...base, kind: "backward", phase: "backward", call, label: `∇ ${call.name}` });
    if (lastOfStep.has(batch.index) || opt) events.push({ ...base, kind: "optimizer", phase: "optimizer", call: opt, label: opt ? `${opt.name}.step()` : "update" });
  }
  return events;
}

// ------------------------------------------------------------------------------------------
// playback
// ------------------------------------------------------------------------------------------
function go(i) {
  R.i = Math.max(0, Math.min(R.events.length - 1, i));
  update();
}

function play() {
  if (R.i >= R.events.length - 1) R.i = R.stops[0] ?? 0;
  R.playing = true;
  tick();
}

function tick() {
  clearTimeout(R.timer);
  if (!R.playing) return update();
  update();
  if (R.i >= R.events.length - 1) {
    R.playing = false;
    return update();
  }
  const event = R.events[R.i];
  const coarse = R.gran === "phase" || R.gran === "batch";
  const dwell = (coarse ? 2200 : event.kind === "data" || event.kind === "loss" || event.kind === "optimizer" ? 1500 : 650) / R.speed;
  R.timer = setTimeout(() => {
    R.i = nextStop(1);
    tick();
  }, dwell);
}

function pause() {
  R.playing = false;
  clearTimeout(R.timer);
}

function toggle() {
  R.playing ? pause() : play();
  update();
}

export function replayKey(event) {
  if (event.key === " ") {
    event.preventDefault();
    toggle();
  } else if (event.key === "ArrowRight" && event.shiftKey) go(phaseJump(R.events, R.i, 1));
  else if (event.key === "ArrowLeft" && event.shiftKey) go(phaseJump(R.events, R.i, -1));
  else if (event.key === "ArrowRight") go(nextStop(1));
  else if (event.key === "ArrowLeft") go(nextStop(-1));
  else if (event.key === "]") shiftSample(1);
  else if (event.key === "[") shiftSample(-1);
  else return false;
  return true;
}

function sampleCount() {
  return followableSamples(currentBatch()).limit;
}

function shiftSample(d) {
  const n = sampleCount();
  const s = ((store.get().sample ?? 0) + d + n) % n;
  store.set({ sample: s });
}

// ------------------------------------------------------------------------------------------
// rendering
// ------------------------------------------------------------------------------------------
function currentBatch() {
  const event = R.events[R.i];
  return event ? store.get().allBatches.find((b) => b.index === event.batch) : null;
}

function buildSkeleton(container) {
  const transport = buildTransport({
    granularities: GRANULARITIES,
    gran: R.gran,
    speed: R.speed,
    onGran: (key) => ((R.gran = key), pause(), renderReplay(container)),
    onSpeed: (x) => (R.speed = x),
    onRestart: () => (pause(), go(0)),
    onPrev: () => go(nextStop(-1)),
    onToggle: toggle,
    onNext: () => go(nextStop(1)),
    onSeek: (i) => (pause(), go(i)),
    onSample: shiftSample,
  });
  const stage = h("div", { class: "stage-view" });
  const detail = h("div", { class: "detail" });
  const root = h("div", { class: "replay" }, transport.root, h("div", { class: "replay-body" }, stage, detail));
  clear(container, root);
  R.dom = { root, transport, stage, detail };
  R.stackBatch = null; // fresh DOM: the layer stack must be rebuilt into it
}

function update() {
  if (!R.dom || !R.events.length) return;
  const event = R.events[R.i];
  const batch = currentBatch();
  const requested = store.get().sample ?? 0;
  const sample = Math.min(requested, sampleCount() - 1);
  R.unstored = requested > sample ? requested : null;
  R.dom.transport.setPlaying(R.playing);
  R.dom.transport.setPosition(R.i);
  R.dom.transport.setSample({ sample, ...followableSamples(batch) });
  renderStage(batch, event, sample);
  renderDetail(batch, event, sample);
}

function renderStage(batch, event, sample) {
  const key = `${batch.index}:${sample}:${R.unstored}`;
  const layers = layerCalls(batch, listGran());
  if (R.stackBatch !== key) {
    R.stackBatch = key;
    R.rows = new Map();
    const rows = layers.map((call) => {
      const out = firstOutput(call);
      const meta = out ? batch.tensors[out.tid] : null;
      const canvas = h("canvas", {});
      const bar = h("i", { style: { width: "0%" } });
      const backward = () => R.events.findIndex((e) => e.batch === batch.index && e.kind === "backward" && e.call?.id === call.id);
      const shape = meta ? (meta.batch_dim?.block === 1 ? meta.shape.slice(1) : meta.shape) : null;
      const row = h(
        "div",
        { class: "lrow pending", onclick: () => (pause(), go(R.events.findIndex((e) => e.batch === batch.index && e.call?.id === call.id))) },
        h("span", { class: "st" }),
        h("span", { class: "nm" }, call.name, h("small", {}, call.kind === "function" ? "function" : call.cls || "")),
        h("span", { class: "shp" }, shape ? fmtShape(shape) : ""),
        canvas,
        h(
          "span",
          { class: "gbar", title: "‖dL/d(output)‖, log scale. Click to see this layer's gradient", onclick: (e) => (e.stopPropagation(), pause(), backward() >= 0 && go(backward())) },
          bar,
        ),
      );
      R.rows.set(call.id, { row, canvas, bar, out, meta, shown: null });
      return row;
    });
    const raw = h("div", { class: "io-card" });
    const result = h("div", { class: "io-card" });
    R.ioRaw = raw;
    R.ioResult = result;
    clear(
      R.dom.stage,
      R.unstored != null
        ? h(
            "div",
            { class: "notice" },
            `Sample #${R.unstored} can't be followed through the layers: only the first ${sampleCount()} samples' activations were stored. Showing sample #${sample}. Re-record with a larger --sample-rows to follow it.`,
          )
        : null,
      h("div", { class: "caption" }, `Sample #${sample} of batch ${batch.index}, through every layer. Thumbnails show each layer's output; the bar on the right fills with its gradient on the way back.`),
      raw,
      h("div", { class: "layers" }, rows),
      result,
    );
    renderInput(raw, batch, sample);
  }
  const fwdIndex = layers.findIndex((c) => c.id === event.call?.id);
  const phaseOrder = ["data", "forward", "loss", "backward", "optimizer"];
  const phaseRank = phaseOrder.indexOf(event.phase);
  const bwdDone = new Set();
  if (event.phase === "backward" || event.phase === "optimizer") {
    for (let i = R.i; i >= 0 && R.events[i].batch === batch.index; i--) {
      if (R.events[i].kind === "backward") bwdDone.add(R.events[i].call.id);
    }
    if (event.phase === "optimizer") layers.forEach((c) => c.bwd_start_ms != null && bwdDone.add(c.id));
  }
  const gradNorms = layers.map((c) => R.rows.get(c.id).meta?.grad_stats?.norm || 0).filter((v) => v > 0);
  const maxLog = Math.log10(Math.max(...gradNorms, 1e-12));
  const minLog = Math.log10(Math.min(...gradNorms, 1)) - 1;
  layers.forEach((call, i) => {
    const r = R.rows.get(call.id);
    let fwdDone;
    if (event.kind === "forward") fwdDone = i <= fwdIndex;
    else fwdDone = phaseRank >= phaseOrder.indexOf("forward") && event.kind !== "data";
    const isCur = event.call?.id === call.id;
    r.row.classList.toggle("pending", !fwdDone);
    r.row.classList.toggle("done", fwdDone && !bwdDone.has(call.id));
    r.row.classList.toggle("bdone", bwdDone.has(call.id));
    r.row.classList.toggle("cur", isCur);
    r.row.classList.toggle("bwd", isCur && event.kind === "backward");
    const norm = r.meta?.grad_stats?.norm;
    r.bar.style.width = bwdDone.has(call.id) && norm ? `${Math.max(4, ((Math.log10(norm) - minLog) / (maxLog - minLog || 1)) * 100)}%` : "0%";
    const want = bwdDone.has(call.id) && r.meta?.grad_stored ? "grad" : fwdDone ? "value" : null;
    if (want && r.shown !== want && r.out && r.meta?.stored !== "none") {
      r.shown = want;
      api("thumb", { run: store.get().runId, batch: r.out.batch, tid: r.out.tid, sample, grad: want === "grad", size: 40 })
        .then((view) => drawHeat(r.canvas, view))
        .catch(() => {});
    }
    if (isCur) r.row.scrollIntoView({ block: "nearest" });
  });
  renderResult(R.ioResult, batch, sample, phaseRank >= phaseOrder.indexOf("loss"));
}

function renderInput(card, batch, sample) {
  const s = batch.samples?.[sample];
  const raw = s?.raw?.value;
  const canvas = h("canvas", {});
  if (raw?.kind === "text") {
    clear(card, h("div", {}, h("div", { class: "faint" }, s.index != null ? `dataset index ${s.index}` : "raw text"), h("div", { class: "txt" }, raw.text)));
    return;
  }
  clear(card, canvas, h("div", {}, h("div", { class: "faint" }, s?.index != null ? `dataset index ${s.index}` : "input"), s?.target ? h("div", {}, "target: ", h("b", {}, s.target.name ?? s.target.text ?? s.target.id)) : null));
  const ref = raw?.kind === "image" ? raw : s?.input;
  if (!ref) return canvas.remove();
  const params = { run: store.get().runId, batch: ref.batch, tid: ref.tid };
  if (ref === s.input) params.sample = sample;
  api("image", params).then((img) => drawImage(canvas, img)).catch(() => canvas.remove());
}

function renderResult(card, batch, sample, revealed) {
  const s = batch.samples?.[sample];
  if (!revealed || !s) return clear(card, h("span", { class: "faint" }, "prediction and loss appear after the forward pass"));
  const p = s.prediction || {};
  const ok = p.correct;
  clear(
    card,
    h(
      "div",
      { style: { flex: 1 } },
      h("div", {}, "prediction ", h("b", { style: { color: ok === false ? "var(--bad)" : ok ? "var(--good)" : "inherit" } }, p.name ?? p.text ?? p.id ?? "—"), p.p != null ? h("span", { class: "faint" }, `  p=${fmtNum(p.p, 3)}`) : null),
      h("div", { class: "faint" }, "target ", s.target?.name ?? s.target?.text ?? s.target?.id ?? "—"),
    ),
    h("div", { style: { textAlign: "right" } }, h("div", { class: "faint" }, "loss, this sample"), h("div", { class: "big" }, fmtNum(s.loss, 3)), h("div", { class: "faint" }, `batch mean ${fmtNum(lossValue(batch), 3)}`)),
  );
}

function renderDetail(batch, event, sample) {
  const state = store.get();
  const el = R.dom.detail;
  const explore = (callId) =>
    h("a", { class: "link", onclick: () => (pause(), store.set({ page: "explore", mode: "batch", batchIndex: batch.index, selection: { call: callId } })) }, "open in Trace");
  if (event.kind === "data") {
    const s = batch.samples?.[sample];
    clear(
      el,
      h("h2", {}, `Batch ${batch.index} loaded`),
      h("div", { class: "sub" }, `${batch.batch_size ?? "?"} samples${batch.indices ? ` · indices ${batch.indices.slice(0, 8).join(", ")}${batch.indices.length > 8 ? "…" : ""}` : ""} · data ${fmtMs(batch.timing?.phases_ms?.data)}`),
      s ? h("div", {}, h("div", { class: "subhead" }, `sample #${sample} through the pipeline`), samplePipeline(batch, s, state)) : null,
    );
    return;
  }
  if (event.kind === "forward" || event.kind === "backward") {
    const call = event.call;
    const out = firstOutput(call);
    const meta = out ? batch.tensors[out.tid] : null;
    const grad = event.kind === "backward";
    clear(
      el,
      h("h2", {}, grad ? `∇ ${call.name}` : call.name),
      h(
        "div",
        { class: "sub" },
        `${call.kind === "function" ? "function" : call.cls}${call.file ? ` · ${call.file}:${call.line}` : ""} · `,
        grad ? `backward ≈ ${fmtMs(call.bwd_ms)}` : `forward ${fmtMs(call.ms)}`,
        " · ",
        explore(call.id),
      ),
      h("div", { class: "lead" }, grad ? `Gradient of the loss with respect to this layer's output, sample #${sample}.` : `What this layer produced for sample #${sample}.`),
      meta ? layerView({ run: state.runId, batch, call, sample, grad, state }) : h("div", { class: "faint" }, "no tensor output"),
      call.inputs?.length ? h("div", { class: "subhead" }, "inputs") : null,
      (call.inputs || []).flatMap((e) => (e.value?.kind === "tensor" ? [tensorCard({ run: state.runId, ref: e.value, meta: batch.tensors[e.value.tid], name: e.name, sample, grad })] : [])),
    );
    return;
  }
  if (event.kind === "loss") {
    const max = Math.max(...(batch.samples || []).map((s) => s.loss ?? 0), 1e-9);
    clear(
      el,
      h("h2", {}, `Loss · ${event.call.name}`),
      h("div", { class: "sub" }, `batch loss ${fmtNum(lossValue(batch), 4)} · reduction ${event.call.extra?.reduction ?? "?"} · `, h("a", { class: "link", onclick: () => (pause(), store.set({ page: "loss", batchIndex: batch.index })) }, "open Loss")),
      h("div", { class: "subhead" }, "per-sample loss"),
      (batch.samples || []).map((s) =>
        h(
          "div",
          { class: "lossrow", style: { gridTemplateColumns: "44px 1fr 60px" }, title: "follow this sample", onclick: () => store.set({ sample: s.position }) },
          h("span", { class: s.position === sample ? "" : "faint" }, s.position === sample ? `● #${s.position}` : `#${s.position}`),
          h("span", { class: "bar" }, h("i", { style: { width: `${((s.loss ?? 0) / max) * 100}%`, background: s.prediction?.correct === false ? "var(--bad)" : "var(--k-loss)" } })),
          h("span", { class: "mono" }, fmtNum(s.loss, 3)),
        ),
      ),
    );
    return;
  }
  if (event.kind === "optimizer") {
    const step = state.steps[event.step];
    const params = [...(step?.params || [])].sort((a, b) => (b.update_ratio ?? 0) - (a.update_ratio ?? 0));
    clear(
      el,
      h("h2", {}, `Weights updated · step ${event.step}`),
      h("div", { class: "sub" }, `${event.call?.name ?? "optimizer"} · ${params.length} parameters · `, h("a", { class: "link", onclick: () => (pause(), store.set({ page: "weights" })) }, "open Weights")),
      h("div", { class: "subhead" }, "relative update ‖Δw‖/‖w‖ (healthy ≈ 1e-3)"),
      params.slice(0, 20).map((p) =>
        h("div", { class: "lossrow", style: { gridTemplateColumns: "1fr 80px" } }, h("span", { class: "mono" }, p.name), h("span", { class: "mono" }, p.update_ratio != null ? p.update_ratio.toExponential(1) : "—")),
      ),
    );
    return;
  }
  clear(
    el,
    h("h2", {}, event.label),
    h("div", { class: "sub" }, `batch ${batch.index} · ${fmtMs(batch.timing?.total_ms)}`),
    h("div", { class: "faint" }, "Pick a layer in the stack to see its output."),
  );
}
