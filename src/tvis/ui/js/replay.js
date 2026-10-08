// Replay page: play the recorded steps back as if training were happening — data → each layer
// forward → loss → gradients flowing back → optimizer update → next batch.

import { api } from "./api.js";
import { firstOutput, followableSamples, index, lossValue } from "./model.js";
import { samplePipeline } from "./samples.js";
import { store } from "./store.js";
import { drawHeat, drawImage, tensorCard } from "./tensorview.js";
import { clear, fmtMs, fmtNum, fmtShape, h, phaseColor } from "./util.js";

const GRANULARITIES = [
  ["layer", "Layer"],
  ["op", "Every call"],
  ["phase", "Phase"],
  ["batch", "Batch"],
];
const SPEEDS = [0.5, 1, 2, 4];
const PHASE_NAME = { data: "load data", forward: "forward", loss: "loss", backward: "backward", optimizer: "optimizer" };

const R = { key: null, events: [], i: 0, playing: false, timer: null, speed: 1, gran: "layer", dom: null, stackBatch: null };

export function renderReplay(container) {
  const state = store.get();
  if (!state.allBatches) return clear(container, h("div", { class: "loading" }, "loading the recording…"));
  const key = `${state.runId}:${R.gran}`;
  if (R.key !== key) {
    const previous = R.events[R.i];
    R.events = buildEvents(state.allBatches, state.steps, R.gran);
    R.i = previous ? Math.max(0, R.events.findIndex((e) => e.batch === previous.batch)) : 0;
    R.key = key;
    R.stackBatch = null;
  }
  if (!R.dom || !container.contains(R.dom.root)) buildSkeleton(container);
  renderEventList();
  update();
}

export function stopReplay() {
  pause();
}

// ------------------------------------------------------------------------------------------
// events
// ------------------------------------------------------------------------------------------
function layerCalls(batch, gran) {
  const idx = index(batch);
  const isLeafModule = (c) => !(idx.children.get(c.id) || []).some((k) => k.kind === "module");
  return batch.calls.filter(
    (c) => c.phase === "forward" && (c.kind === "function" || (c.kind === "module" && (gran === "op" || isLeafModule(c)))),
  );
}

function buildEvents(batches, steps, gran) {
  const events = [];
  const lastOfStep = new Set(steps.map((s) => s.batches[s.batches.length - 1]));
  for (const batch of batches) {
    const base = { batch: batch.index, step: batch.step };
    const layers = layerCalls(batch, gran);
    const loss = [...batch.calls].reverse().find((c) => c.kind === "loss");
    const opt = batch.calls.find((c) => c.kind === "optimizer");
    if (gran === "batch") {
      events.push({ ...base, kind: "batch", phase: "forward", label: `batch ${batch.index}` });
      continue;
    }
    events.push({ ...base, kind: "data", phase: "data", label: `load batch ${batch.index}` });
    if (gran === "phase") events.push({ ...base, kind: "forward-all", phase: "forward", label: "forward pass" });
    else for (const call of layers) events.push({ ...base, kind: "forward", phase: "forward", call, label: call.name });
    if (loss) events.push({ ...base, kind: "loss", phase: "loss", call: loss, label: loss.name });
    if (gran === "phase") events.push({ ...base, kind: "backward-all", phase: "backward", label: "backward pass" });
    else {
      const back = layers.filter((c) => c.bwd_start_ms != null).sort((a, b) => a.bwd_start_ms - b.bwd_start_ms);
      for (const call of back) events.push({ ...base, kind: "backward", phase: "backward", call, label: `∇ ${call.name}` });
    }
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
  if (R.i >= R.events.length - 1) R.i = 0;
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
  const dwell = (event.kind === "data" || event.kind === "loss" || event.kind === "optimizer" ? 1500 : 650) / R.speed;
  R.timer = setTimeout(() => {
    R.i += 1;
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

function jumpPhase(direction) {
  const current = R.events[R.i];
  let i = R.i + direction;
  while (i > 0 && i < R.events.length - 1 && R.events[i].phase === current.phase && R.events[i].batch === current.batch) i += direction;
  go(i);
}

export function replayKey(event) {
  if (event.key === " ") {
    event.preventDefault();
    toggle();
  } else if (event.key === "ArrowRight") go(R.i + 1);
  else if (event.key === "ArrowLeft") go(R.i - 1);
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
  const now = h("div", { class: "now" });
  const playBtn = h("button", { class: "tbtn play", title: "Play / pause (space)", onclick: toggle }, "▶");
  const scrub = h("div", { class: "scrub" });
  scrub.addEventListener("click", (e) => {
    const rect = scrub.getBoundingClientRect();
    go(Math.round(((e.clientX - rect.left) / rect.width) * (R.events.length - 1)));
  });
  const gran = h(
    "div",
    { class: "seg" },
    GRANULARITIES.map(([key, label]) =>
      h("button", { class: key === R.gran ? "on" : "", dataset: { gran: key }, onclick: () => ((R.gran = key), pause(), renderReplay(container)) }, label),
    ),
  );
  const speed = h(
    "select",
    { class: "select", onchange: (e) => (R.speed = Number(e.target.value)), title: "Speed" },
    SPEEDS.map((s) => h("option", { value: s, selected: s === R.speed }, `${s}×`)),
  );
  const sampleNav = h("span", { class: "sample-nav" });
  const events = h("div", { class: "events" });
  const stage = h("div", { class: "stage-view" });
  const detail = h("div", { class: "detail" });
  const root = h(
    "div",
    { class: "replay" },
    h(
      "div",
      { class: "transport" },
      h(
        "div",
        { class: "transport-row" },
        h("button", { class: "tbtn", title: "Restart", onclick: () => go(0) }, "⏮"),
        h("button", { class: "tbtn", title: "Previous phase", onclick: () => jumpPhase(-1) }, "⏪"),
        h("button", { class: "tbtn", title: "Previous (←)", onclick: () => go(R.i - 1) }, "◀"),
        playBtn,
        h("button", { class: "tbtn", title: "Next (→)", onclick: () => go(R.i + 1) }, "▶|"),
        h("button", { class: "tbtn", title: "Next phase", onclick: () => jumpPhase(1) }, "⏩"),
        now,
        h("div", { style: { flex: 1 } }),
        sampleNav,
        h("label", { class: "faint" }, "step by"),
        gran,
        speed,
      ),
      scrub,
    ),
    h("div", { class: "replay-body" }, events, stage, detail),
  );
  clear(container, root);
  R.dom = { root, now, playBtn, scrub, events, stage, detail, sampleNav, gran };
}

function renderEventList() {
  const rows = [];
  let lastBatch = null;
  R.events.forEach((event, i) => {
    if (event.batch !== lastBatch) {
      rows.push(h("div", { class: "grp" }, `batch ${event.batch} · step ${event.step}`));
      lastBatch = event.batch;
    }
    rows.push(
      h(
        "div",
        { class: "ev", dataset: { i }, onclick: () => (pause(), go(i)) },
        h("span", { class: "mark", style: { background: phaseColor(event.phase) } }),
        h("span", { class: "lbl" }, event.label),
        h("span", { class: "faint" }, event.call?.kind === "function" ? "ƒ" : ""),
      ),
    );
  });
  clear(R.dom.events, rows);
  // scrubber track: one segment per event, coloured by phase
  const track = h("div", { class: "track" }, R.events.map((e) => h("span", { style: { flex: 1, background: phaseColor(e.phase) } })));
  const ticks = [];
  R.events.forEach((e, i) => {
    if (i && e.batch !== R.events[i - 1].batch) ticks.push(h("span", { class: "tick", style: { left: `${(i / Math.max(1, R.events.length - 1)) * 100}%` } }));
  });
  R.dom.head = h("span", { class: "head" });
  clear(R.dom.scrub, track, ticks, R.dom.head);
  for (const button of R.dom.gran.querySelectorAll("button")) button.classList.toggle("on", button.dataset.gran === R.gran);
}

function update() {
  if (!R.dom || !R.events.length) return;
  const event = R.events[R.i];
  const batch = currentBatch();
  const sample = Math.min(store.get().sample ?? 0, sampleCount() - 1);
  R.dom.playBtn.textContent = R.playing ? "❚❚" : "▶";
  R.dom.head.style.left = `${(R.i / Math.max(1, R.events.length - 1)) * 100}%`;
  clear(
    R.dom.now,
    h("span", { class: "ph", style: { background: phaseColor(event.phase) } }, PHASE_NAME[event.phase] || event.phase),
    h("b", {}, event.label),
    h("span", { class: "faint" }, `  ·  batch ${event.batch} · step ${event.step} · ${R.i + 1}/${R.events.length}`),
  );
  clear(
    R.dom.sampleNav,
    h("span", { class: "faint" }, "sample"),
    h("button", { onclick: () => shiftSample(-1), title: "Previous sample ([)" }, "‹"),
    h("b", {}, `#${sample}`),
    h("button", { onclick: () => shiftSample(1), title: "Next sample (])" }, "›"),
    (() => {
      const { total, limit } = followableSamples(batch);
      return limit < total ? h("span", { class: "faint", title: "Larger tensors keep only their first rows (raise --max-elems to store more)" }, `of ${limit} stored / ${total}`) : null;
    })(),
  );
  for (const row of R.dom.events.querySelectorAll(".ev")) {
    const i = Number(row.dataset.i);
    row.classList.toggle("cur", i === R.i);
    row.classList.toggle("done", i < R.i);
  }
  R.dom.events.querySelector(".ev.cur")?.scrollIntoView({ block: "nearest" });
  renderStage(batch, event, sample);
  renderDetail(batch, event, sample);
}

function renderStage(batch, event, sample) {
  const key = `${batch.index}:${sample}`;
  const layers = layerCalls(batch, R.gran === "op" ? "op" : "layer");
  if (R.stackBatch !== key) {
    R.stackBatch = key;
    R.rows = new Map();
    const rows = layers.map((call) => {
      const out = firstOutput(call);
      const meta = out ? batch.tensors[out.tid] : null;
      const canvas = h("canvas", {});
      const bar = h("i", { style: { width: "0%" } });
      const shape = meta ? (meta.batch_dim?.block === 1 ? meta.shape.slice(1) : meta.shape) : null;
      const row = h(
        "div",
        { class: "lrow pending", onclick: () => (pause(), go(R.events.findIndex((e) => e.batch === batch.index && e.call?.id === call.id))) },
        h("span", { class: "st" }),
        h("span", { class: "nm" }, call.name, h("small", {}, call.kind === "function" ? "ƒ" : call.cls || "")),
        h("span", { class: "shp" }, shape ? fmtShape(shape) : ""),
        canvas,
        h("span", { class: "gbar", title: "‖dL/d(output)‖" }, bar),
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
      h("div", { class: "faint", style: { marginBottom: "6px", fontSize: "12px" } }, `input · sample #${sample} of batch ${batch.index}`),
      raw,
      h("div", { class: "flow" }, "↓"),
      h("div", { class: "layers" }, rows),
      h("div", { class: "flow" }, "↓"),
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
    if (event.kind === "backward-all" || event.phase === "optimizer") layers.forEach((c) => c.bwd_start_ms != null && bwdDone.add(c.id));
  }
  const gradNorms = layers.map((c) => R.rows.get(c.id).meta?.grad_stats?.norm || 0).filter((v) => v > 0);
  const maxLog = Math.log10(Math.max(...gradNorms, 1e-12));
  const minLog = Math.log10(Math.min(...gradNorms, 1)) - 1;
  layers.forEach((call, i) => {
    const r = R.rows.get(call.id);
    let fwdDone;
    if (event.kind === "forward") fwdDone = i <= fwdIndex;
    else if (event.kind === "batch") fwdDone = true;
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
  renderResult(R.ioResult, batch, sample, phaseRank >= phaseOrder.indexOf("loss") || event.kind === "batch");
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
    h("div", { style: { textAlign: "right" } }, h("div", { class: "faint" }, "loss (this sample)"), h("div", { style: { fontSize: "18px", fontWeight: 600 } }, fmtNum(s.loss, 3)), h("div", { class: "faint" }, `batch ${fmtNum(lossValue(batch), 3)}`)),
  );
}

function renderDetail(batch, event, sample) {
  const state = store.get();
  const el = R.dom.detail;
  const explore = (callId) =>
    h("a", { class: "link", onclick: () => (pause(), store.set({ page: "explore", mode: "batch", batchIndex: batch.index, selection: { call: callId } })) }, "open in Explore →");
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
      grad
        ? h("div", { class: "faint", style: { marginBottom: "8px" } }, "Gradient of the loss with respect to this layer's output, for the followed sample.")
        : h("div", { class: "faint", style: { marginBottom: "8px" } }, "What this layer produced for the followed sample."),
      meta ? tensorCard({ run: state.runId, ref: out, meta, name: grad ? "dL/d(output)" : "output", sample, grad, open: true }) : h("div", { class: "faint" }, "no tensor output"),
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
      h("div", { class: "sub" }, `batch loss ${fmtNum(lossValue(batch), 4)} · reduction ${event.call.extra?.reduction ?? "?"} · `, h("a", { class: "link", onclick: () => (pause(), store.set({ page: "loss", batchIndex: batch.index })) }, "open Loss page →")),
      h("div", { class: "subhead" }, "per-sample loss"),
      (batch.samples || []).map((s) =>
        h(
          "div",
          { class: "lossrow", style: { gridTemplateColumns: "44px 1fr 60px" }, onclick: () => store.set({ sample: s.position }) },
          h("span", { class: s.position === sample ? "" : "faint" }, `#${s.position}`),
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
      h("div", { class: "sub" }, `${event.call?.name ?? "optimizer"} · ${params.length} parameters · `, h("a", { class: "link", onclick: () => (pause(), store.set({ page: "weights" })) }, "open Weights →")),
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
    h("div", { class: "faint" }, event.kind === "backward-all" ? "Gradients flow from the loss back to the input; bars show ‖dL/d(output)‖ per layer." : "Every layer ran; thumbnails show the followed sample's activations."),
  );
}
