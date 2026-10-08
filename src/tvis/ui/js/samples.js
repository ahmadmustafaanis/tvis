// Samples tab: every datapoint of the batch, and the pipeline of the followed sample.

import { api } from "./api.js";
import { index, tensorRefs } from "./model.js";
import { store } from "./store.js";
import { drawImage } from "./tensorview.js";
import { clear, fmtNum, fmtShape, h } from "./util.js";

let sortBy = "position";

export function renderSamples(container) {
  const state = store.get();
  const batch = state.batch;
  if (!batch) return clear(container, h("div", { class: "loading" }, "loading…"));
  if (!batch.samples?.length) return clear(container, h("div", { class: "empty" }, "No per-sample information for this batch."));
  const isText = batch.samples.some((s) => s.raw?.value?.kind === "text" || s.target?.text != null);
  const samples = [...batch.samples].sort(sorter());
  const toolbar = h(
    "div",
    { class: "toolbar" },
    h("span", { class: "muted" }, `${batch.samples.length} samples`),
    h("div", { class: "spacer", style: { flex: 1 } }),
    h("span", { class: "faint" }, "sort"),
    h(
      "div",
      { class: "seg" },
      [["position", "#"], ["loss", "loss ↓"], ["wrong", "wrong first"]].map(([key, label]) =>
        h("button", { class: sortBy === key ? "on" : "", onclick: () => ((sortBy = key), renderSamples(container)) }, label),
      ),
    ),
  );
  const followed = state.sample != null ? batch.samples[state.sample] : null;
  clear(
    container,
    h(
      "div",
      { class: "view" },
      followed ? h("section", {}, h("h3", {}, `Sample #${followed.position}`, followed.index != null ? h("span", { class: "faint" }, ` · dataset index ${followed.index}`) : null), samplePipeline(batch, followed, state)) : null,
      h("section", {}, toolbar, h("div", { class: `samples ${isText ? "text" : ""}` }, samples.map((s) => (isText ? textCard(s, state) : imageCard(batch, s, state))))),
    ),
  );
}

function sorter() {
  if (sortBy === "loss") return (a, b) => (b.loss ?? -1) - (a.loss ?? -1);
  if (sortBy === "wrong") return (a, b) => Number(a.prediction?.correct ?? 1) - Number(b.prediction?.correct ?? 1) || (b.loss ?? 0) - (a.loss ?? 0);
  return (a, b) => a.position - b.position;
}

function follow(position) {
  const state = store.get();
  store.set({ sample: state.sample === position ? null : position });
}

function imageCard(batch, sample, state) {
  const canvas = h("canvas", { class: "thumb" });
  thumbnail(canvas, batch, sample, state.runId);
  const p = sample.prediction;
  return h(
    "div",
    { class: `sample ${state.sample === sample.position ? "on" : ""}`, onclick: () => follow(sample.position) },
    canvas,
    h("div", { class: "meta" }, h("span", { class: "muted" }, `#${sample.position}${sample.index != null ? ` · idx ${sample.index}` : ""}`), sample.loss != null ? h("span", { class: "mono" }, fmtNum(sample.loss, 3)) : null),
    sample.target ? h("div", { class: "meta" }, h("span", { class: "faint" }, "target"), h("span", {}, sample.target.name ?? sample.target.id)) : null,
    p ? h("div", { class: "meta" }, h("span", { class: `verdict ${p.correct ? "ok" : "no"}` }, p.correct ? "✓" : "✗", " ", p.name ?? p.id), h("span", { class: "mono faint" }, `${(p.p * 100).toFixed(0)}%`)) : null,
    p ? h("div", { class: "conf" }, h("i", { style: { width: `${p.p * 100}%`, background: p.correct ? "var(--good)" : "var(--bad)" } })) : null,
  );
}

function textCard(sample, state) {
  const p = sample.prediction;
  const raw = sample.raw?.value?.text;
  return h(
    "div",
    { class: `sample ${state.sample === sample.position ? "on" : ""}`, onclick: () => follow(sample.position) },
    h("div", { class: "meta" }, h("span", { class: "muted" }, `#${sample.position}${sample.index != null ? ` · idx ${sample.index}` : ""}`), h("span", { class: "mono" }, sample.loss != null ? `loss ${fmtNum(sample.loss, 3)}` : "")),
    raw != null ? h("div", { class: "textline" }, h("span", { class: "lbl" }, "input"), raw) : null,
    sample.target?.text != null ? h("div", { class: "textline" }, h("span", { class: "lbl" }, "target"), sample.target.text) : null,
    p?.text != null ? h("div", { class: "textline" }, h("span", { class: "lbl" }, "predict"), tokenDiff(p, sample.target)) : null,
    p?.token_accuracy != null ? h("div", { class: "conf" }, h("i", { style: { width: `${p.token_accuracy * 100}%` } })) : null,
  );
}

function tokenDiff(prediction, target) {
  if (!prediction.ids || !target?.ids) return prediction.text;
  const words = prediction.text.split(/\s+/);
  if (words.length !== prediction.ids.length) return prediction.text;
  return words.map((w, i) => h("span", { style: { color: prediction.ids[i] === target.ids[i] ? "var(--good)" : "var(--bad)" } }, `${w} `));
}

async function thumbnail(canvas, batch, sample, run) {
  const ref = sample.raw?.value?.kind === "image" ? sample.raw.value : sample.input;
  if (!ref) return;
  const params = { run, batch: ref.batch, tid: ref.tid };
  if (ref === sample.input) params.sample = sample.position;
  try {
    drawImage(canvas, await api("image", params));
  } catch {
    canvas.replaceWith(h("div", { class: "faint" }, "no preview"));
  }
}

export function samplePipeline(batch, sample, state) {
  const idx = index(batch);
  const stages = [];
  if (sample.sample_call != null) {
    const walk = (id) => {
      for (const call of idx.children.get(id) || []) {
        if (call.kind === "pipeline" || call.kind === "tokenize") {
          if (!stages.length && call.inputs[0]) stages.push({ title: "raw", value: call.inputs[0].value, call });
          if (call.kind === "tokenize") stages.push({ title: "tokenizer", value: call.outputs[0]?.value, call, tokens: call.extra?.tokens?.[0] });
        }
        if (call.kind === "transform") stages.push({ title: call.name, value: call.outputs[0]?.value, call });
        walk(call.id);
      }
    };
    walk(sample.sample_call);
    if (!stages.length) {
      const item = idx.byId.get(sample.sample_call)?.outputs[0]?.value;
      const first = item?.kind === "seq" ? item.items[0] : item;
      if (first) stages.push({ title: "dataset item", value: first, call: idx.byId.get(sample.sample_call) });
    }
  }
  if (sample.tokens) stages.push({ title: "tokens", tokens: sample.tokens.tokens, call: idx.byId.get(sample.tokens.call) });
  if (sample.input) stages.push({ title: "model input", value: sample.input, sampleSlice: true });
  const root = batch.calls.find((c) => c.kind === "module" && c.extra?.root);
  if (root) stages.push({ title: `${root.name} output`, value: tensorRefs(root.outputs[0]?.value)[0], call: root, sampleSlice: true });
  if (sample.prediction) stages.push({ title: "prediction", prediction: sample.prediction, target: sample.target });

  const cards = [];
  stages.forEach((stage, i) => {
    if (i) cards.push(h("span", { class: "arrow" }, "→"));
    cards.push(stageCard(batch, stage, sample, state));
  });
  return h("div", { class: "pipeline" }, cards);
}

function stageCard(batch, stage, sample, state) {
  const card = h("div", { class: `stage ${stage.call && state.selection?.call === stage.call.id ? "sel" : ""}`, onclick: () => stage.call && store.set({ selection: { call: stage.call.id } }) });
  const add = (...nodes) => card.append(...nodes.filter((n) => n != null && n !== false));
  add(h("div", { class: "title" }, stage.title));
  const value = stage.value;
  if (stage.prediction) {
    const p = stage.prediction;
    add(
      h("div", { class: p.correct === false ? "verdict no" : "verdict ok", style: { fontWeight: 600 } }, p.name ?? p.text ?? p.id),
      p.p != null ? h("div", { class: "sub" }, `p = ${fmtNum(p.p, 3)}`) : null,
      p.token_accuracy != null ? h("div", { class: "sub" }, `token acc ${(p.token_accuracy * 100).toFixed(0)}%`) : null,
      stage.target ? h("div", { class: "sub" }, `target: ${stage.target.name ?? stage.target.text ?? stage.target.id}`) : null,
    );
    return card;
  }
  if (stage.tokens) {
    card.style.width = "220px";
    add(h("div", { class: "tokens" }, stage.tokens.slice(0, 40).map((t) => h("span", { class: `tok ${/PAD/i.test(t) ? "pad" : ""}` }, t))));
    return card;
  }
  if (!value) return card;
  if (value.kind === "text") {
    add(h("div", { class: "textline" }, value.text));
    return card;
  }
  const meta = batch.tensors[value.tid];
  if (!meta) return card;
  const shape = stage.sampleSlice && meta.batch_dim ? (meta.batch_dim.block === 1 ? meta.shape.slice(1) : [meta.batch_dim.block, ...meta.shape.slice(1)]) : meta.shape;
  const isImg = meta.kind === "image" || (shape.length === 3 && (shape[0] === 3 || shape[0] === 1) && shape[1] > 1);
  if (isImg) {
    const canvas = h("canvas", {});
    add(canvas);
    const params = { run: state.runId, batch: value.batch, tid: value.tid };
    if (stage.sampleSlice && meta.batch_dim) params.sample = sample.position;
    api("image", params).then((img) => drawImage(canvas, img)).catch(() => canvas.remove());
  }
  add(h("div", { class: "sub" }, `${fmtShape(shape)} ${meta.dtype.split(":")[0]}`));
  if (meta.stats?.mean != null && meta.kind !== "image") add(h("div", { class: "sub" }, `μ ${fmtNum(meta.stats.mean, 3)} σ ${fmtNum(meta.stats.std, 3)}`));
  return card;
}
