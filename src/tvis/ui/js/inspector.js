// Right panel: details of the selected call or parameter.

import { index, tensorRefs } from "./model.js";
import { store } from "./store.js";
import { tensorCard } from "./tensorview.js";
import { badge, clear, fmtMs, fmtNum, fmtShape, h } from "./util.js";

export function renderInspector(title, body) {
  const state = store.get();
  const sel = state.selection;
  if (sel?.param != null) return renderParam(title, body, sel.param);
  const batch = state.batch;
  const call = sel?.call != null && batch ? index(batch).byId.get(sel.call) : null;
  if (!call) {
    clear(title, h("h2", {}, "Inspector"));
    clear(
      body,
      h(
        "div",
        { class: "empty" },
        "Select a call in the tree, a sample, a line in the source, or a row in Layers.",
        h("div", { class: "faint", style: { marginTop: "8px" } }, "j / k: next / previous batch · Esc: stop following a sample"),
      ),
    );
    return;
  }
  clear(title, badge(call.kind), h("span", { class: "name" }, call.name));
  const kv = h("dl", { class: "kv" });
  const row = (k, v) => v != null && v !== "" && kv.append(h("dt", {}, k), h("dd", {}, v));
  row("kind", call.kind);
  row("class", call.cls);
  row("module", call.module_path || null);
  row("function", call.qualname && call.qualname !== call.name ? call.qualname : null);
  row("defined", call.file ? sourceLink(call.file, call.line) : null);
  row("called at", call.call_site ? sourceLink(call.call_site.file, call.call_site.line) : null);
  row("phase", call.phase);
  row("forward", call.ms != null ? fmtMs(call.ms) : null);
  row("backward", call.bwd_ms != null ? `≈ ${fmtMs(call.bwd_ms)}` : null);
  row("wall", call.wall_ms != null ? `${fmtMs(call.wall_ms)} (incl. capture)` : null);
  const parts = [kv];
  const flags = [];
  if (call.flags?.includes("mixes_samples")) flags.push(h("span", { class: "flag" }, "mixes samples (batch statistics)"));
  if (call.truncated) flags.push(h("span", { class: "flag" }, "spans a batch boundary (truncated)"));
  if (call.bwd_approx) flags.push(h("span", { class: "flag", style: { borderColor: "var(--border-strong)", color: "var(--text-muted)" } }, "backward time inferred from gradient arrival"));
  if (flags.length) parts.push(h("div", { style: { marginBottom: "10px" } }, flags));
  if (call.kind === "tokenize") parts.push(tokensBlock(call, state.sample));
  if (call.kind === "optimizer") parts.push(hyperBlock(call));
  if (call.kind === "loss" && call.extra?.per_sample_loss) parts.push(perSampleLoss(batch));
  parts.push(valuesBlock("Inputs", call.inputs, batch, state));
  parts.push(valuesBlock("Outputs", call.outputs, batch, state));
  clear(body, h("div", { class: "insp" }, parts));
}

function sourceLink(file, line) {
  return h("a", { class: "link mono", onclick: () => store.set({ tab: "source", sourceFile: file, sourceLine: line }) }, `${file}:${line}`);
}

function valuesBlock(label, entries, batch, state) {
  if (!entries?.length) return null;
  return h("div", {}, h("div", { class: "subhead" }, label), entries.map((e) => valueView(e.name, e.value, batch, state)));
}

function valueView(name, value, batch, state) {
  if (!value) return null;
  switch (value.kind) {
    case "tensor":
    case "image":
      return tensorCard({ run: state.runId, ref: value, meta: batch.tensors[value.tid], name, sample: state.sample, open: false });
    case "text":
      return h("div", {}, h("div", { class: "faint", style: { fontSize: "11px" } }, name), h("div", { class: "value-text" }, value.text, value.truncated ? "…" : ""));
    case "scalar":
      return h("div", { class: "value-text" }, h("span", { class: "faint" }, `${name} = `), String(value.value ?? value.repr));
    case "seq":
    case "map": {
      const items = value.kind === "seq" ? value.items.map((v, i) => [`${name}[${i}]`, v]) : Object.entries(value.items).map(([k, v]) => [`${name}.${k}`, v]);
      if (state.sample != null && value.kind === "seq" && items.length > state.sample && items.every(([, v]) => v.kind === "text")) {
        return valueView(`${name}[${state.sample}]`, items[state.sample][1], batch, state);
      }
      const more = value.len > items.length ? h("div", { class: "faint" }, `… ${value.len - items.length} more`) : null;
      return h("div", {}, items.map(([n, v]) => valueView(n, v, batch, state)), more);
    }
    case "error":
      return h("div", { class: "err" }, `${name}: capture failed (${value.message})`);
    default:
      return h("div", { class: "value-text" }, h("span", { class: "faint" }, `${name}: ${value.type} `), value.repr || "");
  }
}

function tokensBlock(call, sample) {
  const rows = call.extra?.tokens || [];
  const shown = sample != null && rows[sample] ? [[sample, rows[sample]]] : rows.slice(0, 8).map((r, i) => [i, r]);
  return h(
    "div",
    {},
    h("div", { class: "subhead" }, `Tokens · ${call.extra?.name_or_path || "tokenizer"}`),
    shown.map(([i, tokens]) =>
      h("div", { style: { marginBottom: "6px" } }, h("div", { class: "faint", style: { fontSize: "11px" } }, `sample ${i}`), h("div", { class: "tokens" }, tokens.map((t) => h("span", { class: `tok ${/^\[?PAD\]?$|^<pad>$/i.test(t) ? "pad" : ""}` }, t)))),
    ),
  );
}

function hyperBlock(call) {
  const groups = call.extra?.hyper || [];
  return h(
    "div",
    {},
    h("div", { class: "subhead" }, "Hyper-parameters"),
    groups.map((g, i) => {
      const kv = h("dl", { class: "kv" });
      kv.append(h("dt", {}, "group"), h("dd", {}, String(i)));
      for (const [k, v] of Object.entries(g)) kv.append(h("dt", {}, k), h("dd", { class: "mono" }, Array.isArray(v) ? `(${v.join(", ")})` : String(v)));
      return kv;
    }),
  );
}

function perSampleLoss(batch) {
  const max = Math.max(...batch.samples.map((s) => s.loss ?? 0), 1e-9);
  return h(
    "div",
    {},
    h("div", { class: "subhead" }, "Per-sample loss"),
    batch.samples.map((s) =>
      h(
        "div",
        { class: "row", style: { gridTemplateColumns: "60px 1fr 70px", paddingLeft: "0" }, onclick: () => store.set({ sample: s.position }) },
        h("span", { class: "faint" }, `#${s.position}`),
        h("span", { class: "conf" }, h("i", { style: { width: `${((s.loss ?? 0) / max) * 100}%`, background: "var(--k-loss)" } })),
        h("span", { class: "mono", style: { textAlign: "right" } }, fmtNum(s.loss)),
      ),
    ),
  );
}

function renderParam(title, body, name) {
  const state = store.get();
  const step = state.steps[state.stepIndex];
  const p = step?.params.find((x) => x.name === name);
  if (!p) return clear(body, h("div", { class: "empty" }, "Parameter not found."));
  clear(title, badge("optimizer"), h("span", { class: "name" }, p.name));
  const batchDoc = state.stepBatches.find((b) => b.index === p.weight?.batch);
  const kv = h("dl", { class: "kv" });
  for (const [k, v] of [
    ["shape", fmtShape(p.shape)],
    ["‖w‖", fmtNum(p.weight_norm)],
    ["‖grad‖", fmtNum(p.grad_norm)],
    ["‖Δw‖", fmtNum(p.update_norm)],
    ["‖Δw‖ / ‖w‖", p.update_ratio != null ? p.update_ratio.toExponential(2) : "undefined (‖w‖ = 0)"],
  ])
    kv.append(h("dt", {}, k), h("dd", {}, v));
  const cards = batchDoc
    ? [["weight (before step)", p.weight], ["gradient", p.grad], ["update Δw", p.update]]
        .filter(([, ref]) => ref)
        .map(([label, ref]) => tensorCard({ run: state.runId, ref, meta: batchDoc.tensors[ref.tid], name: label }))
    : [h("div", { class: "loading" }, "loading…")];
  clear(body, h("div", { class: "insp" }, kv, h("div", { class: "subhead" }, "Tensors"), cards));
}

export function refsOf(call) {
  return [...(call.inputs || []), ...(call.outputs || [])].flatMap((e) => tensorRefs(e.value));
}
