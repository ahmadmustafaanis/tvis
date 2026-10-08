// Source tab: the code snapshot as it ran, annotated with shapes, calls and times per line.

import { api } from "./api.js";
import { firstOutput, index } from "./model.js";
import { store } from "./store.js";
import { clear, fmtMs, fmtShape, h } from "./util.js";

const KEYWORDS = new Set("False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield self".split(" "));
const TOKEN = /(#.*$)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|\b(\d+(?:\.\d*)?(?:e[+-]?\d+)?)\b|([A-Za-z_]\w*)/g;

function highlight(line) {
  const out = [];
  let last = 0;
  let prev = "";
  for (const m of line.matchAll(TOKEN)) {
    if (m.index > last) out.push(line.slice(last, m.index));
    const [text, comment, str, num, word] = m;
    if (comment) out.push(h("span", { class: "tk-com" }, text));
    else if (str) out.push(h("span", { class: "tk-str" }, text));
    else if (num) out.push(h("span", { class: "tk-num" }, text));
    else if (word && KEYWORDS.has(word)) out.push(h("span", { class: "tk-kw" }, text));
    else if (word && (prev === "def" || prev === "class")) out.push(h("span", { class: "tk-fn" }, text));
    else out.push(text);
    if (word) prev = word;
    last = m.index + text.length;
  }
  if (last < line.length) out.push(line.slice(last));
  return out;
}

export async function renderSource(container) {
  const state = store.get();
  const { meta, batch } = state;
  const files = meta?.sources || [];
  if (!files.length) return clear(container, h("div", { class: "empty" }, "No source files were captured."));
  const file = files.includes(state.sourceFile) ? state.sourceFile : files[0];
  const list = h("div", { class: "files" }, files.map((f) => h("div", { class: f === file ? "on" : "", onclick: () => store.set({ sourceFile: f, sourceLine: null }) }, f)));
  const code = h("div", { class: "code" }, h("div", { class: "loading" }, "loading…"));
  clear(container, h("div", { class: "source" }, list, code));
  let text;
  try {
    text = (await api("source", { run: state.runId, path: file })).text;
  } catch (err) {
    return clear(code, h("div", { class: "err", style: { padding: "12px" } }, err.message));
  }
  if (store.get().batch !== batch) return; // a newer render is in flight
  const annotations = batch ? annotate(batch, file) : new Map();
  const selected = batch && state.selection?.call != null ? index(batch).byId.get(state.selection.call) : null;
  const highlightLines = new Set();
  if (selected?.file === file) highlightLines.add(selected.line);
  if (selected?.call_site?.file === file) highlightLines.add(selected.call_site.line);
  if (state.sourceFile === file && state.sourceLine) highlightLines.add(state.sourceLine);
  const lines = text.split("\n");
  const rows = lines.map((line, i) => {
    const n = i + 1;
    const ann = annotations.get(n);
    return h(
      "div",
      { class: `ln ${highlightLines.has(n) ? "hl" : ""} ${ann ? "exec" : ""}`, dataset: { line: n } },
      h("span", { class: "num" }, n),
      h("span", { class: "src" }, highlight(line)),
      h("span", { class: "ann" }, ann || null),
    );
  });
  clear(code, rows);
  const target = code.querySelector(".ln.hl");
  if (target) target.scrollIntoView({ block: "center" });
}

function annotate(batch, file) {
  const byLine = new Map();
  const add = (line, node) => {
    if (!byLine.has(line)) byLine.set(line, []);
    byLine.get(line).push(node);
  };
  // definitions: functions / module classes defined in this file
  const defs = new Map();
  for (const call of batch.calls) {
    if (call.file !== file || call.line == null) continue;
    const entry = defs.get(call.line) || { calls: [], ms: 0, bwd: 0 };
    entry.calls.push(call);
    entry.ms += call.ms || 0;
    entry.bwd += call.bwd_ms || 0;
    defs.set(call.line, entry);
  }
  for (const [line, entry] of defs) {
    const first = entry.calls[0];
    add(
      line,
      h(
        "span",
        { class: `a ${first.kind === "module" ? "mod" : "fn"}`, title: "click to inspect the first call", onclick: () => store.set({ selection: { call: first.id } }) },
        `${first.kind === "module" ? first.cls : "ƒ"} ×${entry.calls.length} · ${fmtMs(entry.ms)}${entry.bwd ? ` · bwd ≈${fmtMs(entry.bwd)}` : ""}`,
      ),
    );
  }
  // call sites: modules and functions called from a line of this file
  const sites = new Map();
  for (const call of batch.calls) {
    if (call.call_site?.file !== file || call.kind === "function") continue;
    const list = sites.get(call.call_site.line) || [];
    list.push(call);
    sites.set(call.call_site.line, list);
  }
  for (const [line, calls] of sites) {
    const call = calls[calls.length - 1];
    const ref = firstOutput(call);
    const meta = ref ? batch.tensors[ref.tid] : null;
    const label = calls.length > 1 ? `${call.name.split(".").pop()} ×${calls.length}` : call.name;
    add(line, h("span", { class: "a mod", onclick: () => store.set({ selection: { call: call.id } }) }, `${label}${meta ? ` → ${fmtShape(meta.shape)}` : ""}`));
  }
  // ops: the shape each line produces (last op on the line, per first call that ran it)
  const ops = new Map();
  for (const op of batch.ops || []) {
    if (op.file !== file) continue;
    const key = op.line;
    const firstCall = ops.get(key)?.call ?? op.call;
    if (op.call !== firstCall) continue;
    ops.set(key, { ...op, ops: [...(ops.get(key)?.ops || []), op.op] });
  }
  for (const [line, op] of ops) {
    if (sites.has(line)) continue;
    add(line, h("span", { class: "a", title: op.ops.join(" → "), onclick: op.call != null ? () => store.set({ selection: { call: op.call } }) : null }, `${op.op} → ${op.shapes.map(fmtShape).join(", ")} ${op.dtype}`));
  }
  return byLine;
}
