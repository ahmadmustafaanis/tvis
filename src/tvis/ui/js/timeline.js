// Timeline tab: forward and backward lanes as a flame chart (SVG).

import { index } from "./model.js";
import { store } from "./store.js";
import { clear, fmtMs, h, kindColor, phaseColor, tooltip } from "./util.js";

const ROW = 18;
const NS = "http://www.w3.org/2000/svg";

function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
}

export function renderTimeline(container) {
  const state = store.get();
  const batch = state.batch;
  if (!batch) return clear(container, h("div", { class: "loading" }, "loading…"));
  const idx = index(batch);
  const fwd = batch.calls.filter((c) => c.start_ms != null && c.end_ms != null && c.kind !== "sample" && c.kind !== "transform");
  const bwd = batch.calls.filter((c) => c.bwd_start_ms != null);
  if (!fwd.length) return clear(container, h("div", { class: "empty" }, "No timing recorded."));
  const t0 = Math.min(...fwd.map((c) => c.start_ms), batch.timing?.start_ms ?? Infinity);
  const t1 = Math.max(...fwd.map((c) => c.end_ms), ...bwd.map((c) => c.bwd_end_ms));
  const width = Math.max(600, container.clientWidth - 32);
  const x = (t) => 70 + ((t - t0) / (t1 - t0 || 1)) * (width - 80);
  const depthOf = new Map();
  for (const c of batch.calls) depthOf.set(c.id, c.depth);
  const fDepth = Math.max(...fwd.map((c) => c.depth)) + 1;
  const bDepth = bwd.length ? Math.max(...bwd.map((c) => c.depth)) + 1 : 0;
  const top = 26;
  const bTop = top + fDepth * ROW + 34;
  const height = bTop + bDepth * ROW + 20;
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, height });

  // phase bands
  const marks = batch.timing?.phases_ms ? phaseSpans(batch) : [];
  for (const span of marks) root.append(svg("rect", { x: x(span.start), y: top - 8, width: Math.max(1, x(span.end) - x(span.start)), height: 4, fill: phaseColor(span.phase), rx: 2 }));
  // axis
  for (let i = 0; i <= 5; i++) {
    const t = t0 + ((t1 - t0) * i) / 5;
    root.append(svg("line", { x1: x(t), x2: x(t), y1: top - 2, y2: height - 14, class: "axis" }));
    root.append(svg("text", { x: x(t) + 3, y: height - 3, class: "tick" }, fmtMs(t - t0)));
  }
  root.append(svg("text", { x: 4, y: top + 12, class: "lane-label" }, "forward"));
  root.append(svg("text", { x: 4, y: bTop + 12, class: "lane-label" }, "backward"));

  const draw = (call, start, end, y, lane) => {
    const w = Math.max(1, x(end) - x(start));
    const sel = state.selection?.call === call.id;
    const rect = svg("rect", { x: x(start), y, width: w, height: ROW - 3, rx: 3, fill: lane === "bwd" ? phaseColor("backward") : kindColor(call.kind), class: `call ${sel ? "sel" : ""}`, opacity: lane === "bwd" ? 0.55 + 0.45 / (1 + call.depth) : 0.9 });
    rect.addEventListener("click", () => store.set({ selection: { call: call.id } }));
    tooltip(rect, `${call.name} · ${lane === "bwd" ? `backward ≈${fmtMs(call.bwd_ms)}` : fmtMs(call.ms)}`);
    root.append(rect);
    if (w > 40) root.append(svg("text", { x: x(start) + 4, y: y + 11.5, class: "lbl" }, call.name.length * 6 > w ? `${call.name.slice(0, Math.floor(w / 6.5))}…` : call.name));
  };
  for (const c of fwd) draw(c, c.start_ms, c.end_ms, top + c.depth * ROW, "fwd");
  for (const c of bwd) draw(c, c.bwd_start_ms, c.bwd_end_ms, bTop + c.depth * ROW, "bwd");

  clear(
    container,
    h(
      "div",
      { class: "view timeline" },
      h("div", { class: "toolbar" }, h("span", { class: "muted" }, `${idx.byId.size} calls`), h("span", { class: "faint" }, "Device time with tvis capture overhead removed. Backward bars are inferred from gradient arrival.")),
      root,
    ),
  );
}

function phaseSpans(batch) {
  // reconstruct contiguous phase spans from the order of calls' phases
  const spans = [];
  let current = null;
  for (const c of batch.calls) {
    if (c.start_ms == null) continue;
    if (!current || current.phase !== c.phase) {
      current = { phase: c.phase, start: c.start_ms, end: c.end_ms };
      spans.push(current);
    } else current.end = Math.max(current.end, c.end_ms);
  }
  return spans;
}
