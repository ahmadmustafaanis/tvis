// Small shared chart pieces: categorical colours and a scatter plot.

import { cssVar, fmtNum, h, tooltip } from "./util.js";

const NS = "http://www.w3.org/2000/svg";
export const svg = (tag, attrs = {}, text) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  return n;
};

export function categorical(labels) {
  const unique = [...new Set(labels.map(String))];
  const map = new Map(unique.map((label, i) => [label, cssVar(`--cat-${i % 10}`)]));
  return { color: (label) => map.get(String(label)) || cssVar("--cat-8"), entries: [...map.entries()] };
}

/** points: [{x, y, label, ...}]. opts: {color(p), title(p), onClick(p), text(p), hollow(p)} */
export function scatter(points, opts = {}) {
  const W = 640;
  const H = 420;
  const pad = 24;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const sx = (x) => pad + ((x - x0) / (x1 - x0 || 1)) * (W - 2 * pad);
  const sy = (y) => H - pad - ((y - y0) / (y1 - y0 || 1)) * (H - 2 * pad);
  const root = svg("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, style: "max-width: 760px" });
  root.append(svg("rect", { x: 1, y: 1, width: W - 2, height: H - 2, rx: 8, fill: "none", stroke: cssVar("--border") }));
  root.append(svg("text", { class: "axis-label", x: W - pad, y: H - 6, "text-anchor": "end" }, "PC 1 →"));
  root.append(svg("text", { class: "axis-label", x: 8, y: 16 }, "↑ PC 2"));
  for (const p of points) {
    const color = opts.color ? opts.color(p) : cssVar("--accent");
    const hollow = opts.hollow?.(p);
    const dot = svg("circle", {
      cx: sx(p.x),
      cy: sy(p.y),
      r: opts.radius || 5,
      fill: hollow ? "none" : color,
      stroke: color,
      "stroke-width": hollow ? 2 : 0,
      "fill-opacity": 0.85,
      style: opts.onClick ? "cursor:pointer" : "",
    });
    if (opts.onClick) dot.addEventListener("click", () => opts.onClick(p));
    if (opts.title) tooltip(dot, opts.title(p));
    root.append(dot);
    const text = opts.text?.(p);
    if (text) root.append(svg("text", { x: sx(p.x) + 7, y: sy(p.y) + 3, fill: cssVar("--text-muted"), "font-size": 10 }, text));
  }
  return root;
}

export function legend(entries, extra = []) {
  return h(
    "div",
    { class: "legend-row" },
    entries.map(([label, color]) => h("span", {}, h("i", { class: "swatch", style: { background: color } }), label)),
    extra,
  );
}

export const pct = (v) => `${fmtNum(v * 100, 3)}%`;
