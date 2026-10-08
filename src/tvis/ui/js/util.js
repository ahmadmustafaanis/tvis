// Small DOM + formatting helpers shared by every view.

export function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else if (key === "html") node.innerHTML = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  append(node, children);
  return node;
}

function append(node, children) {
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export function clear(node, ...children) {
  node.replaceChildren();
  append(node, children);
  return node;
}

export function fmtMs(ms) {
  if (ms == null || Number.isNaN(ms)) return "—";
  if (ms < 0.01) return `${(ms * 1000).toFixed(1)}µs`;
  if (ms < 1) return `${(ms * 1000).toFixed(0)}µs`;
  if (ms < 100) return `${ms.toFixed(2)}ms`;
  if (ms < 10000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function fmtNum(x, digits = 4) {
  if (x == null) return "—";
  if (typeof x !== "number") return String(x);
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return "0";
  const a = Math.abs(x);
  if (a >= 1e5 || a < 1e-3) return x.toExponential(2);
  return Number(x.toPrecision(digits)).toString();
}

export function fmtInt(n) {
  if (n == null) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(1)}k`;
  return n.toLocaleString();
}

export const fmtShape = (shape) => (shape ? `[${shape.join(", ")}]` : "");

export const KIND_LABEL = {
  module: "mod",
  function: "fn",
  data: "data",
  sample: "item",
  pipeline: "pipe",
  transform: "xf",
  tokenize: "tok",
  loss: "loss",
  backward: "bwd",
  optimizer: "opt",
};

const KIND_COLOR = { sample: "data", pipeline: "transform" };
export const kindColor = (kind) => `var(--k-${KIND_COLOR[kind] || kind})`;
export const phaseColor = (phase) => `var(--ph-${phase})`;

export function badge(kind) {
  return h("span", { class: "badge", style: { "--kc": kindColor(kind) } }, KIND_LABEL[kind] || kind);
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

let tooltipNode;
export function tooltip(target, text) {
  tooltipNode ||= document.getElementById("tooltip");
  target.addEventListener("mouseenter", () => {
    const content = typeof text === "function" ? text() : text;
    if (!content) return;
    tooltipNode.textContent = content;
    tooltipNode.classList.remove("hidden");
  });
  target.addEventListener("mousemove", (event) => {
    tooltipNode.style.left = `${Math.min(event.clientX + 12, window.innerWidth - 370)}px`;
    tooltipNode.style.top = `${event.clientY + 14}px`;
  });
  target.addEventListener("mouseleave", () => tooltipNode.classList.add("hidden"));
}

export function decodeFloat32(b64) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return new Float32Array(bytes.buffer);
}

export function decodeBytes(b64) {
  return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
}

export function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}
