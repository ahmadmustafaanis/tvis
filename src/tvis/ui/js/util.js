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

// 16×16 line icons (drawn here so the UI needs no icon font or extra files). "f:" paths are filled.
const ICONS = {
  play: ["f:M5 3.2v9.6L13 8z"],
  pause: ["f:M4.5 3h2.6v10H4.5zM8.9 3h2.6v10H8.9z"],
  prev: ["M4 3.5v9", "f:M13 3.5v9L6 8z"],
  next: ["M12 3.5v9", "f:M3 3.5v9L10 8z"],
  restart: ["M2.5 3v3.5H6", "M2.9 6.4A5.5 5.5 0 1 1 3.2 10.5"],
  left: ["M10 3.5L5.5 8l4.5 4.5"],
  right: ["M6 3.5L10.5 8 6 12.5"],
  close: ["M4 4l8 8M12 4l-8 8"],
  theme: ["M8 2a6 6 0 1 0 0 12A6 6 0 1 0 8 2z", "f:M8 2a6 6 0 0 0 0 12z"],
  alert: ["M8 2.2l6.3 11H1.7z", "M8 6.5v3", "M8 11.4v.2"],
  hide: ["M1.8 8S4 3.8 8 3.8 14.2 8 14.2 8 12 12.2 8 12.2 1.8 8 1.8 8z", "M8 6a2 2 0 1 0 0 4 2 2 0 1 0 0-4z", "M2.5 13.5l11-11"],
  keys: ["M1.8 4h12.4v8H1.8z", "M4.5 6.5h.1M7 6.5h.1M9.5 6.5h.1M12 6.5h.1M5 9.5h6"],
  fit: ["M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10"],
  layers: ["M8 2l6 3-6 3-6-3z", "M2 8l6 3 6-3", "M2 11l6 3 6-3"],
  target: ["M8 2.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 1 0 0-11z", "f:M8 6.2a1.8 1.8 0 1 0 0 3.6 1.8 1.8 0 1 0 0-3.6z"],
};

export function icon(name) {
  const NS = "http://www.w3.org/2000/svg";
  const root = document.createElementNS(NS, "svg");
  root.setAttribute("viewBox", "0 0 16 16");
  root.setAttribute("class", "icon");
  root.setAttribute("aria-hidden", "true");
  for (const spec of ICONS[name] || []) {
    const path = document.createElementNS(NS, "path");
    const filled = spec.startsWith("f:");
    path.setAttribute("d", filled ? spec.slice(2) : spec);
    path.setAttribute("fill", filled ? "currentColor" : "none");
    if (!filled) {
      path.setAttribute("stroke", "currentColor");
      path.setAttribute("stroke-width", "1.5");
      path.setAttribute("stroke-linecap", "round");
      path.setAttribute("stroke-linejoin", "round");
    }
    root.append(path);
  }
  return root;
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
    if (typeof text === "function") {
      const content = text();
      tooltipNode.textContent = content || "";
      tooltipNode.classList.toggle("hidden", !content);
    }
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
