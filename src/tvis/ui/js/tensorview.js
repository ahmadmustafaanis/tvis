// A captured tensor: header with shape/dtype, stats, histogram, and an expandable heatmap/image.

import { api } from "./api.js";
import { isImageShape, sampleShape } from "./model.js";
import { clear, cssVar, decodeBytes, decodeFloat32, fmtNum, fmtShape, h } from "./util.js";

const VIRIDIS = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];

function lerp(stops, t) {
  const x = Math.min(0.9999, Math.max(0, t)) * (stops.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  return stops[i].map((v, k) => v + (stops[i + 1][k] - v) * f);
}

function hexToRgb(hex) {
  const m = hex.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(m.slice(i, i + 2), 16));
}

function diverging() {
  const mid = hexToRgb(cssVar("--bg-elev-2") || "#1b1f26");
  return [[59, 110, 214], mid, [214, 64, 69]];
}

export function colorFor(min, max) {
  if (min != null && max != null && min < 0 && max > 0) {
    const stops = diverging();
    const span = Math.max(-min, max) || 1;
    return { fn: (v) => lerp(stops, (v / span + 1) / 2), ramp: stops, label: [-span, span] };
  }
  const lo = min ?? 0;
  const span = (max ?? 1) - lo || 1;
  return { fn: (v) => lerp(VIRIDIS, (v - lo) / span), ramp: VIRIDIS, label: [min, max] };
}

function rampCss(stops) {
  return `linear-gradient(90deg, ${stops.map((c) => `rgb(${c.join(",")})`).join(",")})`;
}

export function drawHistogram(canvas, hist, color) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 300;
  const height = canvas.clientHeight || 34;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  if (!hist?.counts?.length) return;
  const peak = Math.max(...hist.counts) || 1;
  const w = width / hist.counts.length;
  ctx.fillStyle = color;
  hist.counts.forEach((count, i) => {
    if (!count) return;
    const bh = Math.max(1, (Math.sqrt(count) / Math.sqrt(peak)) * (height - 2));
    ctx.fillRect(i * w + 0.5, height - bh, Math.max(1, w - 1), bh);
  });
  if (hist.lo < 0 && hist.hi > 0) {
    const zero = ((0 - hist.lo) / (hist.hi - hist.lo)) * width;
    ctx.fillStyle = cssVar("--text-faint");
    ctx.fillRect(zero, 0, 1, height);
  }
}

/** Draw a tensor_view/thumb response as a heatmap into `canvas` (one pixel per cell). */
export function drawHeat(canvas, view) {
  const values = decodeFloat32(view.data);
  canvas.width = view.cols;
  canvas.height = view.rows;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(view.cols, view.rows);
  const cmap = colorFor(view.min, view.max);
  for (let i = 0; i < values.length; i++) {
    const [r, g, b] = Number.isFinite(values[i]) ? cmap.fn(values[i]) : [242, 103, 107];
    img.data.set([r, g, b, 255], i * 4);
  }
  ctx.putImageData(img, 0, 0);
}

export function drawImage(canvas, image) {
  const { width, height } = image;
  const rgb = decodeBytes(image.data);
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const data = ctx.createImageData(width, height);
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    data.data[j] = rgb[i];
    data.data[j + 1] = rgb[i + 1];
    data.data[j + 2] = rgb[i + 2];
    data.data[j + 3] = 255;
  }
  ctx.putImageData(data, 0, 0);
}

/**
 * opts: {run, ref, meta, name, sample, grad, open, onToggle}
 * `sample` slices the tensor along its batch dimension when it has one.
 */
export function tensorCard(opts) {
  const { meta, name } = opts;
  const wrap = h("div", { class: "tensor" });
  if (!meta) {
    wrap.append(h("div", { class: "tensor-head" }, h("span", { class: "nm" }, name), h("span", { class: "faint" }, "not captured")));
    return wrap;
  }
  let grad = Boolean(opts.grad) && meta.grad_stats != null;
  let open = Boolean(opts.open);
  const following = opts.sample != null && meta.batch_dim != null;
  const shownShape = following ? sampleShape(meta) : meta.shape;

  const head = h(
    "div",
    { class: "tensor-head", onclick: () => ((open = !open), render()) },
    h(
      "div",
      {},
      h("span", { class: "nm" }, name),
      " ",
      h("span", { class: "sh" }, `${fmtShape(shownShape)} ${meta.dtype}`),
      following ? h("span", { class: "chip small on", style: { marginLeft: "6px" } }, `sample ${opts.sample}`) : null,
      meta.batch_dim?.inferred ? h("span", { class: "faint", title: "batch dimension inferred from a merged leading dim" }, " · batch≈") : null,
    ),
    h("span", { class: "faint caret-indicator" }, open ? "▾" : "▸"),
  );
  const statsRow = h("div", { class: "tensor-stats" });
  const histCanvas = h("canvas", { class: "hist" });
  const body = h("div", { class: "tensor-body" });
  wrap.append(head, statsRow, h("div", { style: { padding: "0 10px 8px" } }, histCanvas), body);

  function render() {
    const stats = grad ? meta.grad_stats : meta.stats;
    clear(
      statsRow,
      stat("mean", stats?.mean),
      stat("std", stats?.std),
      stat("min", stats?.min),
      stat("max", stats?.max),
      stat("|x|", stats?.abs_mean),
      stat("‖x‖", stats?.norm),
      stat("zeros", stats?.zero_frac != null ? `${(stats.zero_frac * 100).toFixed(1)}%` : null),
      stats?.nan || stats?.inf
        ? h("span", { class: "err" }, `${stats.nan} NaN · ${stats.inf} Inf`)
        : h("span", {}, h("b", {}, "n"), (stats?.numel ?? 0).toLocaleString()),
    );
    requestAnimationFrame(() => drawHistogram(histCanvas, stats?.hist, cssVar(grad ? "--k-backward" : "--accent")));
    body.classList.toggle("hidden", !open);
    head.querySelector(".caret-indicator").textContent = open ? "▾" : "▸";
    if (open) renderBody();
  }

  function renderBody() {
    const toggles = h("div", { class: "toolbar" });
    const seg = h("div", { class: "seg" });
    for (const [label, value] of [["Value", false], ["Gradient", true]]) {
      const disabled = value && meta.grad_stats == null;
      seg.append(
        h(
          "button",
          {
            class: grad === value ? "on" : "",
            disabled,
            title: disabled ? "no gradient flowed to this tensor" : "",
            onclick: (e) => {
              e.stopPropagation();
              grad = value;
              render();
            },
          },
          label,
        ),
      );
    }
    toggles.append(seg);
    const stored = grad ? meta.grad_stored : meta.stored;
    if (stored === "rows") toggles.append(h("span", { class: "faint" }, `first ${grad ? meta.grad_stored_rows : meta.stored_rows} rows stored`));
    const view = h("div", {});
    clear(body, toggles, view);
    if (!stored || stored === "none") {
      view.append(h("div", { class: "faint" }, "too large to store — statistics only (raise --max-elems)"));
      return;
    }
    const imageLike = meta.kind === "image" || isImageShape(following ? sampleShape(meta) : meta.shape);
    if (imageLike) renderImage(view);
    else renderHeatmap(view, []);
  }

  async function renderImage(view) {
    const canvas = h("canvas", { class: "heat", style: { maxWidth: "260px" } });
    clear(view, canvas, h("div", { class: "readout" }, grad ? "gradient magnitude per pixel" : meta.kind === "image" ? "raw image" : "un-normalised for display"));
    try {
      const image = await api("image", params());
      drawImage(canvas, image);
    } catch (err) {
      clear(view, h("div", { class: "err" }, String(err.message || err)));
    }
  }

  async function renderHeatmap(view, fixed) {
    clear(view, h("div", { class: "loading" }, "loading…"));
    let data;
    try {
      data = await api("tensor", { ...params(), fixed });
    } catch (err) {
      clear(view, h("div", { class: "err" }, String(err.message || err)));
      return;
    }
    const dims = h("div", { class: "dims" });
    data.leading.forEach((size, i) => {
      if (size <= 1) return;
      const out = h("span", { class: "mono" }, String(data.fixed[i]));
      const input = h("input", {
        type: "range",
        min: 0,
        max: size - 1,
        value: data.fixed[i],
        oninput: () => (out.textContent = input.value),
        onchange: () => {
          const next = [...data.fixed];
          next[i] = Number(input.value);
          renderHeatmap(view, next);
        },
      });
      dims.append(h("label", {}, `dim ${i}`, input, out));
    });
    const canvas = h("canvas", { class: "heat" });
    const readout = h("div", { class: "readout" });
    const cmap = colorFor(data.min, data.max);
    const legend = h(
      "div",
      { class: "heat-legend" },
      h("span", {}, fmtNum(cmap.label[0])),
      h("span", { class: "ramp", style: { background: rampCss(cmap.ramp) } }),
      h("span", {}, fmtNum(cmap.label[1])),
    );
    const note = data.stride[0] > 1 || data.stride[1] > 1 ? h("div", { class: "faint" }, `downsampled ×${data.stride.join("×")}`) : null;
    clear(view, dims, h("div", { class: "heat-wrap" }, canvas), legend, readout, note);
    const values = decodeFloat32(data.data);
    canvas.width = data.cols;
    canvas.height = data.rows;
    const aspect = data.rows / data.cols;
    canvas.style.height = `${Math.max(18, Math.min(360, (canvas.clientWidth || 320) * aspect))}px`;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(data.cols, data.rows);
    const nanColor = hexToRgb(cssVar("--bad") || "#f2676b");
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      const [r, g, b] = Number.isFinite(v) ? cmap.fn(v) : nanColor;
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    canvas.addEventListener("mousemove", (event) => {
      const rect = canvas.getBoundingClientRect();
      const col = Math.floor(((event.clientX - rect.left) / rect.width) * data.cols);
      const row = Math.floor(((event.clientY - rect.top) / rect.height) * data.rows);
      if (row < 0 || col < 0 || row >= data.rows || col >= data.cols) return;
      const idx = [...data.fixed, row * data.stride[0], col * data.stride[1]];
      readout.textContent = `[${idx.join(", ")}] = ${fmtNum(values[row * data.cols + col], 6)}`;
    });
  }

  function params() {
    const p = { run: opts.run, batch: opts.ref.batch, tid: opts.ref.tid, grad };
    if (following) p.sample = opts.sample;
    return p;
  }

  render();
  return wrap;
}

function stat(label, value) {
  return h("span", {}, h("b", {}, label), typeof value === "number" ? fmtNum(value) : value ?? "—");
}
