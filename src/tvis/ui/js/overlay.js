// Heat overlays on input images, shared by Grad-CAM, Attention and the layer views.

import { decodeBytes } from "./util.js";

export function jet(t) {
  const r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3)));
  const g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2)));
  const b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1)));
  return [r * 255, g * 255, b * 255];
}

/** Draw an RGB image scaled to the canvas, then a [rows × cols] heat in [0, 1] smoothed over it. */
export function paintOverlay(canvas, image, values, rows, cols, opacity = 0.55) {
  const ctx = canvas.getContext("2d");
  const base = document.createElement("canvas");
  base.width = image.width;
  base.height = image.height;
  const bctx = base.getContext("2d");
  const rgb = decodeBytes(image.data);
  const img = bctx.createImageData(image.width, image.height);
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) img.data.set([rgb[i], rgb[i + 1], rgb[i + 2], 255], j);
  bctx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(base, 0, 0, canvas.width, canvas.height);
  if (!values) return;
  const heat = document.createElement("canvas");
  heat.width = cols;
  heat.height = rows;
  const hctx = heat.getContext("2d");
  const himg = hctx.createImageData(cols, rows);
  values.forEach((v, i) => himg.data.set([...jet(v), Math.round(255 * opacity * Math.min(1, v * 1.2))], i * 4));
  hctx.putImageData(himg, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(heat, 0, 0, canvas.width, canvas.height);
}
