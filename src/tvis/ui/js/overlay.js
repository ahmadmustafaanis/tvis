// Heat overlays on input images, shared by Grad-CAM, Attention and the layer views.

import { decodeBytes } from "./util.js";

// Inferno (perceptually uniform, readable in greyscale and by colour-blind viewers), as a
// degree-6 polynomial fit per channel. Brighter = more.
const INFERNO = [
  [0.000218940369, 0.001651004631, -0.019480898437],
  [0.106513419486, 0.563956436788, 3.932712388889],
  [11.602493082472, -3.972853965666, -15.942394106291],
  [-41.703996131395, 17.436398882053, 44.354145198728],
  [77.162935699427, -33.402358942101, -81.807309257390],
  [-71.319428244992, 32.626064263977, 73.209519858032],
  [25.131126224773, -12.242668952386, -23.070325002872],
];

export function heatColor(t) {
  const x = Math.min(1, Math.max(0, t));
  return [0, 1, 2].map((ch) => {
    let v = 0;
    for (let k = INFERNO.length - 1; k >= 0; k--) v = v * x + INFERNO[k][ch];
    return Math.round(Math.min(1, Math.max(0, v)) * 255);
  });
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
  values.forEach((v, i) => himg.data.set([...heatColor(v), Math.round(255 * opacity * Math.min(1, v * 1.2))], i * 4));
  hctx.putImageData(himg, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(heat, 0, 0, canvas.width, canvas.height);
}
