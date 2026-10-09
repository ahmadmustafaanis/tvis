// 3D page: the model as a tunnel of glowing slabs built from the recorded dataflow, with the
// replay timeline driving it. Forward: orange particles stream in and each slab lights up with
// its real activation for the followed sample. Backward: blue particles stream back and slabs
// show their gradients, brightness ∝ ‖grad‖. Click a slab for On image / Neurons / Stats.

import { OrbitControls } from "/static/vendor/OrbitControls.js";
import * as THREE from "three";
import { api } from "./api.js";
import { layerView } from "./layerview.js";
import { followableSamples, imageInput, index, lossValue, patchGrid, sampleShape, tensorRefs } from "./model.js";
import { buildEvents, layerCalls } from "./replay.js";
import { store } from "./store.js";
import { clear, decodeBytes, decodeFloat32, fmtNum, h, phaseColor } from "./util.js";

const GRANULARITIES = [
  ["layer", "Layer"],
  ["phase", "Phase"],
  ["batch", "Batch"],
];
const SPEEDS = [0.5, 1, 2, 4];
const BG = 0x07090d;
const WARM = [
  [0, 0, 0],
  [120, 20, 10],
  [235, 90, 20],
  [255, 190, 60],
  [255, 250, 220],
];
const COOL = [
  [0, 0, 0],
  [10, 40, 120],
  [30, 130, 235],
  [120, 220, 255],
  [235, 250, 255],
];

const VIEW_DIR = new THREE.Vector3(-5.5, 3.2, 8.5).normalize();
const OVERVIEW_DIR = new THREE.Vector3(-0.12, 0.32, 1).normalize(); // near-frontal: the whole tunnel reads left to right
const FOLLOW_DISTANCE = 11; // at a 16:9 view; narrower views back off so the same span stays visible
const LOOK_AHEAD = 3.2; // look past the active slab so what comes next is on screen

const S = {
  runKey: null,
  events: [],
  stops: [],
  i: 0,
  playing: false,
  timer: null,
  speed: 1,
  gran: "layer",
  follow: true,
  expanded: new Set(),
  graph: null,
  graphKey: null,
  three: null,
  dom: null,
  textures: new Map(),
  selected: null,
};

if (new URLSearchParams(location.search).has("debug3d")) window.tvis3d = S; // for debugging the scene

// ---------------------------------------------------------------------------------------------
// page lifecycle
// ---------------------------------------------------------------------------------------------
export function renderScene3d(container) {
  const state = store.get();
  if (!state.allBatches) return clear(container, h("div", { class: "loading" }, "loading the recording…"));
  if (S.runKey !== state.runId) {
    S.runKey = state.runId;
    S.events = buildEvents(state.allBatches, state.steps, "layer");
    S.i = 0;
    S.graphKey = null;
    S.selected = null;
  }
  if (!S.dom || !container.contains(S.dom.root)) buildDom(container);
  if (!S.three) initThree();
  else S.dom.viewport.append(S.three.renderer.domElement);
  S.stops = computeStops();
  startLoop();
  update();
}

export function stopScene3d() {
  pause();
  if (S.three?.raf) cancelAnimationFrame(S.three.raf);
  if (S.three) S.three.raf = null;
}

export function scene3dKey(event) {
  if (event.key === " ") {
    event.preventDefault();
    toggle();
  } else if (event.key === "ArrowRight") go(nextStop(1));
  else if (event.key === "ArrowLeft") go(nextStop(-1));
  else if (event.key === "]") shiftSample(1);
  else if (event.key === "[") shiftSample(-1);
  else return false;
  return true;
}

// ---------------------------------------------------------------------------------------------
// graph: nodes from the recorded calls, edges from recorded tensor identity
// ---------------------------------------------------------------------------------------------
const unitKey = (call) => call.module_path || call.qualname || call.name;

function buildGraph(batch) {
  const idx = index(batch);
  const layers = layerCalls(batch, "layer");
  const layerIds = new Set(layers.map((c) => c.id));
  // a "unit" is a repeated block (blocks.0, blocks.1, … or a function called several times) that
  // wraps two or more layers; units render as one slab until expanded (outermost unit wins)
  // repeated = same class at the same position up to indices (blocks.0 … blocks.5 → "blocks.#")
  const signature = (c) =>
    c.kind === "module" ? `m:${c.cls}:${(c.module_path || c.name).replace(/\d+/g, "#")}` : `f:${c.qualname || c.name}`;
  const siblings = new Map();
  for (const c of batch.calls) {
    if (c.phase !== "forward" || (c.kind !== "module" && c.kind !== "function")) continue;
    // grouped by type wherever it's called from: NanoViT runs blocks[:-1] and blocks[-1] separately
    const key = signature(c);
    siblings.set(key, (siblings.get(key) || 0) + 1);
  }
  const descendantLayers = (call) => {
    let n = 0;
    const walk = (id) => {
      for (const k of idx.children.get(id) || []) {
        if (layerIds.has(k.id)) n++;
        walk(k.id);
      }
    };
    walk(call.id);
    return n;
  };
  const isUnit = (c) =>
    c.phase === "forward" &&
    (c.kind === "module" || c.kind === "function") &&
    siblings.get(signature(c)) > 1 &&
    descendantLayers(c) >= 2 &&
    !S.expanded.has(unitKey(c));
  const unitOf = (call) => {
    let top = null;
    let cur = call.parent != null ? idx.byId.get(call.parent) : null;
    while (cur) {
      if (isUnit(cur)) top = cur;
      cur = cur.parent != null ? idx.byId.get(cur.parent) : null;
    }
    return top;
  };
  const nodes = [];
  const byKey = new Map();
  const nodeOfCall = new Map();
  for (const layer of layers) {
    const unit = unitOf(layer);
    const call = unit || layer;
    let node = byKey.get(call.id);
    if (!node) {
      node = { id: call.id, call, unit: !!unit, members: [], index: nodes.length };
      byKey.set(call.id, node);
      nodes.push(node);
    }
    node.members.push(layer.id);
    nodeOfCall.set(layer.id, node);
  }
  // map every call (including ancestors used by events) to its node
  for (const c of batch.calls) {
    if (nodeOfCall.has(c.id)) continue;
    let cur = c;
    while (cur && !byKey.has(cur.id)) cur = cur.parent != null ? idx.byId.get(cur.parent) : null;
    if (cur) nodeOfCall.set(c.id, byKey.get(cur.id));
  }
  // edges: a node consumes a tensor another node produced (exact tensor identity from capture)
  const producer = new Map();
  for (const node of nodes) for (const out of node.call.outputs || []) for (const ref of tensorRefs(out.value)) producer.set(ref.tid, node);
  const edges = [];
  for (const node of nodes) {
    const sources = new Set();
    for (const inp of node.call.inputs || []) for (const ref of tensorRefs(inp.value)) {
      const src = producer.get(ref.tid);
      if (src && src.index < node.index) sources.add(src);
    }
    if (!sources.size && node.index > 0) edges.push({ from: nodes[node.index - 1], to: node, inferred: true });
    for (const src of sources) edges.push({ from: src, to: node, inferred: false });
  }
  return { batch, nodes, edges, nodeOfCall };
}

function nodeShape(batch, node) {
  const ref = tensorRefs(node.call.outputs?.[0]?.value || {})[0];
  const meta = ref ? batch.tensors[ref.tid] : null;
  if (!meta) return { ref: null, meta: null, shape: [], kind: "none" };
  const shape = meta.batch_dim ? sampleShape(meta) : meta.shape;
  let kind = "other";
  if (shape.length === 3 && shape[1] > 1 && shape[2] > 1) kind = "chw";
  else if (shape.length === 2) kind = patchGrid(shape[0]) ? "tokens" : "seq";
  else if (shape.length === 1) kind = "vector";
  return { ref, meta, shape, kind };
}

const lg = (n) => Math.log2(Math.max(1, n));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function slabSize(info) {
  const [a = 1, b = 1, c = 1] = info.shape;
  switch (info.kind) {
    case "chw": {
      const face = clamp(0.9 + 0.22 * lg(b * c), 1, 4.2);
      return { w: face, h: face, t: clamp(0.06 + 0.05 * lg(a), 0.08, 0.7) };
    }
    case "tokens": {
      const face = clamp(0.9 + 0.22 * lg(a), 1, 4.2);
      return { w: face, h: face, t: clamp(0.06 + 0.05 * lg(b), 0.08, 0.7) };
    }
    case "seq":
      return { w: clamp(0.6 + 0.3 * lg(a), 0.6, 4.2), h: clamp(0.6 + 0.3 * lg(b), 0.6, 4.2), t: 0.12 };
    case "vector":
      return { w: 0.35, h: clamp(0.6 + 0.32 * lg(a), 0.6, 4.4), t: 0.35 };
    default:
      return { w: 0.8, h: 0.8, t: 0.2 };
  }
}

// ---------------------------------------------------------------------------------------------
// three.js scene
// ---------------------------------------------------------------------------------------------
function initThree() {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setClearColor(BG);
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(BG, 0.022);
  const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 600);
  camera.position.set(-6, 4, 11);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.addEventListener("start", () => {
    if (S.follow) setFollow(false);
  });
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(-4, 10, 8);
  scene.add(key);
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(800, 60),
    new THREE.MeshStandardMaterial({ color: 0x0b0e14, roughness: 0.35, metalness: 0.6 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -3.2;
  scene.add(floor);
  const world = new THREE.Group();
  scene.add(world);
  S.three = { renderer, scene, camera, controls, world, clock: new THREE.Clock(), raf: null, target: new THREE.Vector3(), flashUntil: 0, distance: FOLLOW_DISTANCE, viewDir: VIEW_DIR };
  S.dom.viewport.append(renderer.domElement);
  // horizontal trackpad swipes / shift+wheel move along the model; vertical wheel still zooms
  S.dom.viewport.addEventListener(
    "wheel",
    (e) => {
      const dx = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.shiftKey ? e.deltaY : 0;
      if (!dx) return;
      e.preventDefault();
      e.stopPropagation();
      panTo(controls.target.x + dx * 0.02);
    },
    { capture: true, passive: false },
  );
  new ResizeObserver(resize).observe(S.dom.viewport);
  resize();
  renderer.domElement.addEventListener("click", onClick);
}

function resize() {
  const { renderer, camera } = S.three;
  const rect = S.dom.viewport.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  renderer.setSize(rect.width, rect.height);
  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
}

function disposeWorld() {
  const { world } = S.three;
  world.traverse((o) => {
    o.geometry?.dispose();
    for (const m of [].concat(o.material || [])) m.dispose();
  });
  world.clear();
}

function buildWorld(graph, sample) {
  disposeWorld();
  const { world } = S.three;
  let x = 0;
  const gap = 1.4;
  for (const node of graph.nodes) {
    node.info = nodeShape(graph.batch, node);
    node.size = slabSize(node.info);
    x += node.size.t / 2;
    node.x = x;
    x += node.size.t / 2 + gap;
    const side = new THREE.MeshStandardMaterial({ color: 0x1c2533, transparent: true, opacity: 0.35, roughness: 0.2, metalness: 0.4 });
    const face = new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: 0.18, emissive: 0xffffff, emissiveIntensity: 0, roughness: 0.4 });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(node.size.t, node.size.h, node.size.w), [face, face, side, side, side, side]);
    mesh.position.set(node.x, 0, 0);
    mesh.userData.node = node;
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry), new THREE.LineBasicMaterial({ color: 0x5b6b85, transparent: true, opacity: 0.6 }));
    mesh.add(edges);
    node.mesh = mesh;
    node.face = face;
    node.edgeLines = edges;
    world.add(mesh);
    const label = textSprite(node.unit ? `${node.call.name} ▸` : node.call.name, node.unit ? "#ffcf8a" : "#aeb8c8");
    label.position.set(node.x, -node.size.h / 2 - 0.45, 0);
    world.add(label);
  }
  graph.length = x;
  // input image plane and output marker
  graph.input = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 2.4), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95, side: THREE.DoubleSide }));
  graph.input.rotation.y = Math.PI / 2;
  graph.input.position.set(-3, 0, 0);
  world.add(graph.input);
  graph.output = textSprite("", "#ffffff", 0.8);
  graph.output.position.set(x + 1.6, 1.2, 0);
  world.add(graph.output);
  // edges: straight to the next slab, arcs for skips (residual-style dataflow)
  for (const edge of graph.edges) {
    const a = new THREE.Vector3(edge.from.x + edge.from.size.t / 2, 0, 0);
    const b = new THREE.Vector3(edge.to.x - edge.to.size.t / 2, 0, 0);
    const skip = edge.to.index - edge.from.index;
    const curve =
      skip > 1
        ? new THREE.QuadraticBezierCurve3(a.clone().setY(0.4), new THREE.Vector3((a.x + b.x) / 2, 1.2 + 0.35 * Math.min(skip, 8), 0), b.clone().setY(0.4))
        : new THREE.LineCurve3(a, b);
    edge.curve = curve;
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(skip > 1 ? 40 : 2)), new THREE.LineBasicMaterial({ color: skip > 1 ? 0x4a5a78 : 0x34405a, transparent: true, opacity: edge.inferred ? 0.25 : 0.7 }));
    world.add(line);
  }
  graph.inputCurve = graph.nodes.length ? new THREE.LineCurve3(new THREE.Vector3(-3, 0, 0), new THREE.Vector3(graph.nodes[0].x - graph.nodes[0].size.t / 2, 0, 0)) : null;
  // particles, re-targeted every event
  const count = 360;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
  const material = new THREE.PointsMaterial({ size: 0.11, color: 0xff9a3c, transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false });
  graph.particles = new THREE.Points(geometry, material);
  graph.particles.userData = { curves: [], reverse: false, seeds: Float32Array.from({ length: count }, () => Math.random()), jitter: Float32Array.from({ length: count * 2 }, () => (Math.random() - 0.5) * 0.35) };
  world.add(graph.particles);
  loadInputImage(graph, sample);
}

function textSprite(text, color, scale = 0.7) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 96;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true, depthWrite: false }));
  sprite.scale.set(scale * 5.3, scale, 1);
  // optional second, dimmer line; both lines shrink to fit the canvas instead of being clipped
  sprite.userData.setText = (t, c = color, sub = "") => {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, 512, 96);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const line = (str, size, weight, y, fill) => {
      ctx.font = `${weight} ${size}px -apple-system, Inter, Segoe UI, sans-serif`;
      const fit = Math.min(size, (size * 496) / Math.max(1, ctx.measureText(str).width));
      ctx.font = `${weight} ${fit}px -apple-system, Inter, Segoe UI, sans-serif`;
      ctx.fillStyle = fill;
      ctx.fillText(str, 256, y);
    };
    const main = t.length > 30 ? `${t.slice(0, 29)}…` : t;
    if (sub) {
      line(main, 34, 500, 30, c);
      line(sub, 26, 400, 72, "#8b95a7");
    } else line(main, 40, 500, 48, c);
    sprite.material.map.needsUpdate = true;
  };
  sprite.userData.setText(text);
  return sprite;
}

async function loadInputImage(graph, sample) {
  const ref = imageInput(graph.batch, sample);
  const mat = graph.input.material;
  if (!ref) {
    const s = graph.batch.samples?.[sample];
    const text = s?.raw?.value?.text;
    mat.map = text ? canvasTexture(textCanvas(text)) : null;
    mat.needsUpdate = true;
    return;
  }
  try {
    const image = await api("image", { run: store.get().runId, batch: ref.batch, tid: ref.tid, sample });
    const canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(image.width, image.height);
    const rgb = decodeBytes(image.data);
    for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) img.data.set([rgb[i], rgb[i + 1], rgb[i + 2], 255], j);
    ctx.putImageData(img, 0, 0);
    mat.map = canvasTexture(canvas);
    mat.needsUpdate = true;
  } catch {
    /* keep the plain plane */
  }
}

function textCanvas(text) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#11161f";
  ctx.fillRect(0, 0, 256, 256);
  ctx.fillStyle = "#e4e7ec";
  ctx.font = "22px ui-monospace, Menlo, monospace";
  const words = text.split(/\s+/);
  let line = "";
  let y = 36;
  for (const w of words) {
    if ((line + w).length > 16) {
      ctx.fillText(line, 14, y);
      line = "";
      y += 30;
    }
    line += `${w} `;
  }
  ctx.fillText(line, 14, y);
  return canvas;
}

function canvasTexture(canvas) {
  const tex = new THREE.CanvasTexture(canvas);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// ---------------------------------------------------------------------------------------------
// activation / gradient textures for the followed sample
// ---------------------------------------------------------------------------------------------
function ramp(stops, t) {
  const x = clamp(t, 0, 0.9999) * (stops.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  return stops[i].map((v, k) => v + (stops[i + 1][k] - v) * f);
}

function summarise(info, values) {
  const shape = info.shape;
  if (info.kind === "chw") {
    const [C, H, W] = shape;
    const map = new Float32Array(H * W);
    for (let c = 0; c < C; c++) for (let p = 0; p < H * W; p++) map[p] += Math.abs(values[c * H * W + p]);
    return { map, rows: H, cols: W };
  }
  if (info.kind === "tokens") {
    const [T, D] = shape;
    const { prefix, grid } = patchGrid(T);
    const map = new Float32Array(grid * grid);
    for (let p = 0; p < grid * grid; p++) {
      let s = 0;
      for (let d = 0; d < D; d++) s += values[(prefix + p) * D + d] ** 2;
      map[p] = Math.sqrt(s);
    }
    return { map, rows: grid, cols: grid };
  }
  if (info.kind === "seq") {
    const [T, D] = shape;
    const cols = Math.min(T, 64);
    const rows = Math.min(D, 64);
    const map = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) map[r * cols + c] = Math.abs(values[Math.floor((c * T) / cols) * D + Math.floor((r * D) / rows)]);
    return { map, rows, cols };
  }
  if (info.kind === "vector") {
    const n = Math.min(shape[0], 128);
    const map = Float32Array.from({ length: n }, (_, i) => Math.abs(values[Math.floor((i * shape[0]) / n)]));
    return { map, rows: n, cols: 1 };
  }
  return null;
}

async function texture(node, sample, grad) {
  const { ref, meta } = node.info;
  if (!ref || !meta) return null;
  const stored = grad ? meta.grad_stored : meta.stored;
  if (!stored || stored === "none") return null;
  const key = `${ref.batch}:${ref.tid}:${sample}:${grad}`;
  if (S.textures.has(key)) return S.textures.get(key);
  const promise = (async () => {
    const params = { run: store.get().runId, batch: ref.batch, tid: ref.tid, grad };
    if (meta.batch_dim) params.sample = sample;
    const data = await api("array", params);
    const summary = summarise(node.info, decodeFloat32(data.data));
    if (!summary) return null;
    let peak = 0;
    for (const v of summary.map) peak = Math.max(peak, v);
    const canvas = document.createElement("canvas");
    canvas.width = summary.cols;
    canvas.height = summary.rows;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(summary.cols, summary.rows);
    summary.map.forEach((v, i) => img.data.set([...ramp(grad ? COOL : WARM, Math.sqrt(v / (peak || 1))), 255], i * 4));
    ctx.putImageData(img, 0, 0);
    return canvasTexture(canvas);
  })().catch(() => null);
  S.textures.set(key, promise);
  return promise;
}

// ---------------------------------------------------------------------------------------------
// playback
// ---------------------------------------------------------------------------------------------
function nodeOfEvent(event) {
  return event?.call && S.graph ? S.graph.nodeOfCall.get(event.call.id) : null;
}

function computeStops() {
  const events = S.events;
  if (S.gran === "layer") {
    // one stop per slab: consecutive events landing on the same (collapsed) node merge
    const stops = [];
    events.forEach((e, i) => {
      const next = events[i + 1];
      const same = next && next.batch === e.batch && next.kind === e.kind && next.call && e.call && sameNode(e, next);
      if (!same) stops.push(i);
    });
    return stops;
  }
  const stops = [];
  events.forEach((e, i) => {
    const next = events[i + 1];
    if (!next || next.batch !== e.batch || (S.gran === "phase" && next.phase !== e.phase)) stops.push(i);
  });
  return stops;
}

function sameNode(a, b) {
  if (!S.graph || S.graph.batch.index !== a.batch) return false;
  return nodeOfEvent(a) === nodeOfEvent(b);
}

function nextStop(direction) {
  if (direction > 0) return S.stops.find((i) => i > S.i) ?? S.events.length - 1;
  return [...S.stops].reverse().find((i) => i < S.i) ?? 0;
}

function go(i) {
  S.i = clamp(i, 0, S.events.length - 1);
  update();
}

function play() {
  if (S.i >= S.events.length - 1) S.i = 0;
  S.playing = true;
  setFollow(true); // playing re-engages the camera after you've panned around
  tick();
}

function pause() {
  S.playing = false;
  clearTimeout(S.timer);
}

function toggle() {
  S.playing ? pause() : play();
  update();
}

function tick() {
  clearTimeout(S.timer);
  if (!S.playing) return;
  update();
  if (S.i >= S.events.length - 1) {
    S.playing = false;
    return update();
  }
  const event = S.events[S.i];
  const coarse = S.gran !== "layer";
  const dwell = (coarse ? 2600 : event.kind === "data" || event.kind === "loss" || event.kind === "optimizer" ? 1800 : 900) / S.speed;
  S.timer = setTimeout(() => {
    S.i = nextStop(1);
    tick();
  }, dwell);
}

function shiftSample(d) {
  const batch = currentBatch();
  const n = followableSamples(batch).limit;
  store.set({ sample: ((store.get().sample ?? 0) + d + n) % n });
}

function setFollow(on) {
  S.follow = on;
  if (on && S.three) S.three.viewDir = VIEW_DIR;
  if (on && S.events.length && S.three) applyFocus();
  if (S.dom) S.dom.followBtn.classList.toggle("on", on);
}

function applyFocus() {
  const event = S.events[S.i];
  const active = nodeOfEvent(event);
  const graph = S.graph;
  if (!graph) return;
  const focus = active || (event.kind === "data" ? null : graph.nodes[graph.nodes.length - 1]);
  S.three.target.set((focus ? focus.x : -3) + LOOK_AHEAD, 0, 0);
  S.three.distance = followDistance();
}

function currentBatch() {
  const event = S.events[S.i];
  return event ? store.get().allBatches.find((b) => b.index === event.batch) : null;
}

// ---------------------------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------------------------
function buildDom(container) {
  const now = h("div", { class: "now" });
  const playBtn = h("button", { class: "tbtn play", title: "Play / pause (space)", onclick: toggle }, "▶");
  const gran = h(
    "div",
    { class: "seg" },
    GRANULARITIES.map(([k, l]) => h("button", { class: k === S.gran ? "on" : "", dataset: { gran: k }, onclick: () => ((S.gran = k), (S.stops = computeStops()), syncControls()) }, l)),
  );
  const speed = h("select", { class: "select", title: "Speed", onchange: (e) => (S.speed = Number(e.target.value)) }, SPEEDS.map((s) => h("option", { value: s, selected: s === S.speed }, `${s}×`)));
  const followBtn = h("button", { class: `chip ${S.follow ? "on" : ""}`, title: "Camera follows the active layer", onclick: () => setFollow(!S.follow) }, "follow");
  const sampleNav = h("span", { class: "sample-nav" });
  const viewport = h("div", { class: "s3-viewport" });
  const hud = h("div", { class: "s3-hud" });
  const track = h("input", {
    type: "range",
    class: "s3-track",
    min: -4,
    max: 10,
    step: 0.1,
    value: 0,
    title: "Scroll along the model (or swipe sideways / shift+scroll)",
    oninput: (e) => panTo(Number(e.target.value)),
  });
  const panel = h("div", { class: "s3-panel hidden" });
  const root = h(
    "div",
    { class: "s3" },
    h(
      "div",
      { class: "transport" },
      h(
        "div",
        { class: "transport-row" },
        h("button", { class: "tbtn", title: "Restart", onclick: () => go(0) }, "⏮"),
        h("button", { class: "tbtn", title: "Previous (←)", onclick: () => go(nextStop(-1)) }, "◀"),
        playBtn,
        h("button", { class: "tbtn", title: "Next (→)", onclick: () => go(nextStop(1)) }, "▶|"),
        now,
        h("div", { style: { flex: 1 } }),
        sampleNav,
        h("label", { class: "faint" }, "step by"),
        gran,
        speed,
        followBtn,
        h("button", { class: "chip", title: "Fit the whole model on screen", onclick: overview }, "overview"),
        h("button", { class: "chip", title: "Expand or collapse every repeated block", onclick: toggleExpandAll }, "blocks ⇄"),
      ),
    ),
    h("div", { class: "s3-body" }, viewport, panel),
    hud,
  );
  viewport.append(hud, h("div", { class: "s3-track-wrap" }, h("span", {}, "scroll"), track));
  clear(container, root);
  S.dom = { root, now, playBtn, gran, sampleNav, viewport, hud, panel, followBtn, track };
}

function syncControls() {
  for (const b of S.dom.gran.querySelectorAll("button")) b.classList.toggle("on", b.dataset.gran === S.gran);
}

function toggleExpandAll() {
  const graph = S.graph;
  if (!graph) return;
  const units = graph.nodes.filter((n) => n.unit);
  if (units.length) for (const n of units) S.expanded.add(unitKey(n.call));
  else S.expanded.clear();
  S.graphKey = null;
  S.stops = computeStops();
  update();
}

// ---------------------------------------------------------------------------------------------
// per-event update
// ---------------------------------------------------------------------------------------------
function update() {
  if (!S.dom || !S.events.length || !S.three) return;
  const state = store.get();
  const event = S.events[S.i];
  const batch = currentBatch();
  const { limit, total } = followableSamples(batch);
  const sample = Math.min(state.sample ?? 0, limit - 1);
  const key = `${batch.index}:${sample}:${[...S.expanded].sort().join(",")}`;
  if (S.graphKey !== key) {
    S.graph = buildGraph(batch);
    buildWorld(S.graph, sample);
    S.graphKey = key;
    S.stops = computeStops();
    if (!S.positioned) {
      placeCamera(-3 + LOOK_AHEAD + 2, followDistance(), true); // the input plus the first layers
      S.positioned = true;
    }
    S.dom.track.max = String(Math.max(1, S.graph.length));
  }
  S.dom.playBtn.textContent = S.playing ? "❚❚" : "▶";
  clear(
    S.dom.now,
    h("span", { class: "ph", style: { background: phaseColor(event.phase) } }, event.phase),
    h("b", {}, event.label),
    h("span", { class: "faint" }, `  ·  batch ${event.batch} · step ${event.step}`),
  );
  clear(
    S.dom.sampleNav,
    h("span", { class: "faint" }, "sample"),
    h("button", { onclick: () => shiftSample(-1) }, "‹"),
    h("b", {}, `#${sample}`),
    h("button", { onclick: () => shiftSample(1) }, "›"),
    limit < total ? h("span", { class: "faint" }, `of ${limit} stored`) : null,
  );
  applyEvent(event, batch, sample);
  renderPanel(batch, sample);
}

function applyEvent(event, batch, sample) {
  const graph = S.graph;
  const active = nodeOfEvent(event);
  const phaseRank = ["data", "forward", "loss", "backward", "optimizer"].indexOf(event.phase);
  const backDone = new Set();
  if (phaseRank >= 3) {
    for (let i = S.i; i >= 0 && S.events[i].batch === batch.index; i--) {
      const node = S.events[i].kind === "backward" ? nodeOfEvent(S.events[i]) : null;
      if (node) backDone.add(node);
    }
    if (event.phase === "optimizer") for (const n of graph.nodes) backDone.add(n);
  }
  const norms = graph.nodes.map((n) => n.info.meta?.grad_stats?.norm || 0).filter((v) => v > 0);
  const hiLog = Math.log10(Math.max(...norms, 1e-12));
  const loLog = Math.log10(Math.min(...norms, 1)) - 1;
  for (const node of graph.nodes) {
    const forwardDone = phaseRank > 1 || (event.kind === "forward" && active && node.index <= active.index);
    const backward = backDone.has(node);
    const isActive = node === active;
    const face = node.face;
    let opacity = 0.12;
    let glow = 0;
    if (backward) {
      const norm = node.info.meta?.grad_stats?.norm || 0;
      const level = norm ? clamp((Math.log10(norm) - loLog) / (hiLog - loLog || 1), 0.08, 1) : 0.05;
      opacity = 0.35 + 0.6 * level;
      glow = 0.25 + 0.75 * level;
    } else if (forwardDone) {
      opacity = 0.92;
      glow = 0.55;
    }
    if (isActive) glow = 1.6;
    face.opacity = opacity;
    face.userData.baseGlow = glow;
    face.userData.active = isActive;
    node.edgeLines.material.color.set(isActive ? (event.phase === "backward" ? 0x7cc4ff : 0xffb15c) : 0x5b6b85);
    const want = backward ? "grad" : forwardDone ? "value" : null;
    if (node.showing !== `${want}:${sample}`) {
      node.showing = `${want}:${sample}`;
      if (!want) {
        face.map = null;
        face.emissiveMap = null;
        face.color.set(0x2a3446);
        face.needsUpdate = true;
      } else {
        texture(node, sample, want === "grad").then((tex) => {
          if (node.showing !== `${want}:${sample}`) return;
          face.map = tex;
          face.emissiveMap = tex;
          face.color.set(tex ? 0xffffff : want === "grad" ? 0x2b6cb0 : 0xc2592b);
          face.needsUpdate = true;
        });
      }
    }
  }
  // particles: where data (or gradient) is flowing right now
  const pu = graph.particles.userData;
  pu.curves = [];
  pu.reverse = false;
  graph.particles.material.color.set(0xff9a3c);
  if (event.kind === "data" && graph.inputCurve) pu.curves = [graph.inputCurve];
  else if (event.kind === "forward" && active) {
    pu.curves = graph.edges.filter((e) => e.to === active).map((e) => e.curve);
    if (!pu.curves.length && active.index === 0 && graph.inputCurve) pu.curves = [graph.inputCurve];
  } else if (event.kind === "backward" && active) {
    pu.curves = graph.edges.filter((e) => e.from === active).map((e) => e.curve);
    pu.reverse = true;
    graph.particles.material.color.set(0x5ab8ff);
  } else if (event.kind === "loss" && graph.nodes.length) {
    const last = graph.nodes[graph.nodes.length - 1];
    pu.curves = [new THREE.LineCurve3(new THREE.Vector3(last.x, 0, 0), new THREE.Vector3(graph.length + 1.2, 0.8, 0))];
  }
  graph.particles.visible = pu.curves.length > 0;
  if (event.kind === "optimizer") S.three.flashUntil = performance.now() + 900;
  // prediction appears once the loss is computed
  const s = batch.samples?.[sample];
  if (phaseRank >= 2 && s) {
    const p = s.prediction || {};
    // a language model's prediction is a whole sequence: show only its last few words
    const words = p.name == null && p.text != null ? p.text.trim().split(/\s+/) : null;
    const predicted = words ? `${words.length > 4 ? "… " : ""}${words.slice(-4).join(" ")}` : `${p.name ?? p.id ?? "?"}${p.p != null ? ` ${Math.round(p.p * 100)}%` : ""}`;
    graph.output.userData.setText(predicted, p.correct === false ? "#ff8a8a" : p.correct ? "#8af0b8" : "#ffffff", `loss ${fmtNum(s.loss, 3)}`);
  } else graph.output.userData.setText("");
  if (S.follow) {
    const focus = active || (event.kind === "data" ? null : graph.nodes[graph.nodes.length - 1]);
    S.three.target.set((focus ? focus.x : -3) + LOOK_AHEAD, 0, 0);
    S.three.distance = followDistance();
  }
  S.hudText = event.kind === "loss" ? `batch loss ${fmtNum(lossValue(batch), 4)}` : "";
  S.dom.hud.textContent = S.hudText || "drag to orbit · scroll to zoom · swipe sideways or shift+scroll to move along · click a slab to inspect · ▸ = collapsed block";
}

// ---------------------------------------------------------------------------------------------
// render loop, picking, side panel
// ---------------------------------------------------------------------------------------------
function startLoop() {
  if (S.three.raf) return;
  const frame = () => {
    S.three.raf = requestAnimationFrame(frame);
    const t = S.three.clock.getElapsedTime();
    const { camera, controls, renderer, scene, target } = S.three;
    if (S.graph) {
      const flash = performance.now() < S.three.flashUntil;
      for (const node of S.graph.nodes) {
        const f = node.face;
        const base = f.userData.baseGlow || 0;
        f.emissiveIntensity = f.userData.active ? base * (0.75 + 0.35 * Math.sin(t * 6)) : base;
        if (flash) f.emissive.setRGB(0.4, 1, 0.6);
        else f.emissive.setRGB(1, 1, 1);
        const s = f.userData.active ? 1.06 + 0.03 * Math.sin(t * 6) : 1;
        node.mesh.scale.set(1, s, s);
      }
      animateParticles(t);
    }
    if (!S.follow && performance.now() < (S.three.glideUntil || 0)) {
      controls.target.lerp(target, 0.08);
      camera.position.lerp(target.clone().addScaledVector(S.three.viewDir, S.three.distance), 0.08);
    }
    if (S.follow) {
      controls.target.lerp(target, 0.06);
      camera.position.lerp(target.clone().addScaledVector(S.three.viewDir, S.three.distance), 0.04);
    }
    controls.update();
    if (S.dom.track && document.activeElement !== S.dom.track) S.dom.track.value = String(controls.target.x);
    renderer.render(scene, camera);
  };
  frame();
}

function animateParticles(t) {
  const points = S.graph.particles;
  const { curves, reverse, seeds, jitter } = points.userData;
  if (!curves.length) return;
  const pos = points.geometry.attributes.position.array;
  const v = new THREE.Vector3();
  for (let i = 0; i < seeds.length; i++) {
    const curve = curves[i % curves.length];
    let u = (seeds[i] + t * 0.45 * S.speed) % 1;
    if (reverse) u = 1 - u;
    curve.getPoint(u, v);
    pos[i * 3] = v.x;
    pos[i * 3 + 1] = v.y + jitter[i * 2];
    pos[i * 3 + 2] = v.z + jitter[i * 2 + 1];
  }
  points.geometry.attributes.position.needsUpdate = true;
}

/** Move the view along the tunnel, keeping the current viewing angle and zoom. */
function panTo(x) {
  const { camera, controls } = S.three;
  const lo = -4;
  const hi = (S.graph?.length || 0) + 2;
  const nx = clamp(x, lo, hi);
  const dx = nx - controls.target.x;
  controls.target.x += dx;
  camera.position.x += dx;
  S.three.glideUntil = 0; // an unfinished 'overview' glide would pull the view back
  setFollow(false);
}

/** Camera distance while following, backed off on narrow views so the same span stays visible. */
function followDistance() {
  const aspect = S.three?.camera.aspect || 1.6;
  return FOLLOW_DISTANCE * Math.max(1, 1.7 / aspect);
}

/** Put the camera at the current viewing angle looking at x from `distance`. */
function placeCamera(x, distance, immediate = false) {
  const { camera, controls, target } = S.three;
  target.set(x, 0, 0);
  S.three.distance = distance;
  if (immediate) {
    controls.target.copy(target);
    camera.position.copy(target.clone().addScaledVector(S.three.viewDir, distance));
  }
}

/** Fit the whole model, input to output, on screen, seen almost from the front. */
function overview() {
  if (!S.graph) return;
  const { camera } = S.three;
  const xmin = -4.5;
  const xmax = S.graph.length + 2.5;
  const vfov = (camera.fov * Math.PI) / 180;
  const hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
  const distance = clamp(((xmax - xmin) / 2 / Math.tan(hfov / 2)) * 1.08, 8, 600);
  setFollow(false);
  S.three.viewDir = OVERVIEW_DIR;
  placeCamera((xmin + xmax) / 2, distance);
  S.three.glideUntil = performance.now() + 1600;
}

function onClick(e) {
  if (!S.graph) return;
  const rect = S.three.renderer.domElement.getBoundingClientRect();
  const pointer = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  const ray = new THREE.Raycaster();
  ray.setFromCamera(pointer, S.three.camera);
  const hit = ray.intersectObjects(S.graph.nodes.map((n) => n.mesh), false)[0];
  if (!hit) return;
  const node = hit.object.userData.node;
  if (node.unit && S.selected === node.call.id) {
    S.expanded.add(unitKey(node.call)); // second click on a collapsed block opens it up
    S.graphKey = null;
  }
  S.selected = node.call.id;
  setFollow(false);
  panTo(node.x);
  update();
}

function renderPanel(batch, sample) {
  const panel = S.dom.panel;
  const call = S.selected != null ? index(batch).byId.get(S.selected) : null;
  if (!call) {
    panel.classList.add("hidden");
    return clear(panel);
  }
  panel.classList.remove("hidden");
  const event = S.events[S.i];
  const node = S.graph.nodeOfCall.get(call.id);
  const grad = (event.phase === "backward" || event.phase === "optimizer") && node?.showing?.startsWith("grad");
  const panelKey = `${call.id}:${sample}:${grad}`;
  if (panel.dataset.key === panelKey) return;
  panel.dataset.key = panelKey;
  clear(
    panel,
    h(
      "div",
      { style: { display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" } },
      h("h2", { style: { flex: 1, margin: 0, fontSize: "15px" } }, grad ? `∇ ${call.name}` : call.name),
      node?.unit ? h("button", { class: "chip small", onclick: () => (S.expanded.add(unitKey(call)), (S.graphKey = null), update()) }, "expand block") : null,
      h("button", { class: "chip small", onclick: () => ((S.selected = null), (panel.dataset.key = ""), update()) }, "✕"),
    ),
    h("div", { class: "faint", style: { marginBottom: "8px" } }, `${call.kind === "function" ? "function" : call.cls} · forward ${call.ms != null ? `${call.ms.toFixed(2)} ms` : "—"}`),
    layerView({ run: store.get().runId, batch, call, sample, grad, state: store.get() }),
  );
}
