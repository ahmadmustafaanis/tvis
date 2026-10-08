// Derived views over a batch document: call index, children, tensor lookup, visible tree.

import { hiddenKey } from "./store.js";

const indexCache = new WeakMap();

export function index(batch) {
  if (!batch) return null;
  let idx = indexCache.get(batch);
  if (idx) return idx;
  const byId = new Map();
  const children = new Map();
  for (const call of batch.calls) {
    byId.set(call.id, call);
    const key = call.parent ?? null;
    if (!children.has(key)) children.set(key, []);
    children.get(key).push(call);
  }
  for (const call of batch.calls) {
    if (call.parent != null && !byId.has(call.parent)) children.get(null).push(call);
  }
  idx = { byId, children, maxMs: Math.max(1e-9, ...batch.calls.map((c) => Math.max(c.ms || 0, c.bwd_ms || 0))) };
  indexCache.set(batch, idx);
  return idx;
}

export function tensorRefs(value, out = []) {
  if (!value) return out;
  if (value.kind === "tensor" || value.kind === "image") out.push(value);
  else if (value.kind === "seq") value.items.forEach((v) => tensorRefs(v, out));
  else if (value.kind === "map") Object.values(value.items).forEach((v) => tensorRefs(v, out));
  return out;
}

export function firstOutput(call) {
  for (const output of call.outputs || []) {
    const ref = tensorRefs(output.value)[0];
    if (ref) return ref;
  }
  return null;
}

export function tensorMeta(batch, ref) {
  return ref ? batch.tensors[ref.tid] : null;
}

export function hasProblem(batch, call) {
  for (const entry of [...(call.inputs || []), ...(call.outputs || [])]) {
    for (const ref of tensorRefs(entry.value)) {
      const meta = batch.tensors[ref.tid];
      if (!meta) continue;
      if (meta.stats?.nan || meta.stats?.inf || meta.grad_stats?.nan || meta.grad_stats?.inf) return true;
    }
  }
  return false;
}

const GROUP_MIN = 3;
const groupKey = (call) => `${call.kind}:${call.name.replace(/\[[^\]]*\]$/, "[·]").replace(/\.\d+$/, ".#")}`;

/**
 * The tree as displayed: hidden calls are removed (their children are promoted), kind filters
 * apply, and runs of >= 3 similar consecutive siblings collapse into a group node.
 */
export function visibleChildren(batch, parentId, state) {
  const idx = index(batch);
  const out = [];
  for (const call of idx.children.get(parentId) || []) {
    if (state.hidden.has(hiddenKey(call)) || !kindVisible(call, state.kinds)) {
      out.push(...visibleChildren(batch, call.id, state));
    } else {
      out.push(call);
    }
  }
  return group(out);
}

function kindVisible(call, kinds) {
  const family = { sample: "data", pipeline: "data", transform: "data", tokenize: "data" }[call.kind] || call.kind;
  return kinds.has(family);
}

function group(calls) {
  const result = [];
  let i = 0;
  while (i < calls.length) {
    let j = i + 1;
    const key = groupKey(calls[i]);
    while (j < calls.length && groupKey(calls[j]) === key) j++;
    if (j - i >= GROUP_MIN && calls[i].kind !== "module") {
      result.push({ group: true, id: `g${calls[i].id}`, key, calls: calls.slice(i, j) });
    } else {
      result.push(...calls.slice(i, j));
    }
    i = j;
  }
  return result;
}

export function matchesFilter(batch, call, filter) {
  if (!filter) return true;
  const f = filter.toLowerCase();
  const self = [call.name, call.qualname, call.cls, call.module_path].some((s) => s && s.toLowerCase().includes(f));
  if (self) return true;
  return (index(batch).children.get(call.id) || []).some((c) => matchesFilter(batch, c, filter));
}

export function lossValue(batch) {
  const loss = [...batch.calls].reverse().find((c) => c.kind === "loss");
  const ref = loss && firstOutput(loss);
  return ref ? batch.tensors[ref.tid]?.stats?.mean : null;
}

export function isImageShape(shape) {
  return shape && shape.length === 3 && (shape[0] === 3 || shape[0] === 1) && shape[1] > 1 && shape[2] > 1;
}

export function sampleShape(meta) {
  const bd = meta?.batch_dim;
  if (!bd) return null;
  return bd.block === 1 ? meta.shape.slice(1) : [bd.block, ...meta.shape.slice(1)];
}

/**
 * How many samples of a batch can be followed: tensors larger than --max-elems keep only their
 * first rows, so following sample i only works while i is below the smallest stored row count.
 */
export function followableSamples(batch) {
  const total = batch?.samples?.length || batch?.batch_size || 1;
  let limit = total;
  for (const meta of Object.values(batch?.tensors || {})) {
    if (!meta.batch_dim || meta.stored !== "rows" || meta.stored_rows == null) continue;
    limit = Math.min(limit, Math.floor(meta.stored_rows / (meta.batch_dim.block || 1)));
  }
  return { total, limit: Math.max(1, limit) };
}
