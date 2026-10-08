// Single app state + subscribers. Views re-render from state; nothing else holds UI state.

const state = {
  runs: [],
  runId: null,
  meta: null,
  steps: [],
  page: "replay", // replay | loss | weights | attention | gradcam | embeddings | explore
  mode: "batch", // "batch" | "step"
  batchIndex: 0,
  stepIndex: 0,
  batch: null, // batch document for batchIndex
  stepBatches: [], // batch documents of the selected step
  selection: null, // {call: id} | {param: name}
  sample: null, // followed sample position, or null
  tab: "overview",
  filter: "",
  kinds: new Set(["data", "module", "function", "loss", "backward", "optimizer"]),
  hidden: new Set(),
  expanded: new Map(),
  sourceFile: null,
  sourceLine: null,
  error: null,
};

const subscribers = new Set();

export const store = {
  get: () => state,
  set(patch) {
    Object.assign(state, patch);
    for (const fn of subscribers) fn(state, patch);
  },
  subscribe(fn) {
    subscribers.add(fn);
    return () => subscribers.delete(fn);
  },
};

export function hiddenKey(call) {
  if (call.kind === "module") return `module:${call.cls}`;
  return `${call.kind}:${call.qualname || call.name}`;
}

export function loadHidden(projectDir) {
  try {
    return new Set(JSON.parse(localStorage.getItem(`tvis.hidden.${projectDir}`) || "[]"));
  } catch {
    return new Set();
  }
}

export function saveHidden(projectDir, hidden) {
  try {
    localStorage.setItem(`tvis.hidden.${projectDir}`, JSON.stringify([...hidden]));
  } catch {
    /* storage unavailable: hiding still works for this session */
  }
}
