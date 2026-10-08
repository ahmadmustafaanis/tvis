// Layers tab: every module/function call of the forward and loss as a sortable table with health flags.

import { firstOutput } from "./model.js";
import { store } from "./store.js";
import { badge, clear, fmtMs, fmtNum, fmtShape, h } from "./util.js";

let sortKey = "order";
let sortDir = 1;

const COLUMNS = [
  ["order", "#", (r) => r.order, true],
  ["name", "call", (r) => r.call.name, false],
  ["shape", "output", (r) => r.meta?.numel ?? 0, false],
  ["std", "out σ", (r) => r.meta?.stats?.std ?? -1, true],
  ["absmean", "out |x|", (r) => r.meta?.stats?.abs_mean ?? -1, true],
  ["zeros", "zeros", (r) => r.meta?.stats?.zero_frac ?? -1, true],
  ["grad", "‖grad‖", (r) => r.meta?.grad_stats?.norm ?? -1, true],
  ["ms", "fwd", (r) => r.call.ms ?? -1, true],
  ["bwd", "bwd ≈", (r) => r.call.bwd_ms ?? -1, true],
];

export function renderLayers(container) {
  const state = store.get();
  const batch = state.batch;
  if (!batch) return clear(container, h("div", { class: "loading" }, "loading…"));
  const rows = batch.calls
    .filter((c) => (c.kind === "module" || c.kind === "function" || c.kind === "loss") && (c.phase === "forward" || c.phase === "loss"))
    .map((call, order) => {
      const ref = firstOutput(call);
      return { call, order, meta: ref ? batch.tensors[ref.tid] : null };
    });
  const col = COLUMNS.find((c) => c[0] === sortKey) || COLUMNS[0];
  rows.sort((a, b) => {
    const va = col[2](a);
    const vb = col[2](b);
    return (va < vb ? -1 : va > vb ? 1 : 0) * sortDir;
  });
  const head = h(
    "tr",
    {},
    COLUMNS.map(([key, label, , numeric]) =>
      h(
        "th",
        {
          class: numeric ? "num" : "",
          onclick: () => {
            sortDir = sortKey === key ? -sortDir : key === "order" || key === "name" ? 1 : -1;
            sortKey = key;
            renderLayers(container);
          },
        },
        label,
        sortKey === key ? (sortDir > 0 ? " ↑" : " ↓") : "",
      ),
    ),
  );
  const body = rows.map(({ call, order, meta }) => {
    const s = meta?.stats;
    const g = meta?.grad_stats;
    const bad = (x) => x && (x.nan || x.inf);
    return h(
      "tr",
      { class: state.selection?.call === call.id ? "sel" : "", onclick: () => store.set({ selection: { call: call.id } }) },
      h("td", { class: "num faint" }, order),
      h("td", {}, badge(call.kind), h("span", { style: { paddingLeft: `${call.depth * 8}px` } }, call.name)),
      h("td", { class: "mono" }, meta ? fmtShape(meta.shape) : "—"),
      h("td", { class: `num ${bad(s) ? "bad" : ""}` }, bad(s) ? "NaN/Inf" : fmtNum(s?.std)),
      h("td", { class: "num" }, fmtNum(s?.abs_mean)),
      h("td", { class: `num ${s?.zero_frac > 0.9 ? "warn" : ""}`, title: s?.zero_frac > 0.9 ? "mostly zeros (dead units?)" : "" }, s?.zero_frac != null ? `${(s.zero_frac * 100).toFixed(0)}%` : "—"),
      h("td", { class: `num ${bad(g) ? "bad" : g && g.norm < 1e-8 ? "warn" : ""}`, title: g && g.norm < 1e-8 ? "vanishing gradient" : "" }, bad(g) ? "NaN/Inf" : fmtNum(g?.norm)),
      h("td", { class: "num" }, fmtMs(call.ms)),
      h("td", { class: "num" }, call.bwd_ms != null ? fmtMs(call.bwd_ms) : "—"),
    );
  });
  clear(container, h("div", { class: "view" }, h("table", { class: "grid" }, h("thead", {}, head), h("tbody", {}, body))));
}
