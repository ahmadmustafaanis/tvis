// Left panel: the call tree of a batch (or the batches of a step).

import { firstOutput, hasProblem, index, matchesFilter, tensorMeta, visibleChildren } from "./model.js";
import { hiddenKey, saveHidden, store } from "./store.js";
import { badge, clear, fmtMs, fmtShape, h, icon, phaseColor, tooltip } from "./util.js";

const KIND_FILTERS = [
  ["data", "Data"],
  ["module", "Modules"],
  ["function", "Functions"],
  ["loss", "Loss"],
  ["backward", "Backward"],
  ["optimizer", "Optimizer"],
];

export function renderKindFilters(container) {
  const state = store.get();
  clear(
    container,
    KIND_FILTERS.map(([kind, label]) =>
      h(
        "button",
        {
          class: `chip small ${state.kinds.has(kind) ? "on" : ""}`,
          onclick: () => {
            const kinds = new Set(state.kinds);
            kinds.has(kind) ? kinds.delete(kind) : kinds.add(kind);
            store.set({ kinds });
          },
        },
        label,
      ),
    ),
  );
}

function defaultExpanded(call, depth) {
  if (call.group) return false;
  if (call.kind === "data" || call.kind === "sample" || call.kind === "pipeline") return false;
  return depth < 6;
}

export function renderTree(container) {
  const state = store.get();
  if (state.mode === "step") return renderStepList(container);
  const batch = state.batch;
  if (!batch) return clear(container, h("div", { class: "loading" }, "loading…"));
  const idx = index(batch);
  const rows = [];
  const walk = (nodes, depth) => {
    for (const node of nodes) {
      if (!node.group && !matchesFilter(batch, node, state.filter)) continue;
      const kids = node.group ? node.calls : visibleChildren(batch, node.id, state);
      const expanded = state.filter ? true : state.expanded.get(node.id) ?? defaultExpanded(node, depth);
      rows.push(node.group ? groupRow(node, depth, expanded) : callRow(batch, idx, node, depth, kids.length > 0, expanded));
      if (expanded && kids.length) walk(kids, depth + 1);
    }
  };
  walk(visibleChildren(batch, null, state), 0);
  clear(container, rows.length ? rows : h("div", { class: "empty" }, "No calls match."));
}

function toggle(id, expanded) {
  const next = new Map(store.get().expanded);
  next.set(id, !expanded);
  store.set({ expanded: next });
}

function caret(hasKids, expanded, id) {
  return h(
    "span",
    {
      class: "caret",
      onclick: (e) => {
        e.stopPropagation();
        if (hasKids) toggle(id, expanded);
      },
    },
    hasKids ? (expanded ? "▾" : "▸") : "",
  );
}

function groupRow(group, depth, expanded) {
  const first = group.calls[0];
  const total = group.calls.reduce((s, c) => s + (c.ms || 0), 0);
  return h(
    "div",
    { class: "row group-row", style: { paddingLeft: `${depth * 14 + 4}px` }, onclick: () => toggle(group.id, expanded) },
    h("span", { class: "lead" }, caret(true, expanded, group.id), badge(first.kind)),
    h("span", { class: "name" }, `${group.key.split(":")[1]} ×${group.calls.length}`),
    h("span", { class: "t" }, fmtMs(total)),
  );
}

function callRow(batch, idx, call, depth, hasKids, expanded) {
  const state = store.get();
  const out = firstOutput(call);
  const meta = tensorMeta(batch, out);
  const selected = state.selection?.call === call.id;
  const bar = h(
    "span",
    { class: "tbar" },
    h("i", { style: { width: `${Math.max(2, ((call.ms || 0) / idx.maxMs) * 100)}%`, background: phaseColor("forward") } }),
    call.bwd_ms != null
      ? h("i", { style: { width: `${Math.max(2, (call.bwd_ms / idx.maxMs) * 100)}%`, background: phaseColor("backward") } })
      : h("i", { style: { width: "0" } }),
  );
  const row = h(
    "div",
    {
      class: `row ${selected ? "sel" : ""}`,
      style: { paddingLeft: `${depth * 14 + 4}px` },
      onclick: () => store.set({ selection: { call: call.id } }),
      ondblclick: () => hasKids && toggle(call.id, expanded),
      dataset: { id: call.id },
    },
    h("span", { class: "lead" }, caret(hasKids, expanded, call.id), badge(call.kind)),
    h(
      "span",
      { class: "name" },
      call.name,
      meta ? h("span", { class: "shape" }, fmtShape(meta.shape)) : null,
      hasProblem(batch, call) ? h("span", { class: "warn-dot", title: "NaN/Inf in a value or gradient" }) : null,
    ),
    h(
      "span",
      { class: "t" },
      call.ms != null ? fmtMs(call.ms) : "",
      bar,
      h(
        "button",
        {
          class: "hide-btn",
          title: "Hide calls like this (children stay visible)",
          onclick: (e) => {
            e.stopPropagation();
            const hidden = new Set(state.hidden);
            hidden.add(hiddenKey(call));
            saveHidden(state.meta?.project_dir, hidden);
            store.set({ hidden });
          },
        },
        icon("hide"),
      ),
    ),
  );
  tooltip(row, () => {
    const parts = [`${call.kind} ${call.qualname || call.cls || call.name}`];
    if (call.ms != null) parts.push(`forward ${fmtMs(call.ms)}`);
    if (call.bwd_ms != null) parts.push(`backward ≈${fmtMs(call.bwd_ms)}`);
    if (call.file) parts.push(`${call.file}:${call.line}`);
    return parts.join(" · ");
  });
  return row;
}

function renderStepList(container) {
  const state = store.get();
  const step = state.steps[state.stepIndex];
  if (!step) return clear(container, h("div", { class: "empty" }, "No steps captured."));
  const rows = state.stepBatches.map((batch) =>
    h(
      "div",
      {
        class: "row",
        style: { paddingLeft: "8px" },
        onclick: () => store.set({ mode: "batch", batchIndex: batch.index, selection: null }),
      },
      h("span", { class: "lead" }, h("span", { class: "caret" }, "▸"), badge("data")),
      h("span", { class: "name" }, `batch ${batch.index}`, h("span", { class: "shape" }, `${batch.batch_size ?? "?"} samples`)),
      h("span", { class: "t" }, fmtMs(batch.timing?.total_ms)),
    ),
  );
  const params = step.params.map((p) =>
    h(
      "div",
      {
        class: `row ${state.selection?.param === p.name ? "sel" : ""}`,
        style: { paddingLeft: "8px" },
        onclick: () => store.set({ selection: { param: p.name } }),
      },
      h("span", { class: "lead" }, h("span", { class: "caret" }), badge("optimizer")),
      h("span", { class: "name" }, p.name, h("span", { class: "shape" }, fmtShape(p.shape))),
      h("span", { class: "t" }, p.update_ratio != null ? p.update_ratio.toExponential(1) : ""),
    ),
  );
  clear(container, rows, h("div", { class: "subhead", style: { padding: "0 12px" } }, "Parameters"), params);
}
