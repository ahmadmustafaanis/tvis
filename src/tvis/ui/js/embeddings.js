// Embeddings page: PCA of each layer's per-sample representation (do classes separate with depth?)
// and of embedding tables (which tokens sit together?).

import { api } from "./api.js";
import { categorical, legend, pct, scatter } from "./charts.js";
import { firstOutput } from "./model.js";
import { store } from "./store.js";
import { clear, fmtNum, h } from "./util.js";

const E = { layer: null, table: null };

export function representationLayers(batch) {
  const seen = new Set();
  const layers = [];
  for (const call of batch.calls) {
    if (call.phase !== "forward" || (call.kind !== "module" && call.kind !== "function")) continue;
    const ref = firstOutput(call);
    const meta = ref ? batch.tensors[ref.tid] : null;
    const path = call.module_path || call.name;
    if (!meta || meta.batch_dim?.block !== 1 || meta.stored === "none" || seen.has(path) || meta.shape.length < 2) continue;
    seen.add(path);
    layers.push({ path, call, shape: meta.shape });
  }
  return layers;
}

export function embeddingTables(state) {
  const model = state.meta?.models?.[0];
  const step = state.steps[0];
  if (!model || !step) return [];
  return model.modules.filter((m) => m.cls === "Embedding").map((m) => `${m.path}.weight`).filter((name) => step.params.some((p) => p.name === name));
}

export function renderEmbeddings(container) {
  const state = store.get();
  const batch = state.allBatches?.[0];
  if (!batch) return clear(container, h("div", { class: "loading" }, "loading…"));
  const layers = representationLayers(batch);
  const tables = embeddingTables(state);
  if (!layers.length && !tables.length) return clear(container, h("div", { class: "empty" }, "No per-sample representations or embedding tables were recorded."));
  if (!layers.some((l) => l.path === E.layer)) E.layer = layers[Math.max(0, layers.length - 2)]?.path;
  if (!tables.includes(E.table)) E.table = tables[0];
  const repSlot = h("div", {}, h("div", { class: "loading" }, "projecting…"));
  const tableSlot = h("div", {}, h("div", { class: "loading" }, "projecting…"));
  const layerIndex = layers.findIndex((l) => l.path === E.layer);
  clear(
    container,
    h(
      "div",
      { class: "page-wrap stack-gap" },
      h("div", { class: "page-head" }, h("h1", {}, "Embeddings"), h("span", { class: "lede" }, "2-D PCA projections. Points that sit together are represented alike by the model.")),
      layers.length
        ? h(
            "div",
            { class: "card" },
            h("h3", {}, "Sample representations by layer"),
            h("div", { class: "hint" }, `Every captured sample, coloured by its true label (hollow = misclassified). Slide through the layers to see whether classes separate with depth.`),
            h(
              "div",
              { class: "controls" },
              h("label", {}, "layer", h("input", { type: "range", min: 0, max: layers.length - 1, value: layerIndex, oninput: (e) => ((E.layer = layers[Number(e.target.value)].path), renderEmbeddings(container)) })),
              h("select", { class: "select", onchange: (e) => ((E.layer = e.target.value), renderEmbeddings(container)) }, layers.map((l) => h("option", { value: l.path, selected: l.path === E.layer }, `${l.path}  [${l.shape.join(", ")}]`))),
            ),
            repSlot,
          )
        : null,
      tables.length
        ? h(
            "div",
            { class: "card" },
            h("h3", {}, "Embedding tables"),
            h("div", { class: "hint" }, "Rows of the embedding matrix before the first captured step, labelled with their tokens."),
            tables.length > 1 ? h("div", { class: "controls" }, h("select", { class: "select", onchange: (e) => ((E.table = e.target.value), renderEmbeddings(container)) }, tables.map((t) => h("option", { value: t, selected: t === E.table }, t)))) : null,
            tableSlot,
          )
        : null,
    ),
  );
  if (E.layer) {
    api("projection", { run: state.runId, path: E.layer })
      .then((res) => {
        const colors = categorical(res.points.map((p) => p.label ?? "?"));
        clear(
          repSlot,
          scatter(res.points, {
            color: (p) => colors.color(p.label ?? "?"),
            hollow: (p) => p.correct === false,
            title: (p) => `batch ${p.batch} · sample #${p.position} · ${p.label ?? "?"}${p.loss != null ? ` · loss ${fmtNum(p.loss, 3)}` : ""}`,
            onClick: (p) => store.set({ page: "replay", sample: p.position }),
          }),
          legend(colors.entries.slice(0, 12), [h("span", { class: "faint" }, `${res.points.length} samples · ${res.dim}-d → 2-d · explains ${pct(res.explained[0] + res.explained[1])} of variance`)]),
        );
      })
      .catch((err) => clear(repSlot, h("div", { class: "err" }, err.message)));
  }
  if (E.table) {
    api("weight_projection", { run: state.runId, step: 0, param: E.table })
      .then((res) =>
        clear(
          tableSlot,
          scatter(res.points, { radius: 3.5, title: (p) => `${p.label} (id ${p.id})`, text: res.points.length <= 120 ? (p) => p.label : null }),
          legend([], [h("span", { class: "faint" }, `${res.rows} rows · ${res.dim}-d · explains ${pct(res.explained[0] + res.explained[1])} of variance`)]),
        ),
      )
      .catch((err) => clear(tableSlot, h("div", { class: "err" }, err.message)));
  }
}
