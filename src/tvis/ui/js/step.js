// Step view: the batches of an optimizer step and every parameter's update.

import { lossValue } from "./model.js";
import { store } from "./store.js";
import { clear, fmtMs, fmtNum, fmtShape, h, phaseColor } from "./util.js";

export function renderStep(container) {
  const state = store.get();
  const step = state.steps[state.stepIndex];
  if (!step) return clear(container, h("div", { class: "empty" }, "No steps captured."));
  const batches = state.stepBatches;
  const total = batches.reduce((s, b) => s + (b.timing?.total_ms || 0), 0);
  const kpi = (label, value, sub) => h("div", { class: "kpi" }, h("div", { class: "label" }, label), h("div", { class: "value" }, value), sub ? h("div", { class: "sub" }, sub) : null);
  const losses = batches.map(lossValue).filter((v) => v != null);
  const ratios = step.params.map((p) => p.update_ratio).filter((v) => v != null);
  const median = ratios.length ? [...ratios].sort((a, b) => a - b)[Math.floor(ratios.length / 2)] : null;
  clear(
    container,
    h(
      "div",
      { class: "view" },
      h(
        "section",
        {},
        h("div", { class: "kpis" }, [
          kpi("Step", `#${step.index}`, `${batches.length} micro-batch${batches.length === 1 ? "" : "es"}`),
          kpi("Step time", fmtMs(total), "device time, capture overhead removed"),
          kpi("Loss", losses.length ? fmtNum(losses.reduce((a, b) => a + b, 0) / losses.length) : "—", losses.length > 1 ? "mean over micro-batches" : ""),
          kpi("Median ‖Δw‖/‖w‖", median != null ? median.toExponential(1) : "—", "healthy ≈ 1e-3"),
        ]),
      ),
      h("section", {}, h("h3", {}, "Micro-batches"), timeBars(batches)),
      h("section", {}, h("h3", {}, "Parameter updates"), paramTable(step, state)),
    ),
  );
}

function timeBars(batches) {
  const phases = ["data", "forward", "loss", "backward", "post_backward", "optimizer"];
  const max = Math.max(...batches.map((b) => b.timing?.total_ms || 0), 1e-9);
  return h(
    "div",
    {},
    batches.map((b) =>
      h(
        "div",
        { class: "row", style: { gridTemplateColumns: "90px 1fr 80px", paddingLeft: 0 }, onclick: () => store.set({ mode: "batch", batchIndex: b.index }) },
        h("span", { class: "muted" }, `batch ${b.index}`),
        h(
          "div",
          { class: "phasebar", style: { padding: 0, border: 0, minHeight: 0, background: "transparent" } },
          h(
            "div",
            { class: "bar", style: { width: `${((b.timing?.total_ms || 0) / max) * 100}%` } },
            phases.map((p) => (b.timing?.phases_ms?.[p] ? h("span", { style: { width: `${(b.timing.phases_ms[p] / b.timing.total_ms) * 100}%`, background: phaseColor(p) }, title: `${p} ${fmtMs(b.timing.phases_ms[p])}` }) : null)),
          ),
        ),
        h("span", { class: "mono", style: { textAlign: "right" } }, fmtMs(b.timing?.total_ms)),
      ),
    ),
  );
}

function ratioClass(r) {
  if (r == null) return "faint";
  if (r < 1e-5 || r > 1e-1) return "bad";
  if (r < 1e-4 || r > 1e-2) return "warn";
  return "good";
}

function paramTable(step, state) {
  return h(
    "table",
    { class: "grid" },
    h("thead", {}, h("tr", {}, ["parameter", "shape", "‖w‖", "‖grad‖", "‖Δw‖", "‖Δw‖/‖w‖"].map((t, i) => h("th", { class: i > 1 ? "num" : "" }, t)))),
    h(
      "tbody",
      {},
      step.params.map((p) =>
        h(
          "tr",
          { class: state.selection?.param === p.name ? "sel" : "", onclick: () => store.set({ selection: { param: p.name } }) },
          h("td", { class: "mono" }, p.name),
          h("td", { class: "mono faint" }, fmtShape(p.shape)),
          h("td", { class: "num" }, fmtNum(p.weight_norm)),
          h("td", { class: "num" }, fmtNum(p.grad_norm)),
          h("td", { class: "num" }, fmtNum(p.update_norm)),
          h("td", { class: `num ${ratioClass(p.update_ratio)}` }, p.update_ratio != null ? p.update_ratio.toExponential(2) : "—"),
        ),
      ),
    ),
  );
}
