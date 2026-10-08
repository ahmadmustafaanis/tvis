// Run overview: what was captured, the model, the data, and per-batch loss/time.

import { lossValue } from "./model.js";
import { store } from "./store.js";
import { clear, fmtInt, fmtMs, fmtNum, h, phaseColor } from "./util.js";

export function renderOverview(container) {
  const { meta, steps, batch } = store.get();
  if (!meta) return clear(container, h("div", { class: "loading" }, "loading…"));
  const model = meta.models?.[0];
  const data = meta.data || {};
  const kpi = (label, value, sub) => h("div", { class: "kpi" }, h("div", { class: "label" }, label), h("div", { class: "value" }, value), sub ? h("div", { class: "sub" }, sub) : null);
  const notices = (meta.notices || []).map((n) => h("div", { class: `notice ${n.where === "capture value" ? "bad" : ""}` }, h("b", {}, n.where), ": ", n.message, n.count > 1 ? h("span", { class: "faint" }, ` (×${n.count})`) : null));
  if (meta.status === "error") notices.unshift(h("div", { class: "notice bad" }, h("b", {}, "The script failed. "), h("pre", { class: "mono", style: { whiteSpace: "pre-wrap", margin: "6px 0 0" } }, meta.error || "")));

  clear(
    container,
    h(
      "div",
      { class: "view" },
      notices.length ? h("section", {}, notices) : null,
      h(
        "section",
        {},
        h("div", { class: "kpis" }, [
          kpi("Steps", `${meta.steps_captured}/${meta.steps_requested}`, `${meta.batches_captured} batches`),
          model ? kpi("Model", model.cls, `${fmtInt(model.n_params)} params · ${model.device} · ${model.dtype}`) : null,
          data.dataset ? kpi("Dataset", data.dataset.cls, `${fmtInt(data.dataset.len)} samples${data.loader ? ` · batch ${data.loader.batch_size}` : ""}`) : null,
          kpi("Clock", meta.clock === "cuda_event" ? "CUDA events" : "host clock", meta.env?.device || "CPU"),
          kpi("Status", meta.status, meta.host),
        ]),
      ),
      h("section", {}, h("h3", {}, "Batches"), batchesTable(steps, store.get().allBatches || (batch ? [batch] : []))),
      model ? h("section", {}, h("h3", {}, "Model"), moduleTable(model)) : null,
      h(
        "section",
        {},
        h("h3", {}, "Run"),
        h(
          "dl",
          { class: "insp kv", style: { padding: 0 } },
          [
            ["script", meta.script],
            ["arguments", (meta.argv || []).join(" ") || "—"],
            ["project", meta.project_dir],
            ["python / torch", `${meta.env?.python} / ${meta.env?.torch}${meta.env?.cuda ? ` (CUDA ${meta.env.cuda})` : ""}`],
            ["started", meta.started_at],
            ["normalisation", data.normalize ? `mean ${data.normalize.mean.map((v) => fmtNum(v, 3)).join(", ")} · std ${data.normalize.std.map((v) => fmtNum(v, 3)).join(", ")}` : "—"],
          ].flatMap(([k, v]) => [h("dt", {}, k), h("dd", { class: "mono" }, v ?? "—")]),
        ),
      ),
    ),
  );
}

function batchesTable(steps, batches) {
  if (!batches.length) return h("div", { class: "faint" }, "—");
  const phases = ["data", "forward", "loss", "backward", "post_backward", "optimizer"];
  const max = Math.max(...batches.map((b) => b.timing?.total_ms || 0), 1e-9);
  return h(
    "table",
    { class: "grid" },
    h("thead", {}, h("tr", {}, h("th", {}, "batch"), h("th", {}, "step"), h("th", { class: "num" }, "loss"), h("th", {}, "time breakdown"), h("th", { class: "num" }, "total"))),
    h(
      "tbody",
      {},
      batches.map((b) =>
        h(
          "tr",
          { onclick: () => store.set({ mode: "batch", batchIndex: b.index, tab: "samples" }) },
          h("td", {}, b.index),
          h("td", {}, b.step),
          h("td", { class: "num" }, fmtNum(lossValue(b))),
          h(
            "td",
            { style: { width: "50%" } },
            h(
              "div",
              { class: "phasebar", style: { padding: 0, border: 0, minHeight: 0, background: "transparent" } },
              h(
                "div",
                { class: "bar", style: { width: `${((b.timing?.total_ms || 0) / max) * 100}%` } },
                phases.map((p) => {
                  const ms = b.timing?.phases_ms?.[p];
                  return ms ? h("span", { style: { width: `${(ms / b.timing.total_ms) * 100}%`, background: phaseColor(p) }, title: `${p} ${fmtMs(ms)}` }) : null;
                }),
              ),
            ),
          ),
          h("td", { class: "num" }, fmtMs(b.timing?.total_ms)),
        ),
      ),
    ),
  );
}

function moduleTable(model) {
  const rows = model.modules.filter((m) => m.path && !m.path.includes(".")).slice(0, 40);
  return h(
    "table",
    { class: "grid" },
    h("thead", {}, h("tr", {}, h("th", {}, "module"), h("th", {}, "class"), h("th", { class: "num" }, "params (incl. children)"))),
    h(
      "tbody",
      {},
      rows.map((m) => {
        const total = model.modules.filter((x) => x.path === m.path || x.path.startsWith(`${m.path}.`)).reduce((s, x) => s + x.params, 0);
        return h("tr", {}, h("td", { class: "mono" }, m.path), h("td", {}, m.cls), h("td", { class: "num" }, fmtInt(total)));
      }),
    ),
  );
}
