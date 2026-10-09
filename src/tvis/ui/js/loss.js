// Loss page: the loss over the run, how one batch's loss is built, and predictions vs targets.

import { api } from "./api.js";
import { followableSamples, lossValue } from "./model.js";
import { openInReplay, store } from "./store.js";
import { clear, cssVar, fmtNum, h, tooltip } from "./util.js";

const NS = "http://www.w3.org/2000/svg";
const svg = (tag, attrs = {}, text) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  return n;
};

let sortBy = "loss";

export function renderLoss(container) {
  const state = store.get();
  const batches = state.allBatches;
  if (!batches) return clear(container, h("div", { class: "loading" }, "loading…"));
  const batch = batches.find((b) => b.index === state.batchIndex) || batches[0];
  const lossCall = [...batch.calls].reverse().find((c) => c.kind === "loss");
  const detailSlot = h("div", { class: "card" }, h("div", { class: "loading" }, "loading…"));
  const predSlot = h("div", { class: "card" });
  clear(
    container,
    h(
      "div",
      { class: "page-wrap" },
      h(
        "div",
        { class: "grid-2" },
        h(
          "div",
          { class: "card" },
          h("h3", {}, "Loss per batch"),
          lossChart(batches, state.steps, batch.index),
          h("div", { class: "hint" }, "Click a point to inspect that batch. Each step sees different data, so this mixes learning progress with how hard each batch is."),
        ),
        h("div", { class: "card" }, h("h3", {}, `Batch ${batch.index}: distribution of per-sample loss`), lossHistogram(batch)),
      ),
      h("div", { class: "gap-y" }),
      lossCall ? h("div", { class: "formula" }, formula(lossCall, batch)) : h("div", { class: "notice" }, "No loss function call was recorded in this batch (only torch.nn.functional losses are recognised)."),
      h("div", { class: "gap-y" }),
      h("div", { class: "grid-2" }, detailSlot, predSlot),
    ),
  );
  if (!lossCall) return clear(detailSlot, h("div", { class: "faint" }, "—"));
  api("loss_detail", { run: state.runId, batch: batch.index })
    .then((detail) => {
      if (detail.cols > 1) {
        renderTokens(detailSlot, detail, batch, state);
        renderSequenceSummary(predSlot, detail, batch);
      } else {
        renderPerSample(detailSlot, detail, batch, state);
        renderConfusion(predSlot, batches, state);
      }
    })
    .catch((err) => clear(detailSlot, h("div", { class: "err" }, err.message)));
}

function formula(call, batch) {
  const shape = (name) => {
    const ref = call.inputs.find((e) => e.name === name)?.value;
    const meta = ref?.kind === "tensor" ? batch.tensors[ref.tid] : null;
    return meta ? `[${meta.shape.join(", ")}]` : "";
  };
  const extra = call.extra || {};
  const args = [`input ${shape("input")}`, `target ${shape("target")}`];
  if (extra.ignore_index != null && extra.ignore_index !== -100) args.push(`ignore_index=${extra.ignore_index}`);
  return [
    h("span", { class: "faint" }, `${call.call_site ? `${call.call_site.file}:${call.call_site.line}  ` : ""}`),
    `${call.name}(${args.join(", ")}) `,
    h("span", { class: "faint" }, `reduction=${extra.reduction ?? "mean"} → `),
    h("b", {}, fmtNum(lossValue(batch), 5)),
  ];
}

function lossChart(batches, steps, selected) {
  const values = batches.map((b) => ({ b: b.index, step: b.step, v: lossValue(b) })).filter((p) => p.v != null);
  if (!values.length) return h("div", { class: "faint" }, "no losses recorded");
  const W = 520;
  const H = 180;
  const pad = { l: 44, r: 12, t: 10, b: 24 };
  const lo = Math.min(...values.map((p) => p.v));
  const hi = Math.max(...values.map((p) => p.v));
  const span = hi - lo || Math.abs(hi) || 1;
  const inset = 18; // keep the first/last points clear of the axis labels
  const x = (i) => pad.l + inset + (values.length === 1 ? (W - pad.l - pad.r - 2 * inset) / 2 : (i / (values.length - 1)) * (W - pad.l - pad.r - 2 * inset));
  const y = (v) => pad.t + (1 - (v - (lo - span * 0.1)) / (span * 1.2)) * (H - pad.t - pad.b);
  const root = svg("svg", { class: "chart", viewBox: `0 0 ${W} ${H}` });
  // step bands
  let bandStart = 0;
  values.forEach((p, i) => {
    const last = i === values.length - 1 || values[i + 1].step !== p.step;
    if (last) {
      if (p.step % 2 === 1) root.append(svg("rect", { class: "step-band", x: x(bandStart) - 8, y: pad.t, width: x(i) - x(bandStart) + 16, height: H - pad.t - pad.b, rx: 4 }));
      bandStart = i + 1;
    }
  });
  for (let k = 0; k <= 3; k++) {
    const v = lo - span * 0.1 + (span * 1.2 * k) / 3;
    root.append(svg("line", { class: "grid-line", x1: pad.l, x2: W - pad.r, y1: y(v), y2: y(v) }));
    root.append(svg("text", { class: "axis-label", x: pad.l - 6, y: y(v) + 3, "text-anchor": "end" }, fmtNum(v, 3)));
  }
  root.append(svg("polyline", { class: "series", points: values.map((p, i) => `${x(i)},${y(p.v)}`).join(" ") }));
  values.forEach((p, i) => {
    const dot = svg("circle", { class: `pt ${p.b === selected ? "sel" : ""}`, cx: x(i), cy: y(p.v), r: p.b === selected ? 6 : 4 });
    dot.addEventListener("click", () => store.set({ batchIndex: p.b }));
    tooltip(dot, `batch ${p.b} · step ${p.step} · loss ${fmtNum(p.v, 4)}`);
    root.append(dot);
    root.append(svg("text", { class: "axis-label", x: x(i), y: H - 6, "text-anchor": "middle" }, `b${p.b}`));
  });
  return root;
}

function lossHistogram(batch) {
  const losses = (batch.samples || []).map((s) => s.loss).filter((v) => v != null);
  if (!losses.length) return h("div", { class: "faint" }, "no per-sample losses");
  const bins = 12;
  const lo = Math.min(...losses);
  const hi = Math.max(...losses);
  const width = (hi - lo) / bins || 1;
  const counts = Array(bins).fill(0);
  for (const v of losses) counts[Math.min(bins - 1, Math.floor((v - lo) / width))]++;
  const peak = Math.max(...counts);
  const W = 520;
  const H = 150;
  const root = svg("svg", { class: "chart", viewBox: `0 0 ${W} ${H}` });
  const bw = (W - 40) / bins;
  counts.forEach((c, i) => {
    const bh = (c / peak) * (H - 40);
    const rect = svg("rect", { x: 20 + i * bw + 1, y: H - 22 - bh, width: bw - 2, height: bh, rx: 2, fill: cssVar("--k-loss") });
    tooltip(rect, `${c} sample(s) with loss ${fmtNum(lo + i * width, 3)} – ${fmtNum(lo + (i + 1) * width, 3)}`);
    root.append(rect);
  });
  root.append(svg("text", { class: "axis-label", x: 20, y: H - 6 }, fmtNum(lo, 3)));
  root.append(svg("text", { class: "axis-label", x: W - 20, y: H - 6, "text-anchor": "end" }, fmtNum(hi, 3)));
  const mean = losses.reduce((a, b) => a + b, 0) / losses.length;
  const mx = 20 + ((mean - lo) / ((hi - lo) || 1)) * (W - 40);
  root.append(svg("line", { x1: mx, x2: mx, y1: 8, y2: H - 22, stroke: cssVar("--text-muted"), "stroke-dasharray": "3 3" }));
  root.append(svg("text", { class: "axis-label", x: mx + 4, y: 16 }, `mean ${fmtNum(mean, 3)}`));
  return root;
}

function renderPerSample(slot, detail, batch, state) {
  const classes = state.meta?.data?.dataset?.classes;
  const rows = (batch.samples || []).map((s, i) => ({ s, loss: detail.per_element[i]?.[0], p: detail.p_target?.[i]?.[0] }));
  rows.sort(sortBy === "loss" ? (a, b) => (b.loss ?? 0) - (a.loss ?? 0) : (a, b) => a.s.position - b.s.position);
  const max = Math.max(...rows.map((r) => r.loss ?? 0), 1e-9);
  const { limit: followable, total } = followableSamples(batch);
  const name = (t) => (t == null ? "—" : t.name ?? classes?.[t.id] ?? t.id);
  clear(
    slot,
    h(
      "div",
      { style: { display: "flex", alignItems: "center", gap: "10px" } },
      h("h3", { style: { margin: 0, flex: 1 } }, "How this batch's loss is built"),
      h("div", { class: "seg" }, [["loss", "worst first"], ["position", "#"]].map(([k, l]) => h("button", { class: sortBy === k ? "on" : "", onclick: () => ((sortBy = k), renderPerSample(slot, detail, batch, state)) }, l))),
    ),
    h(
      "div",
      { class: "hint", style: { marginTop: "6px" } },
      "loss_i = −log p(target_i); the batch loss is their mean. Click a sample to watch it go through the layers in Replay",
      followable < total ? ` (the first ${followable} of ${total} were stored for that).` : ".",
    ),
    h("div", { class: "lossrow head" }, h("span", {}, "#"), h("span", {}, "target → predicted"), h("span", {}, "p(target)"), h("span", {}, "loss"), h("span", {}, "")),
    rows.map(({ s, loss, p }) =>
      h(
        "div",
        s.position < followable
          ? { class: "lossrow", title: "follow this sample through the layers in Replay", onclick: () => openInReplay(batch.index, s.position) }
          : { class: "lossrow", style: { cursor: "default" }, title: `only the first ${followable} samples' activations were stored (--sample-rows)` },
        h("span", { class: "faint" }, `#${s.position}`),
        h("span", {}, name(s.target), " → ", h("b", { style: { color: s.prediction?.correct ? "var(--good)" : "var(--bad)" } }, name(s.prediction))),
        h("span", { class: "mono" }, p != null ? fmtNum(p, 3) : "—"),
        h("span", { class: "mono" }, fmtNum(loss, 3)),
        h("span", { class: "bar" }, h("i", { style: { width: `${((loss ?? 0) / max) * 100}%` } })),
      ),
    ),
  );
}

function renderConfusion(slot, batches, state) {
  const classes = state.meta?.data?.dataset?.classes;
  const samples = batches.flatMap((b) => b.samples || []).filter((s) => s.target?.id != null && s.prediction?.id != null);
  if (!samples.length) return clear(slot, h("h3", {}, "Predictions vs targets"), h("div", { class: "faint" }, "no class predictions recorded"));
  const ids = [...new Set(samples.flatMap((s) => [s.target.id, s.prediction.id]))].sort((a, b) => a - b).slice(0, 20);
  const counts = new Map();
  for (const s of samples) counts.set(`${s.target.id}:${s.prediction.id}`, (counts.get(`${s.target.id}:${s.prediction.id}`) || 0) + 1);
  const peak = Math.max(...counts.values());
  const label = (id) => classes?.[id] ?? id;
  const correct = samples.filter((s) => s.prediction.correct).length;
  clear(
    slot,
    h("h3", {}, "Predictions vs targets"),
    h("div", { class: "hint" }, `all ${samples.length} captured samples · accuracy ${((correct / samples.length) * 100).toFixed(1)}% · rows = target, columns = prediction`),
    h(
      "table",
      { class: "confusion" },
      h("tr", {}, h("th", {}), ids.map((id) => h("th", {}, label(id)))),
      ids.map((t) =>
        h(
          "tr",
          {},
          h("th", { style: { textAlign: "right" } }, label(t)),
          ids.map((p) => {
            const c = counts.get(`${t}:${p}`) || 0;
            const color = t === p ? "--good" : "--bad";
            return h("td", { style: { background: c ? `color-mix(in srgb, var(${color}) ${Math.round((c / peak) * 70) + 10}%, transparent)` : "transparent" } }, c || "");
          }),
        ),
      ),
    ),
  );
}

function renderTokens(slot, detail, batch, state) {
  const vocab = state.meta?.data?.vocab;
  const tok = (id) => (id == null ? "?" : vocab?.[id] ?? String(id));
  const max = Math.max(...detail.per_element.flat().filter((v) => v != null), 1e-9);
  const samples = batch.samples || [];
  clear(
    slot,
    h("h3", {}, "Loss per token"),
    h("div", { class: "hint" }, "Each target token shaded by its loss: the redder, the more the model was surprised. Struck-through tokens are padding and don't count. Hover a token for p(target) and the prediction."),
    detail.per_element.map((row, i) =>
      h(
        "div",
        {},
        h("div", { class: "faint", style: { fontSize: "11px" } }, `#${i} · loss ${fmtNum(samples[i]?.loss, 3)}`),
        h(
          "div",
          { class: "tokrow" },
          row.map((loss, j) => {
            const valid = detail.valid[i][j];
            const target = detail.target_ids?.[i]?.[j];
            const pred = detail.pred_ids?.[i]?.[j];
            const t = h(
              "span",
              { class: `t ${valid ? "" : "ign"}`, style: valid ? { background: `color-mix(in srgb, var(--bad) ${Math.round(((loss ?? 0) / max) * 75)}%, transparent)` } : {} },
              valid ? tok(target) : "pad",
            );
            if (valid) tooltip(t, `target "${tok(target)}" · predicted "${tok(pred)}" · p(target) ${fmtNum(detail.p_target?.[i]?.[j], 3)} · loss ${fmtNum(loss, 3)}`);
            return t;
          }),
        ),
      ),
    ),
  );
}

function renderSequenceSummary(slot, detail, batch) {
  const rows = (batch.samples || []).map((s, i) => {
    const valid = detail.valid[i] || [];
    const n = valid.filter(Boolean).length;
    const hits = (detail.pred_ids?.[i] || []).filter((p, j) => valid[j] && p === detail.target_ids?.[i]?.[j]).length;
    return { s, n, hits };
  });
  clear(
    slot,
    h("h3", {}, "Predictions vs targets"),
    rows.map(({ s, n, hits }) =>
      h(
        "div",
        { style: { marginBottom: "10px" } },
        h("div", { class: "faint", style: { fontSize: "11px" } }, `#${s.position} · ${hits}/${n} tokens right`),
        h("div", { class: "textline" }, h("span", { class: "lbl" }, "target"), s.target?.text ?? "—"),
        h("div", { class: "textline" }, h("span", { class: "lbl" }, "predict"), s.prediction?.text ?? "—"),
      ),
    ),
  );
}
