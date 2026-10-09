// Playback controls shared by Replay and 3D: buttons, where you are in the training loop, sample
// picker, step size, speed, and a map of the whole recording you can click or drag through.

import { clear, h, icon, phaseColor, tooltip } from "./util.js";

/** The training loop, in order. Every replay event belongs to one of these. */
export const LOOP = [
  ["data", "Data"],
  ["forward", "Forward"],
  ["loss", "Loss"],
  ["backward", "Backward"],
  ["optimizer", "Update"],
];
const RANK = Object.fromEntries(LOOP.map(([phase], i) => [phase, i]));
const SPEEDS = [0.5, 1, 2, 4];

/**
 * opts: {granularities: [[key, label]], gran, speed, onGran(key), onSpeed(x), onRestart(), onPrev(),
 *        onToggle(), onNext(), onSeek(i), onSample(delta), extras: [Node]}
 */
export function buildTransport(opts) {
  const playBtn = h("button", { class: "tbtn play", title: "Play / pause  (space)", onclick: opts.onToggle }, icon("play"));
  const loop = h("div", { class: "loop", role: "group", "aria-label": "Training loop" });
  const sampleNav = h("span", { class: "sample-nav" });
  const gran = h(
    "div",
    { class: "seg", title: "How far one step of playback moves" },
    opts.granularities.map(([key, label]) => h("button", { class: key === opts.gran ? "on" : "", dataset: { gran: key }, onclick: () => opts.onGran(key) }, label)),
  );
  const speed = h(
    "select",
    { class: "select", title: "Playback speed", onchange: (e) => opts.onSpeed(Number(e.target.value)) },
    SPEEDS.map((s) => h("option", { value: s, selected: s === opts.speed }, `${s}×`)),
  );
  const map = h("div", { class: "map" });
  const count = h("span", { class: "count" });
  const root = h(
    "div",
    { class: "transport" },
    h(
      "div",
      { class: "transport-row" },
      h(
        "div",
        { class: "buttons" },
        h("button", { class: "tbtn", title: "Back to the start", onclick: opts.onRestart }, icon("restart")),
        h("button", { class: "tbtn", title: "Previous  (←)", onclick: opts.onPrev }, icon("prev")),
        playBtn,
        h("button", { class: "tbtn", title: "Next  (→)", onclick: opts.onNext }, icon("next")),
      ),
      loop,
      h("div", { class: "grow" }),
      h("span", { class: "ctl", title: "Which sample of the batch to follow  ([ and ])" }, "sample", sampleNav),
      h("span", { class: "ctl" }, "step by", gran),
      speed,
      opts.extras || [],
    ),
    h("div", { class: "scrub" }, map, count),
  );

  const T = { root, events: [], blocks: [], i: 0 };

  // ---- the map: one block per batch, its phases as segments ----
  T.setEvents = (events) => {
    T.events = events;
    T.blocks = [];
    for (let i = 0; i < events.length; i++) {
      const e = events[i];
      let block = T.blocks[T.blocks.length - 1];
      if (!block || block.batch !== e.batch) {
        block = { batch: e.batch, step: e.step, start: i, count: 0, segs: [] };
        T.blocks.push(block);
      }
      block.count++;
      const seg = block.segs[block.segs.length - 1];
      if (seg && seg.phase === e.phase) seg.count++;
      else block.segs.push({ phase: e.phase, start: i, count: 1 });
    }
    T.head = h("span", { class: "head" });
    clear(
      map,
      T.blocks.map((block) => {
        block.el = h(
          "div",
          { class: "blk", style: { flex: String(Math.max(1, block.count)) } },
          h("span", { class: "lab" }, `batch ${block.batch} · step ${block.step}`),
          h(
            "div",
            { class: "segs" },
            block.segs.map((seg) => (seg.el = h("span", { style: { flex: String(seg.count), background: phaseColor(seg.phase) } }))),
          ),
        );
        return block.el;
      }),
      T.head,
    );
  };

  const indexAt = (clientX) => {
    for (const block of T.blocks) {
      const rect = block.el.getBoundingClientRect();
      if (clientX <= rect.right + 3 || block === T.blocks[T.blocks.length - 1]) {
        const f = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
        return block.start + Math.min(block.count - 1, Math.floor(f * block.count));
      }
    }
    return 0;
  };
  let dragging = false;
  map.addEventListener("pointerdown", (e) => {
    dragging = true;
    map.setPointerCapture(e.pointerId);
    opts.onSeek(indexAt(e.clientX));
  });
  map.addEventListener("pointermove", (e) => {
    T.hover = indexAt(e.clientX);
    if (dragging) opts.onSeek(T.hover);
  });
  map.addEventListener("pointerup", () => (dragging = false));
  tooltip(map, () => {
    const e = T.events[T.hover];
    return e ? `${e.label} · ${LOOP[RANK[e.phase]]?.[1] ?? e.phase} · batch ${e.batch}` : "";
  });

  T.setPosition = (i) => {
    T.i = i;
    const event = T.events[i];
    if (!event) return;
    for (const block of T.blocks) {
      block.el.classList.toggle("cur", block.batch === event.batch);
      for (const seg of block.segs) seg.el.classList.toggle("past", seg.start <= i);
      if (block.batch === event.batch && map.offsetWidth) {
        const x = block.el.offsetLeft + ((i - block.start + 0.5) / block.count) * block.el.offsetWidth;
        T.head.style.left = `${(x / map.offsetWidth) * 100}%`;
      }
    }
    count.textContent = `${i + 1} / ${T.events.length}`;
    renderLoop(event);
  };

  // ---- where in the loop: Data → Forward → Loss → Backward → Update ----
  function renderLoop(event) {
    const inBatch = new Map();
    T.events.forEach((e, i) => e.batch === event.batch && !inBatch.has(e.phase) && inBatch.set(e.phase, i));
    const rank = RANK[event.phase] ?? -1;
    const items = [];
    LOOP.forEach(([phase, label], k) => {
      if (k) items.push(h("span", { class: "sep" }, "→"));
      const target = inBatch.get(phase);
      items.push(
        h(
          "button",
          {
            class: k < rank ? "done" : k === rank ? "cur" : "",
            style: { "--pc": phaseColor(phase) },
            disabled: target == null,
            title: target == null ? `no ${label.toLowerCase()} in this batch` : `jump to ${label.toLowerCase()} of batch ${event.batch}`,
            onclick: () => opts.onSeek(target),
          },
          h("i"),
          label,
        ),
      );
    });
    clear(loop, items);
  }

  T.setPlaying = (playing) => clear(playBtn, icon(playing ? "pause" : "play"));
  T.setGran = (key) => {
    for (const button of gran.querySelectorAll("button")) button.classList.toggle("on", button.dataset.gran === key);
  };
  T.setSample = ({ sample, limit, total }) =>
    clear(
      sampleNav,
      h("button", { onclick: () => opts.onSample(-1), title: "Previous sample  ([)" }, icon("left")),
      h("b", {}, `#${sample}`),
      h("button", { onclick: () => opts.onSample(1), title: "Next sample  (])" }, icon("right")),
      limit < total ? h("span", { class: "faint", title: "Larger tensors keep only their first rows (raise --sample-rows to store more)" }, ` ${limit} of ${total} stored`) : null,
    );
  return T;
}

/** Index of the first event of the next (direction 1) or previous (-1) phase block. */
export function phaseJump(events, i, direction) {
  const current = events[i];
  let j = i + direction;
  while (j > 0 && j < events.length - 1 && events[j].phase === current.phase && events[j].batch === current.batch) j += direction;
  return Math.max(0, Math.min(events.length - 1, j));
}
