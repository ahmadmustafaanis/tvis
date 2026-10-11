---
name: run-tvis
description: Run tvis, the PyTorch training debugger, and walk a user through debugging their training. Use to record a training script with `tvis run`, read or summarise a recording, run health checks (NaN/Inf, loss not decreasing, double softmax, missing zero_grad, frozen/dead layers, unnormalised inputs, learning rate too high), inspect layers/gradients/parameter updates/samples/data pipeline/timing, compare a run before and after a fix, open or screenshot the tvis viewer, or test the tvis app itself.
---

# run-tvis

tvis records the first N optimizer steps of an unmodified PyTorch script (`tvis run`) and shows
them in a browser viewer (`tvis open`). An agent can't read the browser, so this skill's driver
**`tvis_debug.py`** reads a recording as text, through tvis's own viewer API (numpy + stdlib, no
torch, local or over SSH). It also runs heuristic health checks, and every finding names the
next command to run.

`$D` is this skill's directory: `.claude/skills/run-tvis` in the tvis repo, or wherever the
skill was installed (e.g. `~/.claude/skills/run-tvis`). Commands below are written for the tvis
repo, where tvis lives in `.venv`. In a user's project, use their `tvis` and the Python that has
tvis installed in place of `.venv/bin/tvis` / `.venv/bin/python`.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"     # in the tvis repo
```

The driver needs only a Python with `tvis` installed. The user's training environment needs
torch, but the machine reading the runs doesn't.

## Agent path: debug a user's training

Shown on the bundled fixture. For a real user, swap in their script and arguments.

```bash
D=.claude/skills/run-tvis
# 1. record, with the same args the user trains with; --out keeps before/after runs in one place
.venv/bin/tvis run --steps 3 --out /tmp/tv $D/buggy/train.py --bug softmax
# 2. overview + health findings (start here; --run defaults to the newest run)
.venv/bin/python $D/tvis_debug.py summary /tmp/tv
# 3. follow each finding's "next:" line (it prints the full command), e.g.
.venv/bin/python $D/tvis_debug.py call /tmp/tv --run 0 37 --batch 0
# 4. fix the script, record again into the same --out, check the fix
.venv/bin/tvis run --steps 3 --out /tmp/tv $D/buggy/train.py --bug none
.venv/bin/python $D/tvis_debug.py compare /tmp/tv --run 1 --against 0   # 1 = before, 0 = newest
```

How to work with the user:

1. Ask for (or find) the exact command they train with, and record that. Use 3 steps for
   correctness bugs, more (`--steps 10`) for "loss doesn't go down".
2. Read `summary`, then work through findings in severity order: `error`, then `warn`, then `info`.
   Findings are heuristics. Confirm each one with `call` / `tensor` / `source` before you tell the
   user it's the cause.
3. Explain the cause with the evidence (call id, source line, numbers), propose the fix, apply it
   only if the user agrees, re-record, and show `compare`.
4. If the user wants to look for themselves, give them `tvis open /tmp/tv` (or `HOST:PATH`) and a
   deep link from `link`. Use `shot` when you need to see a page yourself.

### Commands (`tvis_debug.py <cmd> TARGET [--run ID] ...`)

TARGET is what `tvis open` takes: a runs dir, a single run dir, `HOST:PATH`, or a saved target
(`--via HOP`, `--remote-tvis PATH` work as in `tvis open`). `--run` takes a run id, any unique part
of one, or an index (0 = newest).

| command | use it for |
|---|---|
| `runs` | list recordings (status, steps, script, host) |
| `summary` | script, model, optimizer, data, loss/accuracy per batch, phase times, health |
| `health [--json]` | only the findings; `--json` gives `where: {batch, call, tid}` for scripting |
| `layers [--batch K] [--grep S] [--depth N]` | every forward call: output mean/std/zeros, ‖dL/d(out)‖, NaN/Inf, fwd/bwd ms, flags |
| `params [--step S] [--grep S]` | per parameter per step: ‖w‖, ‖grad‖, ‖Δw‖, ‖Δw‖/‖w‖ |
| `samples [--batch K] [--top N]` | per sample, worst loss first: target, prediction, p(target), raw text |
| `data [--batch K] [--samples N]` | raw item → each transform's output stats → collated batch |
| `call ID [--batch K]` | one call: inputs/outputs + gradient stats, children, line ops, source around it |
| `tensor TID [--batch K] [--grad] [--sample I] [--fixed 0,3]` | stats and a window of actual values |
| `source FILE [--line N] [--context C]` | source snapshot as executed, each line annotated with op output shapes |
| `timing [--batch K] [--top N]` | per-batch phase times, slowest forward/backward calls |
| `compare --run A --against B` | loss/acc per batch, last-step grads/updates, health fixed/new |
| `link [--page P] [--tab T] [--call ID] [--sample I] [--viewer URL]` | deep link into a running viewer |
| `shot [--page P] [--tab T] [--call ID] [--out F.png]` | serves the viewer itself, PNG via headless Chrome |

Pages: `replay` `insights` (`--insight attention|gradcam|embeddings`) `loss` `weights` `explore`
(the Trace page; `--tab overview|samples|timeline|source|layers|step`).

### Health checks and the bug each one catches

| check | catches |
|---|---|
| `run-failed`, `notice` | the script crashed (traceback tail); capture problems |
| `nonfinite-value` / `nonfinite-grad` | first call whose output / gradient has NaN or Inf |
| `loss-input-probabilities`, `nll-loss-on-logits`, `bce-logits-on-probabilities` | softmax applied twice, missing log_softmax, sigmoid applied twice |
| `initial-loss-high` | first loss ≫ ln(classes): bad input scale or init |
| `loss-rising`, `loss-frozen` | LR too high or wrong sign; nothing learning |
| `input-not-normalized` | raw 0-255 pixels or other huge model inputs |
| `dead-activations`, `gradient-blocked`, `activation-explodes` | dead ReLUs (incl. functional `F.relu` after a layer), blow-ups |
| `layer-not-optimized`, `param-without-grad`, `param-zero-grad` | layer not passed to the optimizer, detached/unused params |
| `update-too-large`, `update-tiny`, `grad-scale-spread` | ‖Δw‖/‖w‖ far from ~1e-3; vanishing/exploding grads |
| `stale-gradients` / `grad-rescaled` | **exact**: recomputes the last `nn.Linear`'s dL/dW from this step's captured input × output-gradient and compares with `.grad` → missing `zero_grad()` / clipping |
| `single-label-batch`, `duplicate-indices`, `single-prediction` | data/sampler bugs, collapsed predictions |

## Viewer (human path, and screenshots)

```bash
.venv/bin/tvis open /tmp/tv --no-browser --port 8775          # prints http://127.0.0.1:8775/ ; blocks until Ctrl-C
.venv/bin/python $D/tvis_debug.py link /tmp/tv --page explore --tab source --call 37 --viewer http://127.0.0.1:8775/
.venv/bin/python $D/tvis_debug.py shot /tmp/tv --page loss --out /tmp/loss.png   # no server needed
```

`tvis open` blocks, so run it in the background or in its own terminal. Then the agent can call the
same API the UI uses: `curl -s -X POST http://127.0.0.1:8775/api -d '{"method":"ping","params":{}}'`.
In Claude Code's desktop app, use `preview_start` with a `.claude/launch.json` entry that runs
`tvis open <dir> --no-browser --port N`, then navigate the browser pane to a `link` URL.

## Testing changes to tvis or to this driver

```bash
.claude/skills/run-tvis/smoke.sh /tmp/tvis-smoke     # ~15 s: 7 injected bugs → expected checks, every command, a shot
.venv/bin/ruff check . && .venv/bin/ruff format --check .   # covers this directory too
.venv/bin/pytest -q
```

`$D/buggy/train.py --bug none|softmax|nozero|unnorm|frozen|dead|lr` is the fixture. Each bug
must produce its check in `smoke.sh`. Add a case there when you add a check.

## Gotchas

- **A diverging run crashes `tvis run` itself** (as of 14f9f99): once a parameter update becomes
  `inf`, `steps.json` can't be written (`allow_nan=False`) and the exception surfaces in the
  user's `optimizer.step()`. The run ends `status=error` without its last step. `health` still
  reports `run-failed` + `loss-rising` + `update-too-large`. Record fewer steps to get a clean run
  of the steps before the blow-up. With the `lr` fixture, `--steps 3` completes and `--steps 4` crashes.
- `tvis run` stops the script after N optimizer steps. Eval/checkpoint code after the loop never
  runs, and DataLoader workers are forced to 0 (an info `notice`). Data-phase times reflect that.
- Runs default to `<script dir>/tvis_runs/`. Pass the same `--out` for before/after so `compare`
  can see both. `--run 0` is the newest.
- Layer names like `1:conv` mean the module is outside the model the optimizer owns (tvis names
  model 0 by optimizer ownership). That's usually a `layer-not-optimized` bug.
- A conv/linear bias feeding BatchNorm legitimately has ~0 gradient, so the spread/too-large
  checks only look at ≥2-D weights (1-D params only if ‖Δw‖/‖w‖ > 1).
- `stale-gradients` only checks the last `nn.Linear` whose weight the optimizer updates, and needs
  its input and output gradient stored in full (default `--max-elems 2000000`).
- `ms` columns are device times with tvis overhead removed (CUDA events on GPU). Wall times
  are inflated by capture. `~`/`approx` backward times are inferred from gradient arrival.
- Headless Chrome writes the screenshot and then often never exits. `shot` waits for the PNG and
  kills Chrome. Override the binary with `TVIS_CHROME=/path/to/chrome`. macOS has no `timeout`
  command, so don't wrap Chrome in it.
- `tensor` windows print in scientific notation when needed. Gradients are often ~1e-6, and
  fixed-point output would show them as 0.

## Troubleshooting

- `--run 'train' matches 2 runs; available: ...` → pass more of the id, or an index from `runs`.
- `no source snapshot 'foo.py'` → the error lists the snapshots. Paths are relative to the project dir.
- `tvis: cannot listen on port 8765: [Errno 48] Address already in use` → another viewer (often an
  earlier session's) holds the port. Pick another `--port`, or `--port 0` for any free one.
- `tvis_debug: connection to HOST:PATH closed`, with `/bin/sh: .../tvis: No such file or directory`
  above it → `tvis` isn't on the remote non-interactive PATH. Add
  `--remote-tvis ~/venv/bin/tvis`. With `Could not resolve hostname` instead, use the `~/.ssh/config`
  alias or `--via`.
- `tensor ... --sample N` → `sample N out of range`: N is a position in the batch (0..batch_size-1),
  not a dataset index. Get positions from `samples`.
