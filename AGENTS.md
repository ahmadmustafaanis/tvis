# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Cursor, Copilot, …) working in this repo.
Humans: see [README.md](README.md). Product scope: [SCOPE.md](SCOPE.md).

## Commands

Run these from the repo root. They are the source of truth for "done".

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # setup (CPU torch is fine)
.venv/bin/pytest -q                                          # all tests
.venv/bin/pytest -q tests/capture/test_tensors.py            # one file — prefer this while iterating
.venv/bin/pytest -q -m "not slow"                            # skip subprocess end-to-end tests
.venv/bin/ruff check . && .venv/bin/ruff format --check .    # lint + format check
.venv/bin/ruff format .                                      # auto-format
```

Before you say a change is finished, run `ruff check`, `ruff format --check` and the
tests touching the code you changed; run the full suite before committing.

## What this project is

`tvis run train.py` executes an unmodified PyTorch training script in-process, records the first N
optimizer steps end to end (raw input → data pipeline → every module/function → output → loss →
gradients → optimizer update, with GPU-accurate timings), writes a run directory, and exits.
`tvis open host:path` serves a browser UI on the laptop that reads runs locally or over SSH stdio.

## Layout

```
src/tvis/
  cli.py              entry point: run | open | agent | target
  runner.py           runs the user script in-process, stops after N steps
  capture/            everything that executes inside the training process
    session.py        orchestrator: phases, batch/step boundaries, call stack
    tensors.py        tensor registry: ids, stats, value/grad capture
    stats.py          on-device tensor statistics
    timing.py         CUDA-event / perf_counter marks and overhead accounting
    modules.py        global nn.Module hooks
    tracer.py         project-function tracer (sys.setprofile, project files only)
    ops.py            line-level op recorder (TorchFunctionMode)
    data.py           DataLoader / transforms / tokenizer / loss instrumentation
  store/              run-directory format: writer (capture side), reader (viewer side)
  viewer/             HTTP server, request dispatch, stdio agent, SSH transport
  ui/                 static browser UI (vanilla JS modules, no build step)
tests/                mirrors src/tvis/ (tests/capture, tests/store, tests/viewer, tests/e2e)
examples/             small runnable training scripts (images + text) used by e2e tests
```

## Non-negotiable invariants

1. **Observation never changes training numerics.** Hooks must not modify tensors, autograd graphs,
   RNG state or dtype. Copy with `detach().clone()`; never return a value from a gradient hook.
   `tests/e2e/test_numerics.py` compares losses with and without `tvis run`; it must stay green.
2. **Instrumentation must never crash training.** Every hook body goes through
   `Session.guard(...)`, which records the error as a run notice and keeps going.
3. **tvis internals must not record themselves.** Wrap internal torch work in
   `with session.internal():` so the op recorder, tracer and timers ignore it, and its time is
   subtracted as overhead.
4. **`tvis open` must work without torch installed.** Only `capture/` and `runner.py` may import torch,
   and only lazily. `store/reader.py` and `viewer/` depend on numpy + stdlib only.
5. **Anything inferred or approximate is labelled** in the stored data (`"approx": true`,
   `"inferred": true`), never silently presented as exact.

## Code style

- Python ≥ 3.10, type hints on public functions, `from __future__ import annotations`.
- Small, plain data objects (`@dataclass(slots=True)`) for records written to disk; JSON-serialisable
  via `to_json()`.
- No new runtime dependencies without discussion; the viewer side is stdlib + numpy on purpose.
- Prefer explicit names over abbreviations (`grad_arrival_mark`, not `gam`).
- UI: vanilla ES modules, no bundler, no framework. CSS colours only via variables in `ui/style.css`.

## Tests

- Tests describe behaviour, one behaviour per test, named `test_<behaviour>_<condition>`.
  Example: `test_grad_hook_receives_pre_inplace_gradient`.
- Use the fixtures in `tests/conftest.py` (`toy_mlp`, `run_dir`, `capture_session`, …) instead of
  re-building models in each test. Add a fixture when two tests need the same setup.
- Compare tensors with `torch.testing.assert_close` / `numpy.testing.assert_allclose`, and use
  exact equality (`torch.equal`) when asserting "numerics unchanged".
- No sleeps, no network, no GPU requirement. CUDA-specific paths are tested via the timer
  abstraction; mark real-GPU tests with `pytest.mark.skipif(not torch.cuda.is_available(), ...)`.
- End-to-end tests that spawn `tvis run` subprocesses are marked `@pytest.mark.slow`.
- A bug fix lands with a test that fails without the fix.

## Git

- Small, focused commits; imperative subject ≤ 72 chars (`Add gradient hooks to tensor registry`),
  body explains *why*.
- Commits are authored by the repository owner only: **no `Co-Authored-By` trailers**.
- Never commit `runs/`, `.venv/`, credentials, or captured data from real training runs.
- Don't rewrite pushed history.

## Boundaries

- **Always:** run lint + relevant tests; keep SCOPE.md in sync when behaviour changes; update this
  file when commands or layout change.
- **Ask first:** adding dependencies, changing the on-disk run format (bump `FORMAT_VERSION` in
  `store/schema.py`), changing CLI flags, anything touching the user's `~/.ssh` config.
- **Never:** commit secrets; make network calls from capture code; bind the viewer to anything but
  `127.0.0.1`; silence a failing test instead of fixing the cause.
