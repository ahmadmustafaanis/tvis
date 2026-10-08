# tvis — v1 scope

A step-through debugger for PyTorch training. Run your training script under `tvis run`,
it records the first N steps end to end — from raw input file/string to the model's final
output and gradients — then exits. Open the recording in a browser.

## Usage

```bash
# normal training — tvis is never imported, zero impact
python train.py --lr 3e-4

# debug run — same script, same args, no code changes
tvis run --steps 3 train.py --lr 3e-4
#   → runs 3 optimizer steps, writes tvis_runs/<run-id>/, exits

# view — run on your laptop; hops come from ~/.ssh/config (see "Viewing remotely")
tvis open mycluster:~/proj/tvis_runs
```

On SLURM: swap `python train.py` → `tvis run train.py` in the sbatch script (or an `srun` allocation).

## Definitions

- **Batch** — one forward + loss + backward (a micro-batch).
- **Step** — one `optimizer.step()`; contains ≥1 batches (gradient accumulation).
- **Run** — `tvis run` executes exactly `--steps N` steps (default 3), flushes, and exits the process.
  Code after the training loop (eval, checkpoint saving) does not run.

## Integration (no code changes)

`tvis run` executes the script in-process (same `__main__`, `sys.argv`, imports) after installing
instrumentation via PyTorch's global hooks and targeted patches:

| What | How it's found |
|---|---|
| Steps | global optimizer step pre/post hooks (any `torch.optim` optimizer) |
| Batches | `Tensor.backward` patch; fallback step boundary if no optimizer |
| Layers | global module forward hooks + tensor gradient hooks |
| Your functions | profiler restricted to files under the project directory |
| Data | `DataLoader` iteration + sampler wrapped; dataset, transforms, tokenizer, collate traced |
| Normalization / tokenizer | **auto-detected** from the data pipeline (no config) |

Multiple models/optimizers are captured and shown separately.

## What is captured

### 1. Data pipeline — raw input → batch
For every sample in every captured batch:
- **Dataset index** and the **raw item** (image file path / PIL image, or raw text string).
- **Every transform stage** with its output: e.g. `PIL 500×375 → Resize → CenterCrop → ToTensor [3,224,224] → Normalize`;
  or `"the cat sat" → tokenizer → ids [12] + tokens ["the","Ġcat","Ġsat"] → padding/truncation`.
- Your own `Dataset.__getitem__`, preprocessing and `collate_fn` functions (traced as user code).
- The **collated batch** exactly as the model received it (inputs + targets).

DataLoader workers run in the main process during `tvis run` (`num_workers=0`) so the pipeline is
fully observable. This is a debug run, so the speed cost doesn't matter. Augmentation randomness is still
captured faithfully for *this* run.

### 2. Model — batch → output → loss
- **Call tree**: every `nn.Module` call and every function from your project, in execution order,
  linked to file:line. Library internals (torch/HF Python functions) are not traced.
- **Function level**: inputs/outputs of each call: shape, dtype, device, stats
  (mean, std, min, max, norm, % zeros, NaN/Inf, histogram), full values (size-capped).
- **Line level (inline)**: inside your functions, every torch op recorded and mapped to its source line,
  so each line shows its result shape/dtype (TorchDispatchMode).
- **Output decoding**: logits → class names + probabilities (images; class names from `dataset.classes`
  when available); predicted tokens → decoded text vs target text (text).
- **Loss**: total and per-sample (exact for standard loss modules; otherwise marked unavailable).

### 3. Backward
- **Gradients** dL/d(tensor) for every captured tensor that requires grad, with the same stats.
- **Parameters** per step: grad stats, weight stats, update Δw, update/weight ratio.

### 4. Timing
- Forward time per call and backward time per module/function, measured on GPU with CUDA events
  (not Python wall time).
- Per batch: data pipeline → forward → loss → backward. Per step: + optimizer.
- Backward time of plain Python functions is inferred from gradient arrival → labelled "approx".

## UI (browser)

- **Run list** → run overview (model summary, dataset summary, step timings).
- **Granularity toggle**: Batch | Step, with picker.
- **Pipeline strip** (top): raw input → transforms/tokenizer → collate → model → output → loss.
  Click any stage to inspect it.
- **Data tab**: sample grid for the batch (image thumbnails / text rows) with target, prediction,
  confidence, correct/wrong, per-sample loss; sortable by loss.
- **Follow-a-sample mode**: pick one datapoint; every tensor in the UI is sliced to that sample,
  from its raw file/string through every transform, layer and function to its output and gradients.
  Batch dimension tracked through reshapes (marked "inferred" where needed; layers that mix samples,
  e.g. BatchNorm in training mode, are flagged).
- **Call tree** (left): collapsible, repeated blocks grouped, shape + fwd/bwd time bars per row.
  **Hide/skip** any function or module (persisted per project).
- **Center tabs**:
  - *Timeline*: forward/backward lanes, flame-chart style.
  - *Source*: code snapshot from the run, executed functions highlighted, shapes inline per line,
    times per function.
  - *Layers*: sortable table of all modules/functions with stats and times.
- **Inspector** (right): selected call/stage. Inputs/outputs, value | gradient toggle, histogram,
  heatmap slice; images rendered as images (auto un-normalized), token ids as decoded text.

## Viewing remotely

The remote machine can't reliably reach back to the laptop (NAT, no sshd, it only knows the
previous hop), so **the connection is always started from the laptop**. The UI runs locally and
fetches data from a remote agent over SSH's stdin/stdout. No remote ports are opened.

```
Mac                                         cluster login node
┌──────────────────────────┐   ssh (hops)   ┌────────────────────────────┐
│ tvis open → local web UI │◀══ stdin/out ══▶│ tvis agent --stdio         │
│ http://localhost:PORT    │                │ reads runs/ on shared FS   │
└──────────────────────────┘                └────────────────────────────┘
```

```bash
tvis open mycluster:~/proj/tvis_runs     # opens browser on the Mac
```

- Works where TCP forwarding is disabled on the cluster; no open port on shared login nodes;
  remote side lives exactly as long as the SSH session (no orphans).
- Reuses existing SSH sessions (ControlMaster), so 2FA/Duo clusters prompt at most once.
- Lazy fetching: only what is viewed crosses the hops.
- Requires `pip install tvis` on the laptop (UI is static files).
- Local runs: `tvis open ./runs` (no SSH).

### Specifying hops (a "target" = host + runs path)
Resolution order:
1. **`~/.ssh/config`** aliases incl. `ProxyJump` (recommended; no new format): `tvis open mycluster:~/proj/tvis_runs`
2. **CLI**: `tvis open --via A --via B C:~/proj/tvis_runs`
3. **Saved target**: `tvis target add gpu --via A,B --host C --path ~/proj/tvis_runs` → `tvis open gpu`
   (stored in `~/.tvis/targets.json`)

### Backtracking help
`tvis run` ends by printing its hostname, the run path and the exact `tvis open` command to run on
the laptop. The route from the laptop is left to SSH config / `--via`.

### Agents (Claude Code / Codex)
Agents run on the laptop and know the hops they used, so they call `tvis open <target>` directly
and hand back the localhost link.

## Accuracy guarantees

- Observation never changes numerics. Tested: identical losses with and without `tvis run`
  (given the same seed and `num_workers` setting).
- Source code is snapshotted at run start; the Source view is exactly what executed.
- Anything inferred or approximate is labelled in the UI.

## Performance

- `python train.py`: tvis not imported → zero impact.
- `tvis run`: slow by design for the N captured steps (2–10×); per-function GPU timings stay
  accurate. Wall-clock step time is labelled "during capture".
- Captured tensors are offloaded to CPU asynchronously; per-tensor size cap (~4M elements full,
  larger → stats + downsampled preview). Disk writes on a background thread.

## Supported in v1

- PyTorch, single GPU (or CPU), eager mode.
- Modalities: **images** (torchvision transforms v1/v2, PIL) and **text** (HF tokenizers fast/slow).
- Plain training loops, HF Trainer, Lightning (single device).

## Out of scope for v1

- LLM analysis
- Replay of arbitrary step ranges (step N → M)
- Monitoring beyond the captured steps
- Multi-GPU / DDP / FSDP / `torchrun`
- `torch.compile` during the run (runs eager)
- JAX
- Custom `autograd.Function` backward internals; per-sample *parameter* gradients
- Audio, tabular and other modalities

## TODO (after v1)

- [ ] Dataset browser: whole dataset, not just captured batches
- [ ] Replay of step N → M from keyframes
- [ ] LLM analysis of gradients + code
- [ ] Light monitoring for the full run
- [ ] Multi-GPU
- [ ] More modalities (audio, tabular)
- [ ] Opt-in cloud relay for zero-config shareable https links (no SSH)

## Known edge cases

- AMP `GradScaler`: recorded gradients are the scaled gradients.
- `torch.compile`d modules may hide their internals from module hooks.
- Two optimizers (e.g. GANs) close a step per optimizer update, lazily at the next batch.

- HF `datasets` that tokenize in a `.map()` preprocessing pass before training: raw text is
  available only if the dataset keeps a text column; otherwise the pipeline starts at token ids.
- Manual parameter updates without `torch.optim`: steps fall back to `backward()` boundaries.
- Custom iterables that aren't `DataLoader`s: batches still captured at the model input, but no
  per-sample indices or data-wait time.
