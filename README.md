# tvis

A step-through debugger for PyTorch training.

Run your **unmodified** training script under `tvis run`. It records the first few optimizer steps
end to end, then exits:

- the raw image file or text string, every transform and the tokenizer, and the collated batch
- every `nn.Module` and every function in your project, in call order, linked to the source line
- the inputs and outputs of each call, with shape, dtype, statistics, histogram and full values
- the gradient dL/d(tensor) of every captured activation
- the shape produced by each line inside your functions
- the output decoded into class names, or into text with the tokenizer
- per-sample loss
- the parameter update of every weight (‖Δw‖/‖w‖)
- forward and backward time per call, measured on the GPU with CUDA events, with tvis's own
  overhead removed

Then open the recording in your browser, on your laptop, even when training ran on a cluster two
SSH hops away.

```bash
python train.py --lr 3e-4                       # normal training: tvis is never imported
tvis run --steps 3 train.py --lr 3e-4           # debug run: same script, same arguments
tvis open mycluster:~/proj/tvis_runs            # on your laptop: opens the viewer
```

> Early development (v1). PyTorch, single device, eager mode. See [SCOPE.md](SCOPE.md).

## Install

```bash
pip install -e .        # in the training environment (needs torch) and on your laptop (no torch needed)
```

## Recording a run

```bash
tvis run [--steps N] [--out DIR] [--project DIR] train.py [script args...]
```

- The script runs in-process exactly as `python train.py ...` would (`__main__`, `sys.argv`,
  imports). After `N` optimizer steps (default 3), tvis saves the recording and stops the script.
- Runs are written to `<project>/tvis_runs/<timestamp>_<script>_<id>/`. The project directory
  defaults to the script's directory. Functions defined under it are traced; library code is not.
- DataLoader workers are moved into the main process during `tvis run` so the data pipeline is
  observable. The run reports this as a notice.
- With gradient accumulation, a step contains several batches. A script without `torch.optim`
  treats each backward as a step.
- Exit codes are passed through. If the script crashes, everything recorded before the crash is
  kept, and the run is marked as failed with the traceback.

On SLURM, change one line in your job script:

```bash
srun tvis run --steps 3 train.py --lr 3e-4      # was: srun python train.py --lr 3e-4
```

When it finishes, `tvis run` prints the command for viewing the recording from your laptop.

## Viewing runs

`tvis open` starts the viewer on `127.0.0.1` on your laptop and opens your browser.

```bash
tvis open ./tvis_runs                           # runs on this machine
tvis open gpu-login:~/proj/tvis_runs            # any ssh host or ~/.ssh/config alias
tvis open --via bastion --via login c42:/scratch/me/tvis_runs    # explicit jump hosts
```

For remote runs, tvis runs `ssh <host> tvis agent --stdio` and reads the run over that SSH
session's stdin/stdout. No port is opened on the cluster, so it works where port forwarding is
disabled. Only the data you look at is transferred. Hops come from your `~/.ssh/config`
(`ProxyJump`) or from `--via`. tvis reuses SSH connections, so on clusters with 2FA you're asked
to authenticate at most once.

If `tvis` isn't on the remote `PATH` for non-interactive shells (common with conda or venvs), tell
it how to start:

```bash
tvis open gpu:~/proj/tvis_runs --remote-tvis "~/miniconda3/envs/train/bin/tvis"
```

Save a target you use often:

```bash
tvis target add gpu --host c42 --via bastion,login --path ~/proj/tvis_runs --remote-tvis ~/venv/bin/tvis
tvis open gpu
```

## The viewer

| Part | What it shows |
|---|---|
| Phase bar | time split into data → forward → loss → backward → optimizer for the batch or step |
| Calls | the call tree. Filter by name or kind. Hide calls like a given one with ⊘ (their children stay visible). Repeated calls are grouped. |
| Overview | what was captured, model and data summary, per-batch loss and time, notices |
| Samples | every datapoint with target, prediction, confidence and loss. Click one to follow it from the raw input through every stage, and every tensor in the UI is sliced to that sample. |
| Timeline | flame chart of forward and backward |
| Source | your code as it ran, with the shape each line produced, the calls each line made, and per-function times |
| Layers | sortable table of all calls with output σ, % zeros, ‖grad‖ and times. NaN/Inf, vanishing gradients and dead units are flagged. |
| Step | micro-batches, and per-parameter ‖w‖, ‖grad‖, ‖Δw‖, ‖Δw‖/‖w‖ |
| Inspector | the selected call: definition and call site, times, inputs and outputs. Each tensor has stats, a histogram, a value/gradient heatmap with dimension sliders, or an image view. |

Keys: `j` / `k` next / previous batch, `Esc` stop following a sample, `/` filter.

## Accuracy

- **Observing never changes training.** Losses and parameters are bit-identical with and without
  `tvis run`. The test suite checks this in-process and across processes.
- **Gradients are the real ones.** Captured activation gradients equal autograd's, and parameter
  updates match `p_after - p_before` exactly.
- **Timing excludes tvis.** Forward times and timeline positions have tvis's own capture work
  subtracted. On CUDA they are device times (CUDA events), not Python wall time.
- **Approximations are labelled.**
  - Backward time per call is inferred from when gradients arrive.
  - The batch dimension of merged tensors such as `[B*T, V]` is inferred.
  - Layers that mix samples (BatchNorm in training mode) are flagged in per-sample views.

## Limitations (v1)

- One process, one device. No DDP, FSDP or `torchrun` yet.
- `torch.compile`d modules may hide their internals from the hooks.
- With AMP `GradScaler`, recorded gradients are the scaled gradients.
- Tensors above `--max-elems` (default 2M elements) keep stats and their leading rows only.
- Image and text decoding support: torchvision transforms (v1/v2), PIL, HuggingFace tokenizers.

## Development

See [AGENTS.md](AGENTS.md) for commands, layout, invariants and test conventions.

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
```
