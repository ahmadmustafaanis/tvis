# tvis

A step-through debugger for PyTorch training.

Run your unmodified training script under `tvis run`. It records the first few optimizer steps end
to end, from the raw image file or text string through every transform, layer and function of your
model to the output, loss, gradients and optimizer update. Then it exits, and you inspect the
recording in a browser.

```bash
python train.py --lr 3e-4                    # normal training: tvis is not involved at all
tvis run --steps 3 train.py --lr 3e-4        # debug run: same script, no code changes
tvis open mycluster:~/proj/runs              # on your laptop: opens the viewer
```

> Status: early development. See [SCOPE.md](SCOPE.md) for what v1 covers.

## Install

```bash
pip install -e .            # in your training environment (needs torch) and on your laptop (no torch needed)
```
