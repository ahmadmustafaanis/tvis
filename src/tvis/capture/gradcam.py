"""True Grad-CAM, computed during capture.

Grad-CAM needs d(class score)/d(activation), which the training backward never computes (it gives
dL/d(activation)). At the loss call the graph is still alive, so for every 4-D conv output in the
batch we run ``torch.autograd.grad`` of the predicted-class and target-class scores with
``retain_graph=True``. ``autograd.grad`` never writes ``.grad`` and tvis's own gradient hooks are
suppressed during the pass, so training numerics are untouched (checked by tests).

cam[b] = ReLU( sum_c mean_hw(dS/dA[b, c]) * A[b, c] ), normalised to [0, 1] per sample.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from tvis.capture import calls as K
from tvis.capture.values import capture_value, iter_tensor_refs

if TYPE_CHECKING:
    from tvis.capture.session import Session

_CLASSIFICATION_LOSSES = ("cross_entropy", "nll_loss")


def compute(session: Session, loss_call: Any, logits: Any, target: Any) -> None:
    if loss_call.name not in _CLASSIFICATION_LOSSES or not _is_class_logits(logits, target):
        return
    batch = session.batch
    if batch is None:
        return
    size = logits.shape[0]
    activations = []
    for call in batch.calls:
        if call.kind != K.MODULE or not call.cls or "Conv" not in call.cls or not call.outputs:
            continue
        ref = next(iter_tensor_refs(call.outputs[0]["value"]), None)
        tensor = session.registry.live_tensor(ref["tid"]) if ref else None
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.dim() == 4
            and tensor.shape[0] == size
            and tensor.requires_grad
        ):
            activations.append((call, tensor))
    if not activations:
        return
    classes = {"pred": logits.detach().argmax(1)}
    if target.min() >= 0 and target.max() < logits.shape[1]:
        classes["target"] = target
    tensors = [t for _, t in activations]
    for which, cls in classes.items():
        score = logits.gather(1, cls.reshape(-1, 1).long()).sum()
        with session.suppressed_grad_hooks():
            grads = torch.autograd.grad(score, tensors, retain_graph=True, allow_unused=True)
        for (call, act), grad in zip(activations, grads, strict=True):
            if grad is None:
                continue
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * act.detach()).sum(dim=1))
            peak = cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
            ref = capture_value((cam / peak).detach(), session.registry, hook_grad=False)
            call.extra.setdefault("gradcam", {})[which] = ref
            call.extra["gradcam_layer"] = True


def _is_class_logits(logits: Any, target: Any) -> bool:
    return (
        isinstance(logits, torch.Tensor)
        and isinstance(target, torch.Tensor)
        and logits.dim() == 2
        and logits.requires_grad
        and target.dim() == 1
        and target.shape[0] == logits.shape[0]
        and not target.is_floating_point()
    )
