"""Per-sample view of a batch: dataset index, raw input, target, prediction, per-sample loss.

Built at batch finalisation inside the training process (where the tokenizer and class names are
available), from the recorded data/sample/pipeline calls and the last loss call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from tvis.capture import calls as K
from tvis.capture.values import iter_tensor_refs

if TYPE_CHECKING:
    from tvis.capture.session import Batch, Session
    from tvis.capture.tensors import TensorRecord

TOP_K = 5
MAX_TEXT = 400
_CLASSIFICATION_LOSSES = ("cross_entropy", "nll_loss")


def build_samples(
    session: Session, batch: Batch, records: dict[str, TensorRecord], batch_size: int | None
) -> list[dict[str, Any]]:
    if not batch_size:
        return []
    samples: list[dict[str, Any]] = [{"position": i} for i in range(batch_size)]
    if batch.indices is not None and len(batch.indices) == batch_size:
        for sample, index in zip(samples, batch.indices, strict=True):
            sample["index"] = index

    children: dict[int | None, list] = {}
    for call in batch.calls:
        children.setdefault(call.parent, []).append(call)

    sample_calls = [c for c in batch.calls if c.kind == K.SAMPLE]
    if len(sample_calls) == batch_size:
        for sample, call in zip(samples, sample_calls, strict=True):
            sample["sample_call"] = call.id
            raw = _raw_input(call, children)
            if raw is not None:
                sample["raw"] = raw

    # a tokenizer applied to the whole batch (typically in collate_fn): one row of tokens per sample
    batch_tokenize = [
        c for c in batch.calls if c.kind == K.TOKENIZE and len(c.extra.get("tokens", [])) == batch_size
    ]
    if batch_tokenize:
        call = batch_tokenize[-1]
        for sample, tokens in zip(samples, call.extra["tokens"], strict=True):
            sample["tokens"] = {"call": call.id, "tokens": tokens}

    data_call = next((c for c in batch.calls if c.kind == K.DATA and c.outputs), None)
    if data_call is not None:
        first = next(iter_tensor_refs(data_call.outputs[0]["value"]), None)
        if first is not None:
            for sample in samples:
                sample["input"] = {"tid": first["tid"], "batch": first["batch"]}

    loss_call = next(
        (c for c in reversed(batch.calls) if c.kind == K.LOSS and "_per_element" in c.extra), None
    )
    if loss_call is not None:
        try:
            _attach_loss_and_predictions(session, loss_call, records, samples, batch_size)
        finally:
            loss_call.extra.pop("_per_element", None)
    for call in batch.calls:
        call.extra.pop("_per_element", None)
    return samples


def _raw_input(sample_call: Any, children: dict[int | None, list]) -> dict[str, Any] | None:
    """The input of the first pipeline/tokenize stage under a sample call; otherwise the item the
    dataset returned, if it is (or starts with) text or an image."""
    stack = list(children.get(sample_call.id, []))
    while stack:
        call = stack.pop(0)
        if call.kind in (K.PIPELINE, K.TOKENIZE) and call.inputs:
            return {"call": call.id, "value": call.inputs[0]["value"]}
        stack[:0] = children.get(call.id, [])
    if sample_call.outputs:
        item = sample_call.outputs[0]["value"]
        if item.get("kind") == "seq" and item["items"]:
            item = item["items"][0]
        if item.get("kind") in ("text", "image"):
            return {"call": sample_call.id, "value": item}
    return None


def _attach_loss_and_predictions(
    session: Session,
    loss_call: Any,
    records: dict[str, TensorRecord],
    samples: list[dict[str, Any]],
    batch_size: int,
) -> None:
    per_element: torch.Tensor = loss_call.extra["_per_element"].float()
    target = _input_tensor(loss_call, "target", records)
    ignore_index = loss_call.extra.get("ignore_index")

    per_sample = _per_sample_loss(per_element, target, ignore_index, batch_size)
    if per_sample is not None:
        for sample, value in zip(samples, per_sample.tolist(), strict=True):
            sample["loss"] = value if value == value else None  # NaN → None
        loss_call.extra["per_sample_loss"] = True

    if loss_call.name not in _CLASSIFICATION_LOSSES:
        return
    logits = _input_tensor(loss_call, "input", records)
    if logits is None or target is None or logits.dim() < 2:
        return
    if logits.dim() == 3 and logits.shape[0] == batch_size and target.dim() == 2:
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[1])  # [B, C, T] layout
    logits = logits.float()
    target = target.reshape(-1)
    if logits.shape[0] != target.shape[0]:
        return
    is_log_probs = loss_call.name == "nll_loss"
    probs = logits.exp() if is_log_probs else logits.softmax(-1)

    if logits.shape[0] == batch_size:
        _classification(session, probs, target, samples)
    elif logits.shape[0] % batch_size == 0:
        _sequence(session, probs, target, ignore_index, samples, batch_size)


def _per_sample_loss(
    per_element: torch.Tensor, target: torch.Tensor | None, ignore_index: int | None, batch_size: int
) -> torch.Tensor | None:
    if per_element.dim() == 0 or per_element.numel() % batch_size:
        return None
    rows = per_element.reshape(batch_size, -1)
    if ignore_index is not None and target is not None and target.numel() == per_element.numel():
        valid = (target.reshape(batch_size, -1) != ignore_index).float()
        return (rows * valid).sum(1) / valid.sum(1).clamp_min(1)
    return rows.mean(1)


def _classification(session: Session, probs: torch.Tensor, target: torch.Tensor, samples: list[dict]) -> None:
    classes = session.data_info.get("dataset", {}).get("classes")
    k = min(TOP_K, probs.shape[1])
    top_p, top_i = probs.topk(k, dim=1)

    def name(i: int) -> str | None:
        return classes[i] if classes and 0 <= i < len(classes) else None

    for row, sample in enumerate(samples):
        target_id = int(target[row])
        predicted = int(top_i[row, 0])
        sample["target"] = {"id": target_id, "name": name(target_id)}
        sample["prediction"] = {
            "id": predicted,
            "name": name(predicted),
            "p": float(top_p[row, 0]),
            "p_target": float(probs[row, target_id]) if 0 <= target_id < probs.shape[1] else None,
            "correct": predicted == target_id,
            "topk": [
                {"id": int(i), "name": name(int(i)), "p": float(p)}
                for p, i in zip(top_p[row].tolist(), top_i[row].tolist(), strict=True)
            ],
        }


def _sequence(
    session: Session,
    probs: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int | None,
    samples: list[dict],
    batch_size: int,
) -> None:
    predicted = probs.argmax(-1).reshape(batch_size, -1)
    target = target.reshape(batch_size, -1)
    tokenizer = session.tokenizer = session.tokenizer or _find_tokenizer()
    for row, sample in enumerate(samples):
        valid = (
            target[row] != ignore_index
            if ignore_index is not None
            else torch.ones_like(target[row], dtype=torch.bool)
        )
        pred_ids = predicted[row][valid].tolist()
        target_ids = target[row][valid].tolist()
        n = max(1, len(target_ids))
        sample["target"] = {"ids": target_ids[:512], "text": _decode(tokenizer, target_ids)}
        sample["prediction"] = {
            "ids": pred_ids[:512],
            "text": _decode(tokenizer, pred_ids),
            "token_accuracy": sum(int(a == b) for a, b in zip(pred_ids, target_ids, strict=True)) / n,
        }


def _decode(tokenizer: Any, ids: list[int]) -> str | None:
    if tokenizer is None:
        return None
    try:
        return tokenizer.decode(ids)[:MAX_TEXT]
    except Exception:
        return None


def _find_tokenizer() -> Any:
    """Fallback when the script tokenised before the captured steps: find a live HF tokenizer."""
    import gc
    import sys

    if "transformers" not in sys.modules:
        return None
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    for obj in gc.get_objects():
        if issubclass(type(obj), PreTrainedTokenizerBase):
            return obj
    return None


def _input_tensor(call: Any, name: str, records: dict[str, TensorRecord]) -> torch.Tensor | None:
    for entry in call.inputs:
        if entry["name"] == name and entry["value"].get("kind") == "tensor":
            record = records.get(entry["value"]["tid"])
            if record is not None and record.stored == "full":
                return record.value
    return None
