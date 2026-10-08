"""Data-side instrumentation: raw input → per-sample pipeline → collated batch, plus loss functions.

* ``DataLoader``: workers are forced to the main process during ``tvis run`` so the whole pipeline
  is observable; each fetch becomes a ``data`` call with the sampled indices; each ``dataset[i]``
  becomes a ``sample`` call.
* torchvision ``Compose`` (v1 and v2): a ``pipeline`` call with one ``transform`` call per stage.
  ``Normalize`` parameters are remembered so the viewer can un-normalise images.
* transformers tokenizers: a ``tokenize`` call with the raw text, ids and token strings.
* ``torch.nn.functional`` losses: a ``loss`` call plus the unreduced per-element loss, used to
  compute exact per-sample losses.

torchvision / transformers are patched lazily (only if the script has imported them).
"""

from __future__ import annotations

import inspect
import sys
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tvis.capture import calls as K
from tvis.capture.session import functools_wraps

if TYPE_CHECKING:
    from tvis.capture.session import Session

LOSS_FUNCTIONS = (
    "cross_entropy",
    "nll_loss",
    "mse_loss",
    "l1_loss",
    "smooth_l1_loss",
    "huber_loss",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
)
MAX_TOKENS_PER_SAMPLE = 512
MAX_CLASSES = 2000


class DataInstrumentation:
    def __init__(self, session: Session):
        self.session = session
        self._patched_libraries: set[str] = set()

    def install(self) -> None:
        s = self.session
        s.patcher.set(DataLoader, "__init__", self._wrap_loader_init(DataLoader.__init__))
        s.patcher.set(DataLoader, "__iter__", self._wrap_loader_iter(DataLoader.__iter__))
        for name in LOSS_FUNCTIONS:
            original = getattr(F, name, None)
            if original is not None:
                s.patcher.set(F, name, self._wrap_loss(name, original))
        self.patch_libraries()

    def uninstall(self) -> None:
        pass  # all patches are owned by session.patcher

    # -- DataLoader ----------------------------------------------------------------------------
    def _wrap_loader_init(self, original: Any) -> Any:
        session = self.session

        def __init__(loader: DataLoader, *args: Any, **kwargs: Any) -> None:
            original(loader, *args, **kwargs)
            if session.active and not session.done and loader.num_workers > 0:
                session.notice(
                    "data",
                    f"DataLoader num_workers={loader.num_workers} set to 0 so the data pipeline can be traced",
                )
                loader.__dict__["_tvis_num_workers"] = loader.num_workers
                loader.num_workers = 0

        functools_wraps(__init__, original)
        return __init__

    def _wrap_loader_iter(self, original: Any) -> Any:
        session = self.session

        def __iter__(loader: DataLoader) -> Any:
            iterator = original(loader)
            if not session.active or session.done:
                return iterator
            return _LoaderIterator(iterator, loader, session, self)

        functools_wraps(__iter__, original)
        return __iter__

    def describe_loader(self, loader: DataLoader) -> None:
        s = self.session
        if s.data_info.get("dataset"):
            return
        dataset = loader.dataset
        info: dict[str, Any] = {"cls": type(dataset).__name__}
        try:
            info["len"] = len(dataset)  # type: ignore[arg-type]
        except TypeError:
            info["len"] = None
        classes = getattr(dataset, "classes", None)
        if isinstance(classes, (list, tuple)) and classes and all(isinstance(c, str) for c in classes):
            info["classes"] = list(classes[:MAX_CLASSES])
        s.data_info["dataset"] = info
        s.data_info["loader"] = {
            "batch_size": loader.batch_size,
            "num_workers": loader.__dict__.get("_tvis_num_workers", loader.num_workers),
            "drop_last": loader.drop_last,
            "sampler": type(loader.sampler).__name__,
        }

    # -- libraries -----------------------------------------------------------------------------
    def patch_libraries(self) -> None:
        if "torchvision" in sys.modules and "torchvision" not in self._patched_libraries:
            self._patched_libraries.add("torchvision")
            with self.session.guard("patch torchvision"):
                self._patch_torchvision()
        if "transformers" in sys.modules and "transformers" not in self._patched_libraries:
            self._patched_libraries.add("transformers")
            with self.session.guard("patch transformers"):
                self._patch_transformers()

    def _patch_torchvision(self) -> None:
        from torchvision import transforms as v1

        self.session.patcher.set(v1.Compose, "__call__", self._wrap_compose_v1(v1.Compose.__call__))
        try:
            from torchvision.transforms import v2
        except ImportError:
            return
        self.session.patcher.set(v2.Compose, "forward", self._wrap_compose_v2(v2.Compose.forward))

    def _wrap_compose_v1(self, original: Any) -> Any:
        session = self.session

        def __call__(compose: Any, img: Any) -> Any:
            if not session.recording:
                return original(compose, img)
            call = session.open_call(K.PIPELINE, "Compose", inputs=[("input", img)], hook_grad=False)
            for transform in compose.transforms:
                img = self._run_transform(transform, (img,), single=True)
            session.close_call(call, outputs=[("output", img)], hook_grad=False)
            return img

        functools_wraps(__call__, original)
        return __call__

    def _wrap_compose_v2(self, original: Any) -> Any:
        session = self.session

        def forward(compose: Any, *inputs: Any) -> Any:
            if not session.recording:
                return original(compose, *inputs)
            needs_unpacking = len(inputs) > 1
            call = session.open_call(K.PIPELINE, "Compose", inputs=[("input", inputs)], hook_grad=False)
            outputs: Any = inputs[0] if not needs_unpacking else inputs
            for transform in compose.transforms:
                outputs = self._run_transform(transform, inputs, single=False)
                inputs = outputs if needs_unpacking else (outputs,)
            session.close_call(call, outputs=[("output", outputs)], hook_grad=False)
            return outputs

        functools_wraps(forward, original)
        return forward

    def _run_transform(self, transform: Any, inputs: tuple[Any, ...], *, single: bool) -> Any:
        s = self.session
        self._remember_normalize(transform)
        call = s.open_call(
            K.TRANSFORM,
            type(transform).__name__,
            inputs=[("input", inputs[0] if len(inputs) == 1 else inputs)],
            hook_grad=False,
            extra={"repr": repr(transform)[:300]},
        )
        s._tl.suppress_modules = getattr(s._tl, "suppress_modules", 0) + 1
        try:
            out = transform(*inputs)
        finally:
            s._tl.suppress_modules -= 1
        s.close_call(call, outputs=[("output", out)], hook_grad=False)
        return out

    def _remember_normalize(self, transform: Any) -> None:
        if type(transform).__name__ != "Normalize" or "normalize" in self.session.data_info:
            return
        mean, std = getattr(transform, "mean", None), getattr(transform, "std", None)
        if mean is not None and std is not None:
            self.session.data_info["normalize"] = {
                "mean": [float(v) for v in torch.as_tensor(mean).flatten()],
                "std": [float(v) for v in torch.as_tensor(std).flatten()],
            }

    def _patch_transformers(self) -> None:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        self.session.patcher.set(
            PreTrainedTokenizerBase, "__call__", self._wrap_tokenizer(PreTrainedTokenizerBase.__call__)
        )

    def _wrap_tokenizer(self, original: Any) -> Any:
        session = self.session

        def __call__(tokenizer: Any, *args: Any, **kwargs: Any) -> Any:
            session.tokenizer = tokenizer
            if not session.recording:
                return original(tokenizer, *args, **kwargs)
            text = args[0] if args else kwargs.get("text")
            call = session.open_call(
                K.TOKENIZE,
                type(tokenizer).__name__,
                inputs=[("text", text)],
                hook_grad=False,
                extra={"name_or_path": getattr(tokenizer, "name_or_path", None)},
            )
            encoding = original(tokenizer, *args, **kwargs)
            if call is not None:
                with session.guard("tokenize"), session.internal():
                    call.extra["tokens"] = _token_strings(tokenizer, encoding)
            session.close_call(call, outputs=[("encoding", encoding)], hook_grad=False)
            return encoding

        functools_wraps(__call__, original)
        return __call__

    # -- losses --------------------------------------------------------------------------------
    def _wrap_loss(self, name: str, original: Any) -> Any:
        session = self.session
        signature = inspect.signature(original)

        def loss(input: Any, target: Any, *args: Any, **kwargs: Any) -> Any:
            top = session.stack[-1] if session.stack else None
            reentrant = top is not None and top.kind == K.LOSS and top.name == name
            # torch re-dispatches F.<loss> through handle_torch_function when a function mode is
            # active, which calls this wrapper again from inside the first call.
            if not session.recording or reentrant:
                return original(input, target, *args, **kwargs)
            call = session.open_call(
                K.LOSS,
                name,
                inputs=[("input", input), ("target", target)],
                call_site=session.find_call_site(sys._getframe(1)),
            )
            out = original(input, target, *args, **kwargs)
            if call is not None:
                with session.guard("per-sample loss"), session.internal(), torch.no_grad():
                    bound = signature.bind(input, target, *args, **kwargs)
                    bound.apply_defaults()
                    call.extra["reduction"] = bound.arguments.get("reduction")
                    if "ignore_index" in bound.arguments:
                        call.extra["ignore_index"] = bound.arguments["ignore_index"]
                    for legacy in ("size_average", "reduce"):
                        if legacy in bound.arguments:
                            bound.arguments[legacy] = None
                    bound.arguments["reduction"] = "none"
                    call.extra["_per_element"] = original(*bound.args, **bound.kwargs).detach()
            session.close_call(call, outputs=[("loss", out)])
            return out

        functools_wraps(loss, original)
        return loss


class _LoaderIterator:
    """Wraps a DataLoader iterator: each ``next`` is a ``data`` call."""

    def __init__(self, iterator: Any, loader: DataLoader, session: Session, data: DataInstrumentation):
        self._iterator = iterator
        self._loader = loader
        self._session = session
        self._data = data
        self._indices: list[Any] | None = None
        self._shimmed = False

    def __iter__(self) -> _LoaderIterator:
        return self

    def __len__(self) -> int:
        return len(self._iterator)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._iterator, name)

    def __next__(self) -> Any:
        s = self._session
        if not s.active or s.done:
            return next(self._iterator)
        with s.guard("data fetch"):
            self._data.patch_libraries()
            self._data.describe_loader(self._loader)
            self._shim()
            s.on_data_fetch()  # may raise StopCapture (passes the guard)
        call = s.open_call(
            K.DATA,
            "DataLoader",
            hook_grad=False,
            call_site=s.find_call_site(sys._getframe(1)),
        )
        self._indices = None
        try:
            batch = next(self._iterator)
        except StopIteration:
            if call is not None:
                call.extra["exhausted"] = True
            s.close_call(call)
            raise
        s.close_call(call, outputs=[("batch", batch)], hook_grad=False)
        if call is not None and s.batch is not None and self._indices is not None:
            s.batch.indices = _jsonable_indices(self._indices)
            s.batch.batch_size = len(self._indices)
        return batch

    def _shim(self) -> None:
        if self._shimmed:
            return
        self._shimmed = True
        it = self._iterator
        next_index = getattr(it, "_next_index", None)
        if callable(next_index):

            def _next_index() -> Any:
                index = next_index()
                indices = list(index) if isinstance(index, (list, tuple)) else [index]
                # iterable datasets have no indices: the sampler yields None placeholders
                self._indices = None if all(i is None for i in indices) else indices
                return index

            it._next_index = _next_index
        fetcher = getattr(it, "_dataset_fetcher", None)
        if (
            fetcher is not None
            and hasattr(fetcher, "dataset")
            and hasattr(type(fetcher.dataset), "__getitem__")
            and not isinstance(fetcher.dataset, torch.utils.data.IterableDataset)
        ):
            fetcher.dataset = _DatasetProxy(fetcher.dataset, self._session)


class _DatasetProxy:
    """Records ``dataset[i]`` as a ``sample`` call. Forces per-item fetching (hides __getitems__)."""

    def __init__(self, dataset: Any, session: Session):
        object.__setattr__(self, "_dataset", dataset)
        object.__setattr__(self, "_session", session)

    def __getitem__(self, index: Any) -> Any:
        s = self._session
        call = s.open_call(
            K.SAMPLE,
            f"{type(self._dataset).__name__}[{index}]",
            inputs=[("index", index)],
            hook_grad=False,
            extra={"index": _jsonable_indices([index])[0]},
        )
        item = self._dataset[index]
        s.close_call(call, outputs=[("item", item)], hook_grad=False)
        return item

    def __len__(self) -> int:
        return len(self._dataset)

    def __getattr__(self, name: str) -> Any:
        if name == "__getitems__":
            raise AttributeError(name)
        return getattr(self._dataset, name)


def _jsonable_indices(indices: list[Any]) -> list[Any]:
    out = []
    for i in indices:
        if isinstance(i, torch.Tensor):
            i = i.tolist()
        out.append(i if isinstance(i, (int, str, float, list)) else repr(i))
    return out


def _token_strings(tokenizer: Any, encoding: Any) -> list[list[str]]:
    ids = encoding.get("input_ids")
    if ids is None:
        return []
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], int):
        ids = [ids]
    return [tokenizer.convert_ids_to_tokens(list(row)[:MAX_TOKENS_PER_SAMPLE]) for row in ids[:64]]
