"""Raw input → pipeline → batch → per-sample decoding."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from tests.capture import datasets, models


def classify(loader_factory, model_factory=lambda: models.ToyMLP(), steps: int = 1):
    """Training loop over a DataLoader built *inside* the capture (as in a real script)."""

    def loop():
        model = model_factory()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        for x, y in loader_factory():
            F.cross_entropy(model(x), y).backward()
            opt.step()
            opt.zero_grad()

    return loop


class TestDataLoader:
    def test_worker_processes_are_disabled_and_reported(self, capture):
        result = capture(classify(lambda: DataLoader(datasets.VectorDataset(), batch_size=4, num_workers=2)))

        assert result.meta["data"]["loader"]["num_workers"] == 2  # what the script asked for
        assert any("num_workers=2 set to 0" in n["message"] for n in result.meta["notices"])

    def test_sampled_indices_are_recorded_per_batch(self, capture):
        result = capture(classify(lambda: DataLoader(datasets.VectorDataset(), batch_size=4)), steps=2)

        assert result.batch(0)["indices"] == [0, 1, 2, 3]
        assert result.batch(1)["indices"] == [4, 5, 6, 7]
        assert result.batch(0)["batch_size"] == 4

    def test_each_dataset_item_is_a_sample_call_under_the_fetch(self, capture):
        result = capture(classify(lambda: DataLoader(datasets.VectorDataset(), batch_size=4)))

        fetch = result.call(kind="data", name="DataLoader")
        samples = result.calls(kind="sample")
        assert [s["extra"]["index"] for s in samples] == [0, 1, 2, 3]
        assert {s["parent"] for s in samples} == {fetch["id"]}
        assert [c["name"] for c in result.children(samples[0])] == ["__getitem__"]  # user code, traced

    def test_collated_batch_is_the_fetch_output_and_has_a_batch_dimension(self, capture):
        result = capture(classify(lambda: DataLoader(datasets.VectorDataset(), batch_size=4)))

        fetch = result.call(kind="data", name="DataLoader")
        x_ref = fetch["outputs"][0]["value"]["items"][0]
        assert result.tensor(x_ref)["shape"] == [4, 6]
        assert result.tensor(x_ref)["batch_dim"] == {"dim": 0, "block": 1, "inferred": False}

    def test_per_sample_items_before_collation_have_no_batch_dimension(self, capture):
        result = capture(classify(lambda: DataLoader(datasets.VectorDataset(), batch_size=4)))

        item = result.calls(kind="sample")[0]["outputs"][0]["value"]["items"][0]
        assert result.tensor(item)["shape"] == [6]
        assert "batch_dim" not in result.tensor(item)

    def test_dataset_metadata_includes_class_names(self, capture):
        result = capture(classify(lambda: DataLoader(datasets.VectorDataset(), batch_size=4)))

        assert result.meta["data"]["dataset"] == {
            "cls": "VectorDataset",
            "len": 16,
            "classes": datasets.LABELS,
        }

    def test_datasets_with_batched_fetch_still_yield_identical_batches(self, capture):
        captured: list = []

        def loop():
            for x, y in DataLoader(datasets.BatchedFetchDataset(), batch_size=4):
                captured.append((x, y))
                models.ToyMLP()(x).sum().backward()

        result = capture(loop, steps=1, max_batches=2)
        plain = list(DataLoader(datasets.BatchedFetchDataset(), batch_size=4))

        assert all(
            torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(captured, plain, strict=False)
        )
        assert len(result.calls(kind="sample")) == 4

    def test_iterable_datasets_are_supported_without_indices(self, capture):
        result = capture(classify(lambda: DataLoader(datasets.StreamDataset(), batch_size=4)), steps=2)

        assert result.batch(0)["indices"] is None
        assert result.calls(kind="sample") == []
        assert result.batch(0)["batch_size"] == 4  # inferred from the collated batch


class TestLossAndPredictions:
    def test_loss_value_is_returned_unchanged(self, capture):
        logits, target = torch.randn(4, 3), torch.tensor([0, 2, 1, 1])
        expected = F.cross_entropy(logits, target, weight=torch.tensor([1.0, 2.0, 0.5]))
        got: list = []

        def loop():
            model = models.ToyMLP()
            model(torch.randn(4, 6))  # opens a batch
            got.append(F.cross_entropy(logits, target, torch.tensor([1.0, 2.0, 0.5])))  # positional weight

        capture(loop, steps=1)

        assert torch.equal(got[0], expected)

    def test_per_sample_loss_matches_unreduced_cross_entropy(self, capture):
        torch.manual_seed(0)
        model = models.ToyMLP()
        loader = DataLoader(datasets.VectorDataset(), batch_size=4)
        x, y = next(iter(loader))
        expected = F.cross_entropy(model(x), y, reduction="none")

        result = capture(classify(lambda: loader, model_factory=lambda: model))

        losses = [s["loss"] for s in result.batch(0)["samples"]]
        assert losses == pytest.approx(expected.tolist(), rel=1e-6)

    def test_classification_predictions_use_dataset_class_names(self, capture):
        result = capture(classify(lambda: DataLoader(datasets.VectorDataset(), batch_size=4)))

        sample = result.batch(0)["samples"][1]
        assert sample["index"] == 1
        assert sample["target"] == {"id": 1, "name": "dog"}
        prediction = sample["prediction"]
        assert prediction["name"] == datasets.LABELS[prediction["id"]]
        assert prediction["correct"] == (prediction["id"] == 1)
        assert [t["p"] for t in prediction["topk"]] == sorted(
            (t["p"] for t in prediction["topk"]), reverse=True
        )
        assert sum(t["p"] for t in prediction["topk"]) == pytest.approx(1.0, abs=1e-5)


class TestImagePipeline:
    @pytest.fixture
    def transforms(self):
        return pytest.importorskip("torchvision.transforms")

    def image_loop(self, transform):
        model_factory = lambda: nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))  # noqa: E731
        return classify(lambda: DataLoader(datasets.ImageDataset(transform), batch_size=4), model_factory)

    def test_compose_records_each_transform_stage_in_order(self, capture, transforms):
        transform = transforms.Compose(
            [transforms.Resize((8, 8)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.2] * 3)]
        )
        result = capture(self.image_loop(transform))

        pipeline = result.calls(kind="pipeline")[0]
        stages = result.children(pipeline)
        assert [s["name"] for s in stages] == ["Resize", "ToTensor", "Normalize"]
        assert stages[0]["inputs"][0]["value"]["kind"] == "image"
        assert result.tensor(stages[0]["inputs"][0]["value"])["shape"] == [10, 12, 3]
        assert result.tensor(stages[1]["outputs"][0]["value"])["shape"] == [3, 8, 8]

    def test_normalize_parameters_are_remembered_for_display(self, capture, transforms):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5, 0.4, 0.3], [0.2, 0.2, 0.2])]
        )
        transform = transforms.Compose([transforms.Resize((8, 8)), transform])
        result = capture(self.image_loop(transform))

        normalize = result.meta["data"]["normalize"]
        assert normalize["mean"] == pytest.approx([0.5, 0.4, 0.3])
        assert normalize["std"] == pytest.approx([0.2, 0.2, 0.2])

    def test_samples_link_raw_image_to_model_input(self, capture, transforms):
        transform = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
        result = capture(self.image_loop(transform))

        sample = result.batch(0)["samples"][2]
        assert sample["raw"]["value"]["kind"] == "image"
        raw_pixels = result.value(sample["raw"]["value"])
        assert raw_pixels[0, 0].tolist() == [40, 100, 200]  # index 2 → red channel 40
        assert result.tensor(sample["input"])["shape"] == [4, 3, 8, 8]

    def test_transform_internals_are_not_recorded_as_model_modules(self, capture, transforms):
        transform = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
        result = capture(self.image_loop(transform))

        assert {c["name"] for c in result.calls(kind="module")} == {"Sequential", "0", "1"}

    def test_v2_compose_is_recorded(self, capture):
        v2 = pytest.importorskip("torchvision.transforms.v2")
        transform = v2.Compose([v2.Resize((8, 8)), v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True)])
        result = capture(self.image_loop(transform))

        stages = result.children(result.calls(kind="pipeline")[0])
        assert [s["name"] for s in stages] == ["Resize", "PILToTensor", "ToDtype"]


class TestTextPipeline:
    @pytest.fixture
    def tokenizer(self):
        pytest.importorskip("transformers")
        return datasets.build_tokenizer()

    def lm_loop(self, tokenizer):
        def loop():
            torch.manual_seed(0)
            embed, head = nn.Embedding(len(tokenizer), 8), nn.Linear(8, len(tokenizer))
            model = nn.Sequential(embed, head)
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            loader = DataLoader(
                datasets.TextDataset(), batch_size=2, collate_fn=datasets.text_collate(tokenizer)
            )
            for ids, labels in loader:
                logits = model(ids)
                F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
                ).backward()
                opt.step()

        return loop

    def test_tokenizer_call_records_text_and_tokens(self, capture, tokenizer):
        result = capture(self.lm_loop(tokenizer))

        tok = result.call(kind="tokenize")
        assert tok["inputs"][0]["value"]["items"][0]["text"] == "the cat sat"
        assert tok["extra"]["tokens"][0][:3] == ["the", "cat", "sat"]

    def test_samples_carry_raw_text_and_their_own_tokens(self, capture, tokenizer):
        result = capture(self.lm_loop(tokenizer))

        first, second = result.batch(0)["samples"]
        assert first["raw"]["value"]["text"] == "the cat sat"
        assert second["raw"]["value"]["text"] == "a dog ran far away"
        assert first["tokens"]["tokens"][:3] == ["the", "cat", "sat"]

    def test_sequence_targets_are_decoded_without_padding(self, capture, tokenizer):
        result = capture(self.lm_loop(tokenizer))

        first, second = result.batch(0)["samples"]
        assert first["target"]["text"] == "the cat sat"
        assert second["target"]["text"] == "a dog ran far away"
        assert 0.0 <= first["prediction"]["token_accuracy"] <= 1.0
        assert len(first["prediction"]["ids"]) == 3

    def test_per_sample_loss_ignores_padding_positions(self, capture, tokenizer):
        result = capture(self.lm_loop(tokenizer))

        call = result.call(kind="loss", name="cross_entropy")
        logits = torch.from_numpy(result.value(result.input_ref(call, "input")))
        target = torch.from_numpy(result.value(result.input_ref(call, "target")))
        per_token = F.cross_entropy(logits, target, ignore_index=-100, reduction="none").reshape(2, -1)
        valid = (target.reshape(2, -1) != -100).float()
        expected = (per_token * valid).sum(1) / valid.sum(1)

        assert [s["loss"] for s in result.batch(0)["samples"]] == pytest.approx(expected.tolist(), rel=1e-5)
        assert result.tensor(result.input_ref(call, "input"))["batch_dim"] == {
            "dim": 0,
            "block": 5,
            "inferred": True,
        }
