from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from tests.viewer.conftest import RUN_ID, VOCAB
from tvis.store import schema
from tvis.viewer import views
from tvis.viewer.api import Api, ApiError


@pytest.fixture
def api(run_root: Path) -> Api:
    return Api(run_root)


def floats(view: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(view["data"]), dtype="<f4").reshape(view["shape"])


class TestArray:
    def test_returns_every_value_with_its_shape(self, api: Api):
        view = api.dispatch("array", {"run": RUN_ID, "batch": 0, "tid": "t1"})

        np.testing.assert_array_equal(floats(view), np.arange(30, dtype=np.float32).reshape(2, 3, 5))

    def test_can_select_one_sample(self, api: Api):
        view = api.dispatch("array", {"run": RUN_ID, "batch": 0, "tid": "t1", "sample": 1})

        assert view["shape"] == [3, 5] and view["min"] == 15.0

    def test_refuses_tensors_too_large_to_send(self):
        with pytest.raises(views.ViewError):
            views.full_array(np.zeros(views.MAX_ARRAY_ELEMS + 1, np.float32))


class TestThumbnail:
    def test_channel_tensors_are_summarised_by_mean_absolute_value(self):
        chw = np.stack([np.full((2, 2), -2.0), np.full((2, 2), 4.0)]).astype(np.float32)

        view = views.thumbnail(chw)

        assert (view["rows"], view["cols"]) == (2, 2)
        assert view["min"] == view["max"] == 3.0

    def test_two_dimensional_samples_are_shown_as_they_are(self, api: Api):
        view = api.dispatch("thumb", {"run": RUN_ID, "batch": 0, "tid": "t1", "sample": 0})

        assert (view["rows"], view["cols"]) == (3, 5) and view["min"] == 0.0


class TestLossDetail:
    def test_per_element_loss_is_laid_out_per_sample(self, api: Api):
        detail = api.dispatch("loss_detail", {"run": RUN_ID, "batch": 0})

        assert (detail["rows"], detail["cols"]) == (2, 3)
        assert detail["per_element"][1] == pytest.approx([1.5, 0.3, 0.01])
        assert detail["loss"] == 1.25

    def test_padding_positions_are_marked_invalid(self, api: Api):
        detail = api.dispatch("loss_detail", {"run": RUN_ID, "batch": 0})

        assert detail["valid"] == [[True, True, False], [True, True, True]]

    def test_predictions_and_target_probabilities_come_from_the_logits(self, api: Api):
        detail = api.dispatch("loss_detail", {"run": RUN_ID, "batch": 0})

        assert detail["pred_ids"] == [[0, 1, 2], [0, 3, 1]]
        expected = np.exp(2) / (np.exp(2) + 3)  # row 0: logits [2,0,0,0], target 0
        assert detail["p_target"][0][0] == pytest.approx(expected)

    def test_batches_without_a_loss_are_reported(self, run_root: Path):
        path = run_root / RUN_ID / schema.BATCHES_DIR / schema.batch_dir_name(0) / schema.BATCH_FILE
        doc = json.loads(path.read_text())
        doc["calls"] = [c for c in doc["calls"] if c["kind"] != "loss"]
        path.write_text(json.dumps(doc))

        with pytest.raises(ApiError) as info:
            Api(run_root).dispatch("loss_detail", {"run": RUN_ID, "batch": 0})
        assert info.value.status == 404


class TestProjections:
    def test_layer_outputs_are_projected_per_sample_with_labels(self, api: Api):
        result = api.dispatch("projection", {"run": RUN_ID, "path": "enc"})

        assert result["dim"] == 5  # [B, T=3, D=5] pooled over T
        assert [p["label"] for p in result["points"]] == ["dog", "cat"]
        assert [p["correct"] for p in result["points"]] == [True, False]
        assert sum(result["explained"]) == pytest.approx(1.0)

    def test_unknown_layer_is_a_404(self, api: Api):
        with pytest.raises(ApiError) as info:
            api.dispatch("projection", {"run": RUN_ID, "path": "nope"})
        assert info.value.status == 404

    def test_weight_rows_are_projected_and_labelled_with_the_vocabulary(self, api: Api):
        result = api.dispatch("weight_projection", {"run": RUN_ID, "step": 0, "param": "embed.weight"})

        assert (result["rows"], result["dim"]) == (5, 3)
        assert [p["label"] for p in result["points"]] == VOCAB


class TestPca:
    def test_recovers_the_dominant_direction(self):
        t = np.linspace(-1, 1, 50)
        x = np.stack([t, 2 * t, np.zeros_like(t)], axis=1)

        coords, ratio = views.pca2(x)

        assert ratio[0] == pytest.approx(1.0)
        assert np.abs(coords[:, 1]).max() == pytest.approx(0.0, abs=1e-9)
        assert np.corrcoef(coords[:, 0], t)[0, 1] ** 2 == pytest.approx(1.0)

    def test_needs_at_least_two_vectors(self):
        with pytest.raises(views.ViewError):
            views.pca2(np.ones((1, 4)))

    @pytest.mark.parametrize(
        ("shape", "pooled"),
        [((2, 3, 4, 4), (2, 3)), ((2, 7, 5), (2, 5)), ((2, 5), (2, 5))],
        ids=["conv-spatial-mean", "sequence-mean", "already-flat"],
    )
    def test_pooling_gives_one_vector_per_sample(self, shape, pooled):
        assert views.pool_features(np.ones(shape)).shape == pooled
