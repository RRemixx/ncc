"""Fast unit tests (CPU, synthetic data)."""
import csv
import pickle

import numpy as np
import pytest
import torch
import yaml

from ncc import NCCConfig
from ncc.experiments import paths
from ncc.experiments.config import load_experiment
from ncc.experiments.evaluate import evaluate
from ncc.experiments.metrics import METRIC_NAMES
from ncc.experiments.run import run_experiment
from ncc.train import LossLog, Patience

NUM_STEPS, LOCATIONS, HORIZON, TEST_STEPS = 30, ["A", "B", "C"], 2, 10

TINY_NCC = {
    "window_size": 3,
    "val_steps": 0,
    "model": {"hidden_dim": 8, "score_encoder_hidden_dim": 8, "score_encoder_num_layers": 1,
              "qhat_encoder_hidden_dim": 8, "qhat_encoder_num_layers": 1, "error_encoder_hidden_dim": 8,
              "alpha_encoder_hidden_dim": 8, "q_fc_hidden_dim": 8},
    "train": {"epochs": 1, "K": 1},
    "retrain": {"period": 3, "epochs": 1},
    "tta": {"enabled": True},
}


def _write_tiny_experiment(root):
    """Synthetic base predictions and a tiny experiment config 'tiny' under root."""
    rng = np.random.RandomState(0)
    base_pred = [({r: rng.uniform(30, 140, 4) for r in LOCATIONS},
                  {r: (rng.uniform(30, 140, 4), "ignored") for r in LOCATIONS}) for _ in range(NUM_STEPS)]
    (root / "results" / "base_pred").mkdir(parents=True)
    with open(root / "results" / "base_pred" / "saved_pred_1.pickle", "wb") as f:
        pickle.dump({"params": {"dataset": "synthetic"}, "base_pred": base_pred}, f)
    (root / "configs" / "experiments").mkdir(parents=True)
    (root / "configs" / "default.yaml").write_text(yaml.safe_dump({"seed": 1, "seeds": [1], "max_steps": 10000}))
    experiment = {"base_pred_id": 1, "horizon": HORIZON, "test_steps": TEST_STEPS, "ncc": TINY_NCC}
    (root / "configs" / "experiments" / "tiny.yaml").write_text(yaml.safe_dump(experiment))


# T1: end to end
@pytest.mark.filterwarnings("ignore:dropout option adds dropout")
def test_run_and_evaluate_end_to_end(tmp_path, monkeypatch):
    _write_tiny_experiment(tmp_path / "repo")
    monkeypatch.setattr(paths, "ROOT", tmp_path / "repo")
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")

    exp = load_experiment("tiny")
    results_path = run_experiment(exp, seed=1, device="cpu")
    csv_path = evaluate(exp, seed=1)

    assert results_path == paths.results("tiny", 1) and results_path.exists()
    assert paths.weights("tiny", 1).exists()
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    assert list(results) == LOCATIONS
    for location in LOCATIONS:
        assert list(results[location]) == [HORIZON]
        entries = results[location][HORIZON]
        assert len(entries) == TEST_STEPS
        q_hats, y_hat, y = entries[0]
        assert q_hats.shape == (len(exp.ncc.alphas),) and y_hat.shape == (1,) and y.shape == (1,)
        assert q_hats.dtype == y_hat.dtype == y.dtype == np.float32
    assert csv_path == paths.csv_dir() / "tiny_1.csv"
    with open(csv_path) as f:
        header = next(csv.reader(f))
    assert header == ["region", "week ahead"] + METRIC_NAMES


# T7: loss log and patience
def test_loss_log_nan_coverage_and_total():
    log = LossLog()
    for parts in [(1.0, 2e7, 3.0, 4.0), (3.0, 2.0, 5.0, 6.0), (2.0, 5.0, float("nan"), 2.0)]:
        log.add([torch.tensor(part) for part in parts])
    total = log.epoch_end()
    # the coverage part 2e7 > 1e7 becomes NaN; each part is a nanmean; the total is their sum
    # (values of LossTracker in 4daa066 on the same steps)
    assert log.epochs["quantile"] == [2.0]
    assert log.epochs["coverage"] == [3.5]
    assert log.epochs["efficiency"] == [4.0]
    assert log.epochs["monotonicity"] == [4.0]
    assert total == 13.5 and log.epochs["total"] == [13.5]


# epoch after which Patience(3) stops, as EarlyStopping(patience=3) in 4daa066 does on the same totals
@pytest.mark.parametrize("totals, stop_epoch", [
    ([5, 4, 4.5, 4.6, 4.7, 1], 5),              # 3 epochs without improvement
    ([5, 6, 7, 4, 8, 9, 10], 7),                # an improvement resets the counter
    ([5, 5, 5, 5, 6, 7, 8], 7),                 # an equal total counts as an improvement
    ([1, float("nan"), 2, 3, 4, 5], 6),         # NaN resets the counter and becomes the best
    ([float("nan"), 1, 2, 3, 4], 5),
    ([5, 4, 3, 2, 1], None),                    # always improving
])
def test_patience_matches_old_early_stopping(totals, stop_epoch):
    patience = Patience(3)
    stopped = next((epoch for epoch, total in enumerate(totals, 1) if patience(total)), None)
    assert stopped == stop_epoch


# NCCConfig.from_dict validation
@pytest.mark.parametrize("change, message", [
    ({"unknown_key": 1}, "unknown key"),
    ({"train": {"K": 1, "lr_typo": 1}}, "unknown key"),
    ({"window_size": None}, "missing required key ncc.window_size"),
    ({"train": {}}, "missing required key ncc.train.K"),
    ({"alphas": [0.1, 0.1, 0.5]}, "strictly ascending"),
    ({"train": {"K": 1, "stages": [{"epochs": 1, "weights": [1, 0, 0]}]}}, "stage_epochs"),
])
def test_config_from_dict_rejects(change, message):
    with pytest.raises(ValueError, match=message):
        NCCConfig.from_dict({**TINY_NCC, **change})


def test_config_from_dict_converts_lists():
    cfg = NCCConfig.from_dict({**TINY_NCC, "alphas": [0.1, 0.5, 1.0],
                               "train": {"K": 1, "stages": [{"epochs": 2, "weights": [1, 0, 0]}]},
                               "retrain": {"stage_epochs": [1]}})
    assert cfg.alphas == (0.1, 0.5, 1.0)
    assert cfg.train.stages[0].epochs == 2 and cfg.train.stages[0].weights == (1, 0, 0)
    assert cfg.retrain.stage_epochs == (1,)
    assert cfg.train.weights.monotonicity == 1 and cfg.device == "cpu"
