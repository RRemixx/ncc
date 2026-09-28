"""Fast unit tests (CPU, synthetic data)."""
import csv
import dataclasses
import math
import pickle

import numpy as np
import pytest
import torch
import yaml

from ncc import NCC, Feedback, NCCConfig
from ncc.experiments import paths
from ncc.experiments.config import load_experiment
from ncc.experiments.evaluate import evaluate
from ncc.experiments.metrics import METRIC_NAMES
from ncc.experiments.run import run_experiment
from ncc.train import LossLog, Patience

# every test builds models with one GRU layer, for which torch warns that the GRU dropout has no effect
pytestmark = pytest.mark.filterwarnings("ignore:dropout option adds dropout")

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
        assert q_hats.dtype == y_hat.dtype == y.dtype == np.float64
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


# --- The step-based core (T2-T6, T8): tiny config, 3 locations, 30 steps, horizon 2 (the feedback of
# origin s arrives at step s + 2), n_warmup 10, val_steps 0, retrain every 3 issued steps, TTA on.

N_WARMUP = 10
FIRST_OUTPUT = N_WARMUP + HORIZON - 1   # at step t, origins 0 .. t - HORIZON are scored
def _cfg(**changes):
    return NCCConfig.from_dict({**TINY_NCC, "n_warmup": N_WARMUP, **changes})


def _series(seed=0):
    """(centers, truths), each (NUM_STEPS x locations), in target units."""
    rng = np.random.RandomState(seed)
    return rng.uniform(30, 140, (NUM_STEPS, len(LOCATIONS))), rng.uniform(30, 140, (NUM_STEPS, len(LOCATIONS)))


def _inputs(data, t, external_miss=None):
    """(centers, feedback) of step t; external_miss(origin) gives the external misses of every location."""
    centers, truths = data
    s = t - HORIZON
    feedback = [] if s < 0 else [
        Feedback(s, r, truths[s, i], abs(truths[s, i] - centers[s, i]), external_miss and external_miss(s))
        for i, r in enumerate(LOCATIONS)]
    return {r: centers[t, i] for i, r in enumerate(LOCATIONS)}, feedback


def _run(ncc, data, num_steps=NUM_STEPS, external_miss=None):
    return [ncc.step(t, *_inputs(data, t, external_miss)) for t in range(num_steps)]


def _same(out_a, out_b):
    return list(out_a) == list(out_b) and all(np.array_equal(out_a[r], out_b[r]) for r in out_a)


def _all_same(outs_a, outs_b):
    return len(outs_a) == len(outs_b) and all(_same(a, b) for a, b in zip(outs_a, outs_b))


# T2: warm-up
def test_warm_up_returns_nothing_until_n_warmup_scored_origins():
    ncc = NCC(_cfg(), seed=1)
    outputs = _run(ncc, _series())
    assert all(out == {} for out in outputs[:FIRST_OUTPUT])
    assert ncc.warmup_step == FIRST_OUTPUT
    for out in outputs[FIRST_OUTPUT:]:
        assert list(out) == LOCATIONS
        for q_hat in out.values():
            assert q_hat.shape == (len(ncc.cfg.alphas),) and q_hat.dtype == np.float64
            assert np.all(np.isfinite(q_hat))


def test_warm_up_waits_for_a_training_origin():
    # 2 scored origins are visible at step 3, but no origin is before t - val_steps until step 6
    ncc = NCC(_cfg(n_warmup=2, val_steps=5), seed=1)
    outputs = _run(ncc, _series(), num_steps=8)
    assert all(out == {} for out in outputs[:6]) and ncc.warmup_step == 6
    assert list(outputs[6]) == LOCATIONS


# T3: prefix stability
@pytest.mark.parametrize("last", [FIRST_OUTPUT - 1, FIRST_OUTPUT + 1, 20])  # step last + 1: warm-up, a retrain, later
def test_prefix_stability(last):
    data = _series()
    short = _run(NCC(_cfg(), seed=1), data, num_steps=last + 1)
    centers, truths = (a.copy() for a in data)
    centers[last + 1] = 1e6                  # the inputs of step last + 1: its centers and the truths of its feedback
    truths[last + 1 - HORIZON] = -1e6
    extended = _run(NCC(_cfg(), seed=1), (centers, truths), num_steps=last + 2)
    assert _all_same(short, extended[:last + 1])


# T4: determinism and isolation
def test_determinism_and_isolation():
    data_a, data_b = _series(0), _series(1)
    alone_a = _run(NCC(_cfg(), seed=1), data_a)
    assert _all_same(alone_a, _run(NCC(_cfg(), seed=1), data_a))
    alone_b = _run(NCC(_cfg(), seed=2), data_b)

    ncc_a, ncc_b = NCC(_cfg(), seed=1), NCC(_cfg(), seed=2)
    outs_a, outs_b = [], []
    for t in range(NUM_STEPS):
        outs_a.append(ncc_a.step(t, *_inputs(data_a, t)))
        np.random.rand(3)                    # the global generators must not matter
        torch.rand(3)
        outs_b.append(ncc_b.step(t, *_inputs(data_b, t)))
    assert _all_same(outs_a, alone_a) and _all_same(outs_b, alone_b)
    assert not _all_same(alone_a, _run(NCC(_cfg(), seed=2), data_a))   # the seed matters


# T5: input checks
CHECK_STEP = FIRST_OUTPUT + 2    # after warm-up; its feedback is for origin CHECK_STEP - HORIZON


def _replace_first(feedback, **changes):
    return [dataclasses.replace(feedback[0], **changes)] + feedback[1:]


BAD_INPUTS = {
    "skipped t": (lambda t, c, f: (t + 1, c, f), "expected t="),
    "float t": (lambda t, c, f: (float(t), c, f), "expected t="),
    "repeated t": (lambda t, c, f: (t - 1, c, f), "expected t="),
    "centers missing a location": (lambda t, c, f: (t, {r: v for r, v in c.items() if r != "A"}, f), r"missing \['A'\]"),
    "centers with an extra location": (lambda t, c, f: (t, {**c, "D": 50.0}, f), r"extra \['D'\]"),
    "non-finite center": (lambda t, c, f: (t, {**c, "B": math.nan}, f), "center of location 'B'"),
    "feedback missing a location": (lambda t, c, f: (t, c, f[1:]), r"missing location\(s\) \['A'\]"),
    "feedback for an extra location": (lambda t, c, f: (t, c, f + [dataclasses.replace(f[0], location="D")]),
                                       "unknown location 'D'"),
    "duplicate feedback row": (lambda t, c, f: (t, c, f + [f[0]]), "duplicate feedback for location 'A'"),
    "feedback skipping an origin": (lambda t, c, f: (t, c, [dataclasses.replace(fb, origin=fb.origin + 1) for fb in f]),
                                    "expected exactly origin"),
    "two origins in one step": (lambda t, c, f: (t, c, f + [dataclasses.replace(fb, origin=fb.origin + 1) for fb in f]),
                                "expected exactly origin"),
    "no feedback after it started": (lambda t, c, f: (t, c, []), "no feedback"),
    "negative score": (lambda t, c, f: (t, c, _replace_first(f, score=-1.0)), "score of location 'A'"),
    "non-finite truth": (lambda t, c, f: (t, c, _replace_first(f, truth=math.inf)), "truth of location 'A'"),
    "external_miss of the wrong length": (lambda t, c, f: (t, c, _replace_first(f, external_miss=(True,))),
                                          "external_miss of location 'A'"),
    "external_miss not a sequence": (lambda t, c, f: (t, c, _replace_first(f, external_miss=True)),
                                     "external_miss of location 'A' must be None or a sequence"),
    "feedback row not a Feedback": (lambda t, c, f: (t, c, [dataclasses.astuple(f[0])] + f[1:]), "not a Feedback"),
}


@pytest.fixture(scope="module")
def reference_run():
    return _run(NCC(_cfg(), seed=1), _series())


@pytest.mark.parametrize("case", sorted(BAD_INPUTS))
def test_bad_input_raises_and_leaves_the_object_usable(case, reference_run):
    make_bad, message = BAD_INPUTS[case]
    data = _series()
    ncc = NCC(_cfg(), seed=1)
    outputs = _run(ncc, data, num_steps=CHECK_STEP)
    with pytest.raises(ValueError, match=message):
        ncc.step(*make_bad(CHECK_STEP, *_inputs(data, CHECK_STEP)))
    outputs += [ncc.step(t, *_inputs(data, t)) for t in range(CHECK_STEP, NUM_STEPS)]
    assert _all_same(outputs, reference_run)


@pytest.mark.parametrize("first_call, message", [
    (lambda c: (0, {}, []), "centers is empty"),
    (lambda c: (1, c, []), "expected t=0"),
    (lambda c: (0, c, [Feedback(0, r, 1.0, 0.0) for r in c]), "not before step 0"),
])
def test_bad_first_call_raises_and_leaves_the_object_usable(first_call, message, reference_run):
    data = _series()
    ncc = NCC(_cfg(), seed=1)
    with pytest.raises(ValueError, match=message):
        ncc.step(*first_call(_inputs(data, 0)[0]))
    assert _all_same(_run(ncc, data), reference_run)


@pytest.mark.parametrize("n_warmup", [None, 0])
def test_ncc_requires_n_warmup(n_warmup):
    with pytest.raises(ValueError, match="n_warmup"):
        NCC(_cfg(n_warmup=n_warmup), seed=1)


# T6: constant location
def test_constant_location_gives_finite_q_hats():
    centers, truths = _series()
    truths[:, 0] = 0.0                               # e.g. all-zero counts
    centers[:, 0] = np.random.RandomState(2).uniform(0, 2, NUM_STEPS)
    ncc = NCC(_cfg(), seed=1)
    outputs = _run(ncc, (centers, truths))
    assert ncc.scaler.stds[LOCATIONS[0]] == 1
    for out in outputs[FIRST_OUTPUT:]:
        assert all(np.all(np.isfinite(q_hat)) for q_hat in out.values())


# T8: where misses come from
ALL_MISSED = (True,) * len(_cfg().alphas)    # an external miss for every alpha


def test_external_miss_before_warm_up_is_ignored(reference_run):
    ncc = NCC(_cfg(), seed=1)
    # origins before warmup_step, including those whose feedback only arrives after warm-up
    outputs = _run(ncc, _series(), external_miss=lambda s: ALL_MISSED if s < FIRST_OUTPUT else None)
    assert ncc.warmup_step == FIRST_OUTPUT
    assert _all_same(outputs, reference_run)


def test_external_miss_of_an_online_origin_counts_from_its_arrival(reference_run):
    origin = _covered_online_origin(reference_run)
    outputs = _run(NCC(_cfg(), seed=1), _series(), external_miss=lambda s: ALL_MISSED if s == origin else None)
    arrival = origin + HORIZON
    assert _all_same(outputs[:arrival], reference_run[:arrival])
    assert all(not _same(out, ref) for out, ref in zip(outputs[arrival:], reference_run[arrival:]))


def test_training_ignores_external_miss():
    # TTA off: TTA fits its head on the online window, so it would carry the external misses into the weights
    cfg = _cfg(tta={"enabled": False})
    own, external = NCC(cfg, seed=1), NCC(cfg, seed=1)
    own_outputs = _run(own, _series())
    external_outputs = _run(external, _series(), external_miss=lambda s: ALL_MISSED if s >= FIRST_OUTPUT else None)
    assert not _all_same(own_outputs, external_outputs)      # the external misses did change the online history
    assert len(own.fit_logs) == len(external.fit_logs) > 1
    for own_log, external_log in zip(own.fit_logs, external.fit_logs):
        assert own_log.epochs.keys() == external_log.epochs.keys()
        for name in own_log.epochs:
            np.testing.assert_array_equal(own_log.epochs[name], external_log.epochs[name])


def _covered_online_origin(outputs):
    """An online origin whose issued q-hat covered its truth for at least one alpha and location."""
    centers, truths = _series()
    last = NUM_STEPS - HORIZON - 2    # leaves a few steps after the feedback arrives
    for s in range(FIRST_OUTPUT, last + 1):
        scores = np.abs(truths[s] - centers[s])
        if any(np.any(outputs[s][r] > scores[i] * (1 + 1e-6)) for i, r in enumerate(LOCATIONS)):
            return s
    raise AssertionError("no online origin where NCC's own q-hat covered")
