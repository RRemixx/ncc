"""R3: paper configs on real base predictions, on Apple-silicon MPS.

Slow, so excluded by default. Run with `pytest -m slow tests/test_real_data.py`.
Needs results/base_pred/saved_pred_<id>.pickle (see README); a test is skipped when its file is
missing or MPS is not available. Outputs go to a temporary directory, never to the repository.
"""
import csv
import time

import pytest
import torch

from ncc.experiments import paths
from ncc.experiments.config import load_experiment
from ncc.experiments.evaluate import evaluate
from ncc.experiments.run import run_experiment

REPO_ROOT = paths.ROOT

# minimum cov_90 (observed coverage of the 90% interval, mean over locations)
THRESHOLDS = {"22052": 0.80, "22055": 0.60}
# evaluate only the steps in data/<dataset>_clip_idx.pkl (electricity has no clip file)
CLIP = {"22052": False, "22055": True}


@pytest.mark.slow
@pytest.mark.parametrize("exp_id", sorted(THRESHOLDS))
def test_cov_90_on_mps(exp_id, tmp_path, monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    base_pred = paths.base_pred(load_experiment(exp_id).base_pred_id)
    if not base_pred.exists():
        pytest.skip(f"missing base predictions {base_pred}")

    (tmp_path / "results").mkdir()
    for linked in ("results/base_pred", "configs", "data"):
        (tmp_path / linked).symlink_to(REPO_ROOT / linked)
    monkeypatch.setattr(paths, "ROOT", tmp_path)

    start = time.perf_counter()
    exp = load_experiment(exp_id)
    run_experiment(exp, seed=1, device="mps")
    csv_path = evaluate(exp, seed=1, clip=CLIP[exp_id])
    cov_90 = _mean_row(csv_path)["cov_90"]
    print(f"R3 {exp_id}: cov_90 = {cov_90:.4f} (threshold {THRESHOLDS[exp_id]}), "
          f"run time {time.perf_counter() - start:.0f} s")
    assert cov_90 >= THRESHOLDS[exp_id]


def _mean_row(csv_path):
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    mean = next(row for row in rows[1:] if row[0] == "mean")
    return {name: float(value) for name, value in zip(header[2:], mean[2:])}
