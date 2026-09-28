# [AAAI 2025] Neural Conformal Control for Time Series Forecasting
This is the official implementation of "[Neural Conformal Control for Time Series Forecasting](https://arxiv.org/pdf/2412.18144)" (NCC) appearing in AAAI 2025 (main track). Authors are Ruipu Li and Alexander Rodríguez from the University of Michigan.

NCC takes the point forecasts of any base forecaster and learns prediction intervals for a set of target error rates (alphas) that keep their coverage over time.

## Repository layout
```
configs/
  default.yaml                    paper defaults, merged under every experiment
  experiments/<exp_id>.yaml       the 10 paper experiments, as overrides of default.yaml
src/ncc/                          the method (numpy and torch only, no file I/O)
  config.py                       NCCConfig: all NCC parameters and their defaults
  layers.py                       encoders and attention blocks used by the model
  model.py                        the NCC model
  losses.py                       quantile, coverage, efficiency and monotonicity losses
  series.py                       per-location normalization, and the store of everything handed to NCC
  history.py                      the calibration history the model reads through a window
  train.py                        training loop, loss log and early stopping
  online.py                       NCC.step: the step-based method, with retraining, ACI and test-time adaptation
  experiments/                    paper reproduction: the only part that reads or writes files
    paths.py, files.py            file locations (relative to the repository root), pickle/YAML/CSV reading and writing
    config.py                     loads configs/experiments/<exp_id>.yaml
    run.py                        base predictions -> NCC, step by step -> results/<exp_id>_<seed>.pkl
    evaluate.py, metrics.py       results -> metrics -> results/csvs/*.csv
    cli.py                        `python -m ncc run` and `python -m ncc evaluate`
tests/                            unit tests, and a slow real-data check
data/<dataset>_clip_idx.pkl       time steps used for evaluation (e.g. the respiratory season for covid)
bin/setup.sh                      installs the package and creates results/ and models/
```

## How to run
The commands below work from any directory: file paths are relative to the repository root.

### Setup
Create an environment with Python >= 3.10 (tested with Python 3.10 and PyTorch 2.4), then run:
```
cd bin && bash setup.sh
```
This installs the repository as the `ncc` package (editable, with `pytest`), and creates `results/base_pred/`, `results/csvs/` and `models/`. Equivalently, run `pip install -e ".[test]"` from the repository root.

Training runs on `cuda:0` by default (`ncc.device` in `configs/default.yaml`). To use another device for one run, pass `--device` (e.g. `--device cpu` or `--device mps`).

Run the unit tests with `pytest tests/`. `pytest -m slow tests/test_real_data.py` runs two paper experiments on real base predictions on Apple-silicon MPS; it needs the base prediction files of Step 1 and takes a long time.

### Step 1: Get the base forecaster predictions
NCC calibrates the predictions of a base forecaster. Download the base prediction files used in the paper from this [link](https://drive.google.com/drive/folders/1RE0Wwfxvz0_mv9Q2gbNZIzGkL23_dHKG?usp=sharing) and put them in `results/base_pred/`.

To use your own base forecaster, save its predictions in the same format (see [Base prediction format](#base-prediction-format)).

### Step 2: Train NCC and calibrate the test steps
Pick an experiment from the table below, then run:
```
python -m ncc run -i <exp_id> [-s <seed>] [--device <cpu|cuda:0|mps>]
```
This feeds the base predictions to NCC one step at a time (see [Using NCC step by step](#using-ncc-step-by-step)). At the first test step, NCC trains on the steps before the validation and test steps whose truth is already known, then produces a forecast at every test step. It saves the test forecasts to `results/<exp_id>_<seed>.pkl` and the final weights to `models/<exp_id>_<seed>.pt`. `-s` is optional and defaults to the `seed` in the config. During the test steps, the model is retrained every `ncc.retrain.period` steps: retraining at test step t trains on the steps `[0, t - val_steps)` whose truth is known at t.

| exp_id | Dataset | Base forecaster | Steps ahead | Base predictions |
|---|---|---|---|---|
| 22055 | Covid-19 hospitalization | seq2seq | 1 | `saved_pred_19.pickle` |
| 29005 | Covid-19 hospitalization | seq2seq | 2 | `saved_pred_19.pickle` |
| 29105 | Covid-19 hospitalization | seq2seq | 3 | `saved_pred_19.pickle` |
| 29205 | Covid-19 hospitalization | seq2seq | 4 | `saved_pred_19.pickle` |
| 22053 | SMD | seq2seq | 1 | `saved_pred_35.pickle` |
| 29003 | SMD | seq2seq | 2 | `saved_pred_35.pickle` |
| 29130 | SMD | seq2seq | 3 | `saved_pred_35.pickle` |
| 29203 | SMD | seq2seq | 4 | `saved_pred_35.pickle` |
| 22052 | Electricity | transformer | 1 | `saved_pred_50.pickle` |
| 22051 | Weather | seq2seq | 1 | `saved_pred_30.pickle` |

The calibrated results from the paper can be downloaded from this [link](https://drive.google.com/drive/folders/1KfYcmDPUVc0vJgN6QZ2eB7Z_DwDNIJE6?usp=sharing). Put them in `results/` to evaluate them directly. Retraining with the current code does not reproduce these files exactly: removing unused modules changed the order in which the model's weights are randomly initialized, and the step-based version changes the numbers again (see [Differences from the paper's code](#differences-from-the-papers-code)).

### Step 3: Evaluate
```
python -m ncc evaluate -i <exp_id> [-s <seed>]   # results/csvs/<exp_id>_<seed>.csv: metrics per region, plus their median and mean
python -m ncc evaluate -i <exp_id> -u            # results/csvs/<exp_id>_multi_seeds.csv: all `seeds` in the config, plus their mean
```
Options:
+ `-c`: only evaluate the time steps in `data/<dataset>_clip_idx.pkl`. Electricity has no such file and is always evaluated on all test steps.
+ `-o`: sort the interval widths at every step so they shrink as alpha grows.

The csv columns are: `rmse`, `nrmse`, `mape` (point forecast errors); `cs_score` (calibration score: mean gap between the target and the observed coverage over all alphas); `norm_crps_val`; `mean_wis` (weighted interval score); `mono` (monotonicity of the intervals in alpha); `cov_50`, `cov_90`, `cov_95` (coverage of the 50/90/95% intervals); `pi_length_90` (half width of the 90% interval); `dvs` (fraction of steps whose intervals are all monotone); `qice`.

## Configuration
An experiment is `configs/experiments/<exp_id>.yaml` merged onto `configs/default.yaml` (nested mappings are merged recursively, lists are replaced). The experiment keys are:

| Key | Meaning |
|---|---|
| `base_pred_id` | base predictions: `results/base_pred/saved_pred_<base_pred_id>.pickle` |
| `horizon` | steps ahead, counted from 1 |
| `test_steps` | the last `test_steps` time steps are the test steps |
| `max_steps` | the run stops after step `max_steps` (default 10000) |
| `seed`, `seeds` | default seed of `run` and `evaluate`; the seeds evaluated by `evaluate -u` |
| `ncc` | the NCC parameters below |

The time steps of a base prediction file are split into train, validation and test, in this order: the last `test_steps` steps are the test steps, and the `ncc.val_steps` steps before them are the validation steps.

The `ncc` parameters and their defaults are defined in `src/ncc/config.py` (`NCCConfig`), the only place that holds defaults. `configs/default.yaml` only sets the paper's values that differ from them. Keys marked *required* have no default. An unknown key raises an error.

| Key | Default | Meaning |
|---|---|---|
| `alphas` | `[0.02, 0.05, 0.1, 0.2, ..., 0.9, 1.0]` | target miscoverage rates, strictly ascending |
| `device` | `cpu` (`cuda:0` in `default.yaml`) | torch device |
| `window_size` | *required* | length of the calibration history the model reads |
| `val_steps` | *required* | training at step t (warm-up or retrain) uses the scored origins `s < t - val_steps`; in the paper experiments, the validation steps before the first test step |
| `init_err_with_alpha` | `false` | start the miss history at alpha instead of 1 |
| `n_warmup` | none | scored origins `NCC.step` waits for before its first training; required by `NCC` (the paper driver derives it so that NCC warms up at the first test step) |
| `model.hidden_dim` | 64 | size of each encoding and of the fused embedding |
| `model.score_encoder_hidden_dim`, `model.score_encoder_num_layers` | 64, 2 | GRU encoder of the (y_hat, y, score) history |
| `model.qhat_encoder_hidden_dim`, `model.qhat_encoder_num_layers` | 32, 4 | GRU encoder of the q-hat history |
| `model.error_encoder_hidden_dim` | 32 | encoder of the miss history |
| `model.alpha_encoder_hidden_dim` | 32 | encoder of the alphas |
| `model.q_fc_hidden_dim` | 128 | hidden size of the quantile head |
| `train.lr` | 0.0001 | Adam learning rate |
| `train.gamma` | 0.01 | learning rate decay factor, applied every 50 epochs |
| `train.epochs` | 500 | epochs of the initial training (ignored with `stages`) |
| `train.K` | *required* | temperature of the sigmoid in the coverage and efficiency losses |
| `train.weights.{quantile, coverage, efficiency, monotonicity}` | 0.01, 1, 0.00108, 1 | loss = sum of weight * loss; with `stages`, only `monotonicity` is used |
| `train.stages` | none | train in stages: a list of `{epochs, weights: [quantile, coverage, efficiency]}` |
| `retrain.period` | 5 | retrain every `period` issued steps (steps that return q-hats; in the paper experiments, the test steps) |
| `retrain.epochs` | 50 | epochs of a retrain without `stages` |
| `retrain.stage_epochs` | `[10, 10, 1]` | epochs of each stage of a retrain with `stages`; needs one entry per stage |
| `tta.enabled` | `false` | test-time adaptation (TTA): fixes q-hats that are not monotone in alpha |
| `tta.lr` | 0.0005 | learning rate of the TTA head |
| `tta.reg_factor` | 1 | weight of the RMS of the TTA output in the TTA loss |
| `tta.aci_lr` | 0.01 | step size of the conformal (ACI) correction, which only runs inside TTA |

Each training run (or stage) has its own Adam optimizer, learning rate schedule and early stopping (patience 50 epochs, on the training loss).

To run NCC on new base predictions, save them as `results/base_pred/saved_pred_<base_pred_id>.pickle` and add a config that sets that `base_pred_id`.

## Using NCC step by step
The core is a stateful object that a sequential calibration pipeline calls once per forecast origin, in ascending order, with only what is visible at that origin. It does no file I/O.
```python
from ncc import NCC, NCCConfig, Feedback

cfg = NCCConfig.from_dict({"window_size": 15, "val_steps": 1, "n_warmup": 30, "train": {"K": 2.26}})
ncc = NCC(cfg, seed=1)                  # one object per target and horizon
for t in range(num_steps):
    q_hats = ncc.step(t, centers[t], feedback[t])
    # q_hats: {location: np.ndarray (len(alphas),)}: the interval for alphas[i] is center ± q_hats[location][i]
```
`step(t, centers, feedback)`:
+ `t`: 0 on the first call, then the previous `t` + 1.
+ `centers`: `{location: base forecast}` (the point forecast, else the median), finite, for every location at every step. The first call fixes the set of locations and their order; every later call must have exactly those keys.
+ `feedback`: the `Feedback(origin, location, truth, score, external_miss=None)` rows that became visible at `t`. `truth` is the first-reported truth; `score` is the task's nonconformity score (one per task, `>= 0`, in target units, e.g. `|truth - center|`).
+ Returns `{location: q_hat}` (float64, target units) for every location, or `{}` while warming up. The q-hats are returned as NCC produced them: not clamped at 0 and possibly not monotone in alpha, so the caller builds, clamps and repairs the intervals.

**Visibility.** The caller decides what is visible; NCC never computes a lag. NCC places an origin in its history when that origin's feedback arrives. For example, in a pipeline where a forecast made at origin t for horizon h (counted from 0) is graded at origin t + h + 1, the feedback of origin s is passed at step s + h + 1. In the paper driver (`horizon` counted from 1), the feedback of step s is passed at step s + `horizon`.

**Input checks.** `step` raises `ValueError` (naming the offending item) and leaves the object unchanged when:
+ `t` is not the previous `t` + 1 (or the first call's `t` is not 0);
+ `centers` is empty on the first call, misses a location, has an extra location, or has a non-finite value;
+ the feedback breaks its rules: it is empty until the first step that brings any; after that, every step's feedback covers exactly one origin, one more than the previous step's (the first is 0), and before `t`; it has exactly one row per location (none missing, none extra, no duplicates); each truth and score is finite, each score `>= 0`; `external_miss` is `None` or has one entry per alpha.

So every location has a center and a truth at every origin, and the history never has gaps. No other data condition raises; errors from torch itself propagate.

**Warm-up.** `step` returns `{}` until `n_warmup` origins have a score and at least one scored origin is before `t - val_steps`. Then NCC fits a scaler per location on the truths of the training origins (a constant location gets std 1), builds and trains the model on the training origins, and replays origins `0 .. t-1` with the trained model to build its history, as if it had run online from the start. It then forecasts at every step and retrains every `retrain.period` forecasts. Training (at warm-up and at each retrain) uses the scored origins `s < t - val_steps`.

**Where misses come from.** The model reads the recent misses (was the truth outside the interval?) of each alpha:
+ in training, and in the warm-up replay: the misses of NCC's own q-hats;
+ for the origins forecast from the warm-up step on: `external_miss[i]` when it is not `None` (e.g. whether the interval that was actually delivered missed), otherwise NCC's own issued q-hat.

**Randomness.** Each object owns its random generators, seeded by `seed`, so on CPU several objects in one process don't disturb each other, and the same inputs give the same outputs. On `cuda` or `mps`, the accelerator's generator is process-global (seeded when an object is created), so runs are seeded but may not repeat exactly. Use the objects from one thread.

### Differences from the paper's code
The paper's code trained and calibrated in two separate processes, with the global random generators. The step-based version gives slightly different numbers in every experiment: the score is given by the caller in float64 and normalized as `score / std` (it used to be `|y_hat - y|` of the normalized float32 values), the random number sequence is no longer reseeded between training and calibration, and the saved arrays are float64. The results differ clearly for 29105 and 29205: their `val_steps` is smaller than `horizon - 1`, and training now only uses the truths that are already known (the paper's code also trained on 1 and 2 steps whose truth wasn't observed yet).

## Base prediction format
`results/base_pred/saved_pred_<base_pred_id>.pickle` is a pickled dict:
```python
{
    'params': {'dataset': 'smd'},  # dataset name, used to find data/<dataset>_clip_idx.pkl
    'base_pred': [                 # one entry per time step, in time order
        (predictions, addition_infos),
        ...
    ],
}
# predictions[region]: np.ndarray with the forecasts for 1, 2, ..., H steps ahead
# addition_infos[region]: tuple whose first element is an np.ndarray with the ground truth for the same H steps
```
The other elements of `addition_infos[region]` are ignored. Single-series datasets use one region (e.g. `'X'`).

NCC saves `results/<exp_id>_<seed>.pkl` as `{region: {steps_ahead: [(q_hats, y_hat, y), ...]}}`, one entry per test step, where the interval for `alphas[i]` is `y_hat ± q_hats[i]`. All three are float64 np.ndarrays (`y_hat` and `y` are the base prediction file's values); `q_hats` has one entry per alpha, `y_hat` and `y` have shape (1,). The paper's result files hold float32 arrays in the same layout, and `evaluate` reads both.

## Contact
If you have any questions, please contact Ruipu Li at liruipu@umich.edu.

## Cite our work
If you find this work helpful, cite our work:

Li, Ruipu, and Alexander Rodríguez. "Neural Conformal Control for Time Series Forecasting". In Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. 2025.
```bibtex
@inproceedings{li2025neural,
  title={Neural Conformal Control for Time Series Forecasting},
  author={Li, Ruipu and Rodr{\'\i}guez, Alexander},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  year={2025}
}
```
