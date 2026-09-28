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
  series.py                       per-location normalization of the base predictions
  history.py                      the calibration history the model reads
  train.py                        training loop, loss log and early stopping
  online.py                       the forecasting loop, with retraining, ACI and test-time adaptation
  experiments/                    paper reproduction: the only part that reads or writes files
    paths.py, files.py            file locations (relative to the repository root), pickle/YAML/CSV reading and writing
    config.py                     loads configs/experiments/<exp_id>.yaml
    run.py                        base predictions -> NCC -> results/<exp_id>_<seed>.pkl
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
This trains NCC on the steps before the validation and test steps, then runs it over every step and saves the test forecasts to `results/<exp_id>_<seed>.pkl` and the final weights to `models/<exp_id>_<seed>.pt`. `-s` is optional and defaults to the `seed` in the config. During the test steps, the model is retrained every `ncc.retrain.period` steps: retraining at test step t trains on steps `[0, t - val_steps)`.

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

The calibrated results from the paper can be downloaded from this [link](https://drive.google.com/drive/folders/1KfYcmDPUVc0vJgN6QZ2eB7Z_DwDNIJE6?usp=sharing). Put them in `results/` to evaluate them directly. Retraining with the current code does not reproduce these files exactly: removing unused modules changed the order in which the model's weights are randomly initialized.

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
| `val_steps` | *required* | steps between the end of the training steps and the first test step (or the retrain step) |
| `init_err_with_alpha` | `false` | start the miss history at alpha instead of 1 |
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
| `retrain.period` | 5 | retrain every `period` test steps |
| `retrain.epochs` | 50 | epochs of a retrain without `stages` |
| `retrain.stage_epochs` | `[10, 10, 1]` | epochs of each stage of a retrain with `stages`; needs one entry per stage |
| `tta.enabled` | `false` | test-time adaptation (TTA): fixes q-hats that are not monotone in alpha |
| `tta.lr` | 0.0005 | learning rate of the TTA head |
| `tta.reg_factor` | 1 | weight of the RMS of the TTA output in the TTA loss |
| `tta.aci_lr` | 0.01 | step size of the conformal (ACI) correction, which only runs inside TTA |

Each training run (or stage) has its own Adam optimizer, learning rate schedule and early stopping (patience 50 epochs, on the training loss).

To run NCC on new base predictions, save them as `results/base_pred/saved_pred_<base_pred_id>.pickle` and add a config that sets that `base_pred_id`.

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

NCC saves `results/<exp_id>_<seed>.pkl` as `{region: {steps_ahead: [(q_hats, y_hat, y), ...]}}`, one entry per test step, where the interval for `alphas[i]` is `y_hat ± q_hats[i]`. All three are float32 np.ndarrays; `q_hats` has one entry per alpha, `y_hat` and `y` have shape (1,).

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
