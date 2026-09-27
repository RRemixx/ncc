# [AAAI 2025] Neural Conformal Control for Time Series Forecasting
This is the official implementation of "[Neural Conformal Control for Time Series Forecasting](https://arxiv.org/pdf/2412.18144)" (NCC) appearing in AAAI 2025 (main track). Authors are Ruipu Li and Alexander Rodríguez from the University of Michigan.

NCC takes the point forecasts of any base forecaster and learns prediction intervals for a set of target error rates (alphas) that keep their coverage over time.

## Repository layout
```
setup/
  e2ecp.yaml                 default NCC parameters
  cp_exp_params/<exp_id>.yaml   per-experiment overrides of e2ecp.yaml
src/forecaster/
  e2ecp.py                   entry point: train NCC / calibrate the test steps
  e2ecp_utils.py             data preparation, training loop, test-time retraining and adaptation
  end2endmodel.py            the NCC model
  basicmodels.py             encoders and attention blocks used by the model
  loss_utils.py              quantile, coverage, efficiency and monotonicity losses
  test_time_adapt.py         monotonicity check used by test-time adaptation
  eval_cp.py                 metrics -> csv
  metrics.py                 metric implementations
data/<dataset>_clip_idx.pkl  time steps used for evaluation (e.g. the respiratory season for covid)
bin/setup.sh                 installs dependencies and creates results/ and models/
```

## How to run
All commands below are run from `src/forecaster/`, because paths in the code and configs are relative to it.

### Setup
Create an environment with Python >= 3.8 (tested with Python 3.10 and PyTorch 2.4), then run:
```
cd bin && bash setup.sh
```
This installs the dependencies, installs `src/` as the `forecaster` package, and creates `results/base_pred/`, `results/csvs/` and `models/`.

Training runs on `cuda:0` by default. To use another device, change `training_params.device` in `setup/e2ecp.yaml`.

### Step 1: Get the base forecaster predictions
NCC calibrates the predictions of a base forecaster. Download the base prediction files used in the paper from this [link](https://drive.google.com/drive/folders/1RE0Wwfxvz0_mv9Q2gbNZIzGkL23_dHKG?usp=sharing) and put them in `results/base_pred/`.

To use your own base forecaster, save its predictions in the same format (see [Base prediction format](#base-prediction-format)).

### Step 2: Train NCC and calibrate the test steps
Pick an experiment from the table below, then run:
```
python e2ecp.py -i=<exp_id> -s=<seed> -t    # train, saves models/<exp_id>_<seed>.pt
python e2ecp.py -i=<exp_id> -s=<seed>       # calibrate the test steps, saves results/<exp_id>_<seed>.pkl
```
`-s` is optional and defaults to the `seed` in the config. During the test steps, the model is retrained every `retrain_period` steps on everything observed so far.

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
python eval_cp.py -i=<exp_id> -s=<seed>    # results/csvs/<exp_id>_<seed>.csv: metrics per region, plus their median and mean
python eval_cp.py -i=<exp_id> -u           # results/csvs/<exp_id>_multi_seeds.csv: all `seeds` in the config, plus their mean
```
Options:
+ `-c`: only evaluate the time steps in `data/<dataset>_clip_idx.pkl`. Electricity has no such file and is always evaluated on all test steps.
+ `-o`: sort the interval widths at every step so they shrink as alpha grows.

The csv columns are: `rmse`, `nrmse`, `mape` (point forecast errors); `cs_score` (calibration score: mean gap between the target and the observed coverage over all alphas); `norm_crps_val`; `mean_wis` (weighted interval score); `mono` (monotonicity of the intervals in alpha); `cov_50`, `cov_90`, `cov_95` (coverage of the 50/90/95% intervals); `pi_length_90` (half width of the 90% interval); `dvs` (fraction of steps whose intervals are all monotone); `qice`.

## Configuration
`setup/e2ecp.yaml` holds the default parameters, with comments. Each `setup/cp_exp_params/<exp_id>.yaml` overrides a subset of them and must set `data_file_id`, `val_weeks`, `test_weeks`, `cov_window_size` and `K`. The time steps of a base prediction file are split into train, validation and test, in this order: the last `test_weeks` steps are the test steps, and the `val_weeks` steps before them are the validation steps.

To run NCC on new base predictions, save them as `results/base_pred/saved_pred_<data_file_id>.pickle` and add a config that sets that `data_file_id`.

## Base prediction format
`results/base_pred/saved_pred_<data_file_id>.pickle` is a pickled dict:
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

NCC saves `results/<exp_id>_<seed>.pkl` as `{region: {steps_ahead: [(q_hats, y_hat, y), ...]}}`, one entry per test step, where the interval for `alphas[i]` is `y_hat ± q_hats[i]`.

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
