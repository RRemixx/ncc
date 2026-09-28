# NCC refactor: step-based core for the sequential calibration pass

Status: approved design, not implemented. Branch: `ncc-refactor`, created from `ncc-cleanup`
(commit `4daa066`). Python environment: the conda env `epi4cast` (Python 3.10, PyTorch 2.4, numpy
1.24). Run every command in it (`conda run -n epi4cast ...` or `conda activate epi4cast`).

## 1. Problem

This repository is the official implementation of "Neural Conformal Control for Time Series
Forecasting" (NCC, AAAI 2025). NCC takes a base forecaster's point forecasts and learns, per target
miscoverage rate alpha, a half interval width q̂ from the recent calibration history.

The code must serve two uses:

1. **Paper reproduction.** Run NCC over the saved base predictions of the 10 paper experiments and
   evaluate the results.
2. **Integration into an external sequential calibration pipeline** (a different repository, not
   touched here). That pipeline builds one method object per run and calls it once per forecast
   origin, in ascending order, with only what is visible at that origin. The method returns the q̂
   it issues, and the pipeline turns them into quantiles. Its requirements:
   - no file I/O inside the method;
   - **prefix stability:** the output at step t depends only on data visible at steps ≤ t, so
     appending a step never changes earlier outputs;
   - the model and training are controlled by a seed;
   - several method objects (one per target × horizon) live in one process, used from one thread.

Today NCC runs as two separate batch processes (train, then calibrate). It reads the whole series
from a pickle, counts its train/val/test split from the end of the series, and uses the global
numpy and torch random generators. None of that fits the pipeline.

**Required (this plan):**
- **Commit 1:** restructure the repository into a package `ncc`, with a core free of file I/O and a
  paper-reproduction layer. The algorithm stays the same, and the results must match `4daa066`
  exactly.
- **Commit 2:** make the core a stateful object with a `step()` method that follows the pipeline's
  visibility rules (section 3.4).
- **After each commit:** a real-data check on Apple-silicon MPS (R3, section 8). `cov_90` (the
  observed coverage of the 90% interval) must reach the R3 thresholds: `≥ 0.80` for electricity and
  `≥ 0.60` for covid. The nominal 0.90 is deliberately not the bar, because it isn't always reachable.

**Out of scope:**
- The adapter class in the external pipeline that will call `NCC.step`. This plan only fixes the
  interface it will use (section 3.4).
- An encoder for auxiliary time series. It is planned for later, as one more `RNNSeqEncoder` whose
  output is a fifth input to the existing self-attention. Nothing in this plan should prepare for it.
- Fixing the latent bugs listed under "kept on purpose" in section 6.
- Rewriting the metrics.

## 2. Existing system (`ncc-cleanup`, commit `4daa066`)

```
bin/setup.sh                        pip install -r requirements.txt; pip install -e src/; mkdir results/base_pred results/csvs models
requirements.txt                    numpy, torch, PyYAML, scikit-learn, properscoring
src/setup.py                        package "forecaster"
setup/e2ecp.yaml                    default parameters (commented)
setup/cp_exp_params/<id>.yaml       10 experiment overrides: 22051 22052 22053 22055 29003 29005 29105 29130 29203 29205
data/{covid,smd,weather}_clip_idx.pkl   time steps used by `eval_cp.py -c`
src/forecaster/
  e2ecp.py          entry point. -i <id> [-s seed] [-t]. -t trains and saves models/<id>_<seed>.pt and
                    models/losses_<id>_<seed>.pkl; without -t it reloads the model and saves results/<id>_<seed>.pkl.
                    load_all_params(id) merges setup/e2ecp.yaml with setup/cp_exp_params/<id>.yaml.
  e2ecp_utils.py    End2endCPDataset, E2ecpScalar, init_model, Batch, RollingSequences,
                    End2endCPTrainerForecaster (forward_pass, tta_forward_pass, train, stage_train, retrain, forecast4eval)
  end2endmodel.py   QuantilePredictor, NCCModel (get_embedding, conformalize, forward, tta_forward)
  basicmodels.py    FFEncoder, ErrEncoder, RNNSeqEncoder, SelfAttention, MultiHeadAttention
  loss_utils.py     LossTracker, QuantileLoss, CoverageLoss, EfficiencyLoss, MonotonicityLoss, LossCalculator
  test_time_adapt.py  monotonicity_measure_func1
  utils.py          pickle_save, pickle_load, load_yaml_params, WriteRows2CSV, EarlyStopping
  eval_cp.py        -i <id> [-s seed | -u] [-c] [-o] → results/csvs/<id>_<seed>.csv or <id>_multi_seeds.csv
  metrics.py        eval_metrics(y_preds, y_trues, lowers, uppers) → 13-tuple
```
All paths are relative (`../../results`, `../../setup`), so every command runs from `src/forecaster/`.

### Base prediction file
`results/base_pred/saved_pred_<data_file_id>.pickle` is not in git. It comes from the Google Drive
folder linked in `README.md`,
https://drive.google.com/drive/folders/1RE0Wwfxvz0_mv9Q2gbNZIzGkL23_dHKG. The format:
```python
{'params': {'dataset': 'smd', ...},
 'base_pred': [(predictions, addition_infos), ...]}     # one entry per time step, in order
# predictions[region]: np.ndarray (H,) forecasts for 1..H steps ahead
# addition_infos[region]: tuple; element 0 is np.ndarray (H,) with the ground truth for those H steps; the rest is ignored
```
### Results file (the layout must stay the same, because the paper's published results use it)
`results/<exp_id>_<seed>.pkl` = `{region: {steps_ahead: [(q_hats, y_hat, y), ...]}}`: one entry per
test step, in original units.
- `q_hats` is an np.ndarray (num_alphas,).
- `y_hat` and `y` are np.ndarray of shape (1,).
- The interval for `alphas[i]` is `y_hat ± q_hats[i]`.
- **In `4daa066` all three are float32.** `y_hat` and `y` are round trips through the normalized
  float32 tensors, not the raw inputs (e.g. 123.456789 is saved as 123.456795). That's because
  `E2ecpScalar.inv_transform` (`e2ecp_utils.py:68-72`) multiplies a float32 array by an `np.float64`
  scalar, which gives float32 under numpy 1.24.

### Current algorithm (one experiment, horizon h = `week_ahead`)
- **Data.** `End2endCPDataset.e2ecp_seq_data` builds `seq_data[region][h] = [(y_hat_t, y_t)]`.
  - `N` = number of time steps.
  - `train_steps = N − val_weeks − test_weeks`. The split is counted from the end.
  - `E2ecpScalar` is fitted per region on `y` of steps `[0, train_steps)`. It maps
    `v → (v − mean)/std`, and maps scores `s → s/std`.
- **Model inputs at step t** (`RollingSequences.batch`, `e2ecp_utils.py:114`), with W =
  `cov_window_size` and A = number of alphas:
  - `err_seq` (B×W×A): past misses;
  - `score_seq` (B×W×3): past (ŷ, y, |y−ŷ|), normalized;
  - `qhat_seq` (B×W×A): always zeros (latent bug, section 6);
  - `input_alphas` (B×A);
  - `q_adj` (B×A): the ACI state.

  The batch is all regions, in a new random order every step (`np.random.shuffle`).
- **Slot history.** Arrays of length `W + N + h − 1`, initialized with misses = 1 (or `alpha` if
  `init_err_seq_w_alpha`), scores = 0 and (ŷ, y) = 0.
  - At step t the model reads slots `[t, t+W)`.
  - `RollingSequences.update` then writes step t's (ŷ, y, score, miss) into slot `t + W + h − 1`,
    where miss = `q̂_t < score_t` (`e2ecp_utils.py:138`).
  - So the window at t holds origins `t−W−h+1 … t−h`: the W latest ones whose truth is known at t.
    Placeholders fill the front while fewer exist.
- **Training.** `train` creates a fresh Adam optimizer, `StepLR(step_size=50, gamma)` and
  `EarlyStopping(patience=50)` (`e2ecp_utils.py:199-203`). Per epoch:
  1. Walk steps `[0, train_steps)` with gradient (the loss is summed from four terms).
  2. Walk `[train_steps, train_steps + val_weeks)` without gradient. These outputs are unused.
  3. Run early stopping on the summed training loss.

  `stage_train` calls `train` once per stage, with that stage's loss weights (`e2ecp_utils.py:241-254`).
  Each stage therefore gets its own optimizer, scheduler and early-stopping counter. `zip` silently
  truncates when the stage lists differ in length.
- **Calibration** (`forecast4eval`) runs in a separate process:
  1. Reseed, build the model, and load the weights.
  2. Walk every step `t < min(N, total_steps_limit + 1)`: forward pass (with TTA if `use_tta`), then
     history write.
  3. From `first_test = N − test_weeks` on, if `(t − first_test + 1) % retrain_period == 0`, retrain
     on `[0, t − val_weeks)` with the retrain epochs.
  4. Save step t's forecast. That q̂ was computed before the retrain, so a retrain only affects later
     steps.

  The calibration history is never rebuilt after a retrain.
- **TTA** (`tta_forward_pass`):
  - Every weight except `tta_model` is frozen.
  - If more than 5% of the batch rows have q̂ increasing in alpha, it runs up to 200 Adam steps on
    the monotonicity loss plus `tta_reg_factor ·` the RMS of the TTA output (`hs.pow(2).mean().sqrt()`,
    `e2ecp_utils.py:186`), checking every 10 steps whether to
    stop.
  - ACI (`conformalize`) runs only here, and only when `t > 2W`.

## 3. Technical plan

### 3.1 Package layout after both commits
```
pyproject.toml                          package "ncc" (src layout); replaces src/setup.py and requirements.txt
bin/setup.sh                            run from bin/ (as today): editable install with the test extra, then creates results/base_pred, results/csvs, models
configs/default.yaml                    paper defaults: experiment keys + the paper's `ncc:` overrides
configs/experiments/<exp_id>.yaml       the 10 experiments, as overrides
data/<dataset>_clip_idx.pkl             unchanged
docs/ncc-refactor-plan.md               this file
src/ncc/                                the method: numpy and torch only; no file I/O, no YAML, no pickle
  __init__.py       exports NCC, NCCConfig, Feedback (NCC and Feedback from commit 2)
  config.py         NCCConfig and its section dataclasses; NCCConfig.from_dict(d)
  layers.py         FFEncoder, ErrEncoder, RNNSeqEncoder, SelfAttention, MultiHeadAttention
  model.py          QuantileHead, NCCModel
  losses.py         QuantileLoss, CoverageLoss, EfficiencyLoss, MonotonicityLoss, NCCLoss, monotonicity_fraction
  series.py         Scaler, and (commit 1) build_series, (commit 2) SeriesStore
  history.py        CalibrationHistory, Batch
  train.py          fit(), LossLog, Patience
  online.py         commit 1: calibrate(); commit 2: Feedback, NCC
  __main__.py       CLI: delegates to ncc.experiments.cli
src/ncc/experiments/                    paper reproduction; the only place with file I/O
  __init__.py
  paths.py          paths relative to the repository root
  config.py         Experiment dataclass; load_experiment(exp_id) from YAML
  run.py            run_experiment(): base_pred pickle → NCC → results pkl + weights
  evaluate.py       results pkl → metrics → CSV
  metrics.py        metric functions (bodies unchanged); METRIC_NAMES; eval_metrics → dict
  files.py          pickle/YAML/CSV helpers; loads pickles with torch tensors mapped to CPU (the paper's
                    base_pred files hold unused CUDA tensors, which a plain pickle.load can't read without CUDA)
  cli.py            argparse for `run` and `evaluate`
tests/test_ncc.py                       fast unit tests (CPU, synthetic data)
tests/test_real_data.py                 R3 (MPS, real data), marked slow
```
Dependency direction:
- `ncc.experiments` imports `ncc`.
- Nothing under `src/ncc/` outside `experiments/` imports `ncc.experiments`, except
  `ncc/__main__.py`.
- Inside the core: `online → train, history, series, model, losses`;
  `train → history, series, losses, model`; `history → series`; `model → layers`.

### 3.2 Config schema

**Where the defaults live:** in the `NCCConfig` dataclasses, which are the single source of
defaults. `configs/default.yaml` contains only the experiment keys and the paper's values that
differ from the `NCCConfig` defaults. So the defaults can't drift between two places. Every field
in `config.py` carries a one-line comment, and the README's configuration section lists them.

`NCCConfig` fields and defaults (`src/ncc/config.py`, frozen dataclasses):
```python
NCCConfig(
  alphas = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),   # strictly ascending; a tuple (a list default is rejected by frozen dataclasses); from_dict converts lists to tuples
  device = "cpu",
  window_size = REQUIRED,                 # was cov_window_size
  val_steps = REQUIRED,                   # was val_weeks
  init_err_with_alpha = False,            # was init_err_seq_w_alpha
  n_warmup = None,                        # commit 2; required by NCC(), set by the paper driver
  model = ModelConfig(hidden_dim=64, score_encoder_hidden_dim=64, score_encoder_num_layers=2,
                      qhat_encoder_hidden_dim=32, qhat_encoder_num_layers=4, error_encoder_hidden_dim=32,
                      alpha_encoder_hidden_dim=32, q_fc_hidden_dim=128),
  train = TrainConfig(lr=0.0001, gamma=0.01, epochs=500, K=REQUIRED,
                      weights=LossWeights(quantile=0.01, coverage=1, efficiency=0.00108, monotonicity=1),
                      stages=None),       # or tuple of Stage(epochs, weights=(quantile, coverage, efficiency))
  retrain = RetrainConfig(period=5, epochs=50, stage_epochs=(10, 10, 1)),
  tta = TTAConfig(enabled=False, lr=0.0005, reg_factor=1, aci_lr=0.01),
)
```
`NCCConfig.from_dict(d)` raises `ValueError` when:
- a key is unknown;
- a REQUIRED key is missing, or `null` (YAML `null` counts as missing);
- `alphas` aren't strictly ascending;
- `train.stages` is set and `len(retrain.stage_epochs) != len(train.stages)`. Today `zip` silently
  truncates in this case.

The external pipeline builds `NCCConfig` from its `params["ncc"]` dict, with `alphas` from its own
config.

`configs/default.yaml`:
```yaml
seed: 1
seeds: [1, 2, 3, 4]          # evaluated by `evaluate -u`
max_steps: 10000             # was training_params.total_steps_limit; the driver stops after step max_steps
# base_pred_id, horizon, test_steps: set per experiment
ncc:
  device: cuda:0             # the paper runs used cuda:0; NCCConfig's own default is cpu
```
An experiment file sets `base_pred_id`, `horizon` and `test_steps`, plus `ncc:` overrides. Nested
mappings are merged recursively onto `default.yaml`, and lists are replaced.

Example `configs/experiments/29130.yaml`:
```yaml
# NCC on smd, seq2seq base forecaster, 3 steps ahead (base predictions: saved_pred_35.pickle)
base_pred_id: 35
horizon: 3
test_steps: 150
ncc:
  val_steps: 100
  window_size: 25
  init_err_with_alpha: true
  model: {score_encoder_hidden_dim: 32, score_encoder_num_layers: 8, qhat_encoder_hidden_dim: 128,
          qhat_encoder_num_layers: 8, error_encoder_hidden_dim: 64, alpha_encoder_hidden_dim: 64}
  train:
    K: 1
    stages:
      - {epochs: 100, weights: [1, 0, 0]}
      - {epochs: 50, weights: [0, 1, 0.05]}
      - {epochs: 10, weights: [0.005399, 1, 0.1251]}
  retrain: {period: 2, stage_epochs: [5, 5, 9]}
  tta: {enabled: true}
```
**Old key → new key** (for converting the 10 files):

| Old (`setup/…`) | New |
|---|---|
| `data_file_id` | `base_pred_id` |
| `week_ahead_list: [h]` | `horizon: h` (every paper config has exactly one) |
| `training_params.test_weeks` | `test_steps` |
| `training_params.total_steps_limit` | `max_steps` |
| `training_params.val_weeks` | `ncc.val_steps` |
| `training_params.cov_window_size` | `ncc.window_size` |
| `training_params.init_err_seq_w_alpha` | `ncc.init_err_with_alpha` |
| `training_params.device` | `ncc.device` |
| `training_params.{lr,gamma,epochs,K}` | `ncc.train.{lr,gamma,epochs,K}` |
| `training_params.{qloss,closs,eloss,mloss}_factor` | `ncc.train.weights.{quantile,coverage,efficiency,monotonicity}` |
| `three_stage_training: true` + `epochs_in_stage[i]` + `loss_factors[i]` | `ncc.train.stages[i] = {epochs, weights}`; `false` → no `stages` |
| `training_params.retrain_period` | `ncc.retrain.period` |
| `training_params.retrain_epoch` | `ncc.retrain.epochs` |
| `training_params.retrain_epochs` | `ncc.retrain.stage_epochs` |
| `training_params.{use_tta,tta_lr,tta_reg_factor,cf_lr}` | `ncc.tta.{enabled,lr,reg_factor,aci_lr}` |
| `model_params.*` | `ncc.model.*` (same names) |
| `alphas` | `ncc.alphas` (the paper value equals the NCCConfig default, so it isn't repeated) |
| `seed`, `seeds` | `seed`, `seeds` |
| `model_path` | removed (paths come from `experiments/paths.py`) |

**Drop keys whose values are never used.**
- With `stages` set, `train.epochs`, `weights.quantile/coverage/efficiency` and `retrain.epochs` are
  ignored, because the stage values override them (`stage_train`, `e2ecp_utils.py:245-252`).
  `weights.monotonicity` is always used.
- Without `stages`, `retrain.stage_epochs` is ignored.
- An override equal to the `NCCConfig` default is also dropped.

Keep each file's header comment.

In commit 2, the paper configs don't set `n_warmup`: the driver derives it (section 3.5).

### 3.3 Module responsibilities (both commits)
- **`layers.py`**: `FFEncoder`, `ErrEncoder`, `RNNSeqEncoder`, `SelfAttention` and
  `MultiHeadAttention`, moved from `basicmodels.py` with their bodies unchanged.
- **`model.py`**: `QuantileHead` (was `QuantilePredictor`) and `NCCModel`.
  - `NCCModel(cfg: ModelConfig, num_alphas: int, window_size: int)`.
  - Attributes, in this construction order: `score_encoder`, `qhat_encoder`, `error_encoder`,
    `alpha_encoder`, `fusion` (was `attn`), `quantile_head` (was `qmodel`), `tta_head` (was
    `tta_model`). The order matters: changing it changes the random initialization.
  - `embed(err_seq, score_seq, qhat_seq, alphas) -> (B, hidden)` keeps the exact code of
    `get_embedding` (`end2endmodel.py:84-92`), including `torch.stack(..., dim=0).permute(1, 0, 2)`.
    A different memory layout changes the matmul summation order. Training amplifies that into
    visible differences: in the earlier cleanup, a one-bit change grew to 1.42 in the test
    predictions.
  - `forward(...) -> q̂` is `quantile_head(embed(...))`.
  - `conformalize` moves to `online.py` as the plain function
    `aci_correct(q, err_seq, alphas, lr, q_adj) -> (q, q_adj)`.
  - The `.pt` weights files change meaning: they now hold the final weights after the whole run, not
    the weights after initial training. The key names change with the renamed attributes. Nothing
    reads these files.
- **`losses.py`**:
  - The four loss classes from `loss_utils.py`, unchanged.
  - `NCCLoss(weights, K, alphas, device)` replaces `LossCalculator`. `__call__(q_hat, err_seq, alphas,
    score) -> (total, parts)` with `parts = (quantile, coverage, efficiency, monotonicity)`. Each
    part is already multiplied by its weight, and `total` is their sum, as today
    (`loss_utils.py:159-168`).
  - `monotonicity_fraction`, moved from `test_time_adapt.py`.
- **`series.py`**: `Scaler` (was `E2ecpScalar`, for one horizon) with `fit(location, truths)`,
  `transform(location, v)` and `inverse(location, v, is_score=False)`. Commit 1 adds
  `build_series`, and commit 2 adds `SeriesStore`.
- **`history.py`**: `CalibrationHistory` (was `RollingSequences`) and `Batch`.
- **`train.py`**: `fit(...) -> LossLog`, with the bodies of `train` and `stage_train`.
  - **Nesting, as today:** each stage calls a single-run routine that builds its own Adam, its own
    `StepLR(step_size=50, gamma)` and its own `Patience(50)`, then loops over epochs. So does each
    retrain stage. Without stages there is one run. The `monotonicity` weight is the same in every
    stage.
  - **`LossLog`** (replaces `LossTracker`, `loss_utils.py:6-59`), with the same arithmetic:
    - per gradient step, the four parts are detached to numpy;
    - a coverage part above 1e7 becomes NaN;
    - per epoch, each part is `np.nanmean` over its steps, and the epoch total is the sum of the
      four means;
    - the log keeps per-epoch values of each part and the total.

    It no longer stores the duplicate "val" copy.
  - **`Patience(patience)`** replaces `EarlyStopping` (`utils.py:28-73`), with the exact comparison:
    - `score = −total`;
    - the first epoch sets `best = score`;
    - after that, `score < best` (delta 0) increments the counter, and reaching `patience` stops the run;
    - otherwise `best = score` and the counter resets.

    NaN behaves as that comparison implies (`utils.py:52`): a NaN score compares False, so it resets
    the counter and becomes `best`. The next epoch compares `score < NaN`, also False, so it becomes
    `best` too, and counting then resumes normally. Example: totals `[1, nan, 2, 3, 4, 5]` with
    `Patience(3)` stop after the 6th epoch.
  - **No "best" state is reloaded.** Today's reload is a no-op, because `state_dict()` returns tensors
    that share storage with the live parameters (`utils.py:72`, `e2ecp_utils.py:237-238`).
- **`online.py`**: the forecasting loop, ACI and TTA. Freezing uses a context manager
  `frozen_except(model, module)`, which restores `requires_grad=True` on exit, even on exceptions.
  Keep the TTA constants as named constants: threshold 0.05, 200 steps at most, a stop check every
  10 steps.
- **Logging**: replace every `print` with `logging.getLogger(__name__)` calls.
  - Epoch losses and retrain start/end at INFO.
  - "Infinite loss, skipped!" and "Loss is nan" at WARNING.
  - The CLI configures INFO logging to stderr.

### 3.4 Commit 2: the step-based core

**Visibility rules** are set by the pipeline and applied once, by the caller, never by NCC. For a
forecast made at origin t for horizon h (counted from 0 in the pipeline):
- the prediction is issued at t, for target week t + h;
- that week's truth is first reported one week later;
- it is graded at origin t + h + 1, so it reaches NCC in the feedback of step t + h + 1.

At step t, the newest visible score comes from origin t − (h + 1). Scores and truths are always the
first-reported values.

**Why NCC doesn't know the lag.** NCC places an origin in its history when the caller hands over
that origin's feedback. The old code computed the slot `s + W + h − 1` itself. When feedback arrives
on schedule, both give the same windows. Keeping the lag out of NCC means the visibility rule exists
in only one place, the caller.

**Interface** (`src/ncc/online.py`, exported from `ncc/__init__.py`):
```python
@dataclass(frozen=True)
class Feedback:
    origin: int                     # step index of the graded origin
    location: str
    truth: float                    # first-reported truth, target units, finite
    score: float                    # nonconformity score of that task (one per task), finite, >= 0, in target units
                                    # (NCC normalizes it by the truth's std, so it must scale with the target, e.g. |y − ŷ|)
    external_miss: tuple[bool | None, ...] | None = None
        # optional, per alpha: whether the interval that was actually delivered missed (e.g. not `covered`).
        # None, or None for one alpha → use NCC's own issued q̂ for that alpha. The alpha = 1 entry is normally None.

class NCC:
    def __init__(self, cfg: NCCConfig, seed: int): ...
    def step(self, t: int, centers: Mapping[str, float],
             feedback: Sequence[Feedback]) -> dict[str, np.ndarray]:
        """Advance to step t (0 on the first call, then previous t + 1).
        centers: finite base forecast (point, else the median) of every location, at every step.
            The first call fixes the location set and its order; later calls must have exactly those keys.
        feedback: the grades that became visible at t (rules below).
        Returns {location: q̂ of shape (len(alphas),), float64, target units} for every location;
        {} while warming up. q̂ is returned as NCC produced it: not clamped at 0 and possibly
        non-monotone in alpha (the caller builds, clamps and repairs the intervals).
        Raises ValueError on malformed input (see "Input checks"). No data condition raises anything
        else; errors from torch itself (e.g. out of memory, or inside TTA) propagate unchanged."""
```

**Input checks** (`step` raises `ValueError`, with a message naming the offending item). Your
pipeline treats the error as a caller bug and aborts the run.
- `t` isn't the previous `t` + 1 (the first call must have `t == 0`).
- `centers` is empty on the first call, or a later call's keys differ from the first call's (a
  missing or an extra location). The base forecaster is assumed to always produce a forecast for
  every location; if it doesn't, the run fails.
- A center is not finite.
- Feedback breaks any of these rules:
  - The feedback is empty until the first step that brings any. After that, every step's feedback
    covers exactly one origin `o`, and `o` is one more than the origin of the previous step's
    feedback. The first origin is 0.
  - It has exactly one row per location: none missing, none extra, no duplicates.
  - `o < t`.
  - Each `truth` and `score` is finite, and each score is `>= 0`.
  - `external_miss` is `None` or has length `len(alphas)`.

  These checks need no lag. They make "a center and a truth always exist" an enforced rule rather
  than an assumption: every location has an entry at every origin, and the history never has gaps.

**State** (all inside the `NCC` object):
- `store: SeriesStore`, per location and origin: `center` (added at step `origin`), then `truth`,
  `score`, `external_miss` and `arrived` (the step at which the feedback came in).
- `scaler: Scaler`, fitted once at warm-up.
- `locations: list[str]`, fixed by the first call's `centers`, in their order.
- `model: NCCModel | None`.
- `fit_logs: list[LossLog]`, one per `fit` call (warm-up and each retrain). Not part of the output;
  tests read it.
- `history: CalibrationHistory`, the online history.
- `q_adj: dict[location → tensor (A,)]`, the ACI state.
- `issued_steps: int`.
- `warmup_step: int | None`.
- The random generators:
  - `self.np_rng = np.random.RandomState(seed)`;
  - `self.torch_state`, the CPU torch generator state, produced at construction by
    `torch.Generator().manual_seed(seed).get_state()`. That equals the state `torch.manual_seed(seed)`
    gives, without touching any process-global generator.
  - Only when `cfg.device` is not `cpu`: also seed that device's global generator at construction
    (`torch.cuda.manual_seed(seed)` or `torch.mps.manual_seed(seed)`). The encoder RNN has
    `dropout=0.1` (`basicmodels.py:58`), which draws from the device generator during training.

  Every model construction, training call and inference call runs inside
  `torch.random.fork_rng(devices=[])`: it loads `self.torch_state` first and saves it again at exit.
  The location shuffle uses `self.np_rng.shuffle`. `np.random.seed(s)` followed by
  `np.random.shuffle` draws the same permutations as `RandomState(s).shuffle`, so this changes
  nothing numerically, while keeping several `NCC` objects in one process independent.
- **Scope:** seed control and isolation between objects are guaranteed on CPU only.
  - On `cuda` or `mps`, the accelerator generator is process-global and seeded once per object at
    construction; `fork_rng(devices=[])` doesn't isolate it.
  - Accelerator runs are seeded but may not repeat exactly, and they aren't isolated between objects.
  - Objects must be used from one thread, because `fork_rng` changes process-global state.

**History entries** (ordered by origin, placed when their feedback arrives):
- **Created** when a q̂ row is issued or replayed for (origin s, location r). An entry holds
  `(origin, q̂n, ŷn)`, where q̂n is the normalized q̂ and ŷn the normalized center.
- **Completed** once the store has s's truth and score: it adds `yn`, `score_n` and a miss per alpha.
  - `score_n = store.normalized_score(r, s)`.
  - The miss uses one comparison, the same strict test as today's `q̂ < score`:

    ```
    miss[α] = score_n > q̂n[α]
    ```
- **Which q̂ is compared:**
  - *Training replays:* always the q̂ the replay itself computed.
  - *Warm-up replay:* always NCC's own replay q̂. It ignores `external_miss`.
  - *Online history* (origins issued at or after `warmup_step`): for each alpha, `external_miss[α]`
    when it isn't `None`; otherwise the q̂ NCC issued.
- **The window** at step t for location r: the `window_size` completed entries with the largest
  origins among those with `arrived ≤ t`, sorted by origin, with the section 2 placeholders in front
  when fewer exist. The input checks make feedback arrive in origin order, so "latest by origin" and
  "latest by arrival" pick the same entries.

**`Batch`** (redefined in commit 2):
- `locations: list[str]` (shuffled order);
- `err_seq` (B×W×A);
- `score_seq` (B×W×3) = `[ŷn, yn, score_n]` per window entry;
- `qhat_seq` = zeros (B×W×A);
- `input_alphas` (B×A);
- `q_adj` (B×A);
- `window_origins: np.ndarray` (B×W) of int, with −1 for placeholders.

It carries no current truth or score, because those are unknown when the batch is built. Training
reads its targets from the store.

**`step(t, centers, feedback)`**
```python
check inputs                                               # raises ValueError
store.add_centers(t, centers)
for fb in feedback: store.add_truth(fb, arrived=t)
if model is None:
    if store.num_scored_origins() < cfg.n_warmup or not train_origins(t): return {}
    warm_up(t)
issued = infer(t, locations)
issued_steps += 1
if issued_steps % cfg.retrain.period == 0:
    fit(train_origins(t), retrain epochs)                  # the online history is not rebuilt, as today
return issued
```
- `train_origins(t)`: the origins `s < t − cfg.val_steps` that have a score, in ascending order.
  Because feedback always covers every location, an origin is scored for all locations or none.
- `num_scored_origins()`: the number of distinct origins with a score.
- Warm-up waits while `train_origins(t)` is empty (possible only when `val_steps` is large relative to
  `n_warmup`), so the scaler and the first fit always have data.

**`warm_up(t)`**
1. `warmup_step = t`. The location set was fixed at `t = 0` and never changes: the model is trained
   across locations, and each location needs a fitted scaler.
2. For each location, `scaler.fit` on its truths among `train_origins(t)`. If `std == 0` (e.g.
   all-zero counts), use `std = 1`, so NaN can't poison the shared model.
3. Build `NCCModel` from the object's torch generator, then `fit(train_origins(t), cfg.train)`.
4. Replay origins `0 … t−1` in inference mode (the same code as `infer`, with TTA and ACI if
   enabled), each with all locations. This builds `history` and `q_adj` as if the trained model had
   run online from the start, which is what today's calibrate process does from step 0.

**Training (`train.fit`)** uses visible truths and scores only, and never an issued or external
interval. For each stage (or the single run): a fresh optimizer, scheduler and `Patience`, then per
epoch:
```python
replay = CalibrationHistory(locations, ...)       # fresh placeholders
for s in train_origins:                            # gradient steps
    B = locations                                  # all of them: every origin has every location
    batch = replay.window(s, B)                    # entries with arrived <= s
    q̂n = model(batch); total, parts = loss(q̂n, batch.err_seq, batch.input_alphas, score_n[s, B])
    zero_grad; backward; clip_grad_norm_(1, error_if_nonfinite=True); step    # RuntimeError → warning, skip
    replay.record(s, B, q̂n)                        # completes from the store; misses from this q̂n
for s in range(train_origins[-1] + 1, t):          # no gradient; outputs unused; keeps the old RNG sequence
    replay.record(s, locations, model(replay.window(s, locations)))
log.epoch_end(); scheduler.step(); if patience says stop: break
```
Training targets and inputs come only from the store, so they are visible at t by construction.
That removes a look-ahead in today's code for 29105 (h=3) and 29205 (h=4). Their `val_weeks: 1` makes
the initial training, the scaler fit and every retrain use 1 and 2 origins whose truth isn't
observed yet.

**`infer(t, locs)`** runs the model in eval mode (`locs` is always all locations):
1. `batch = history.window(t, locs)`, shuffled with `np_rng`.
2. If `cfg.tta.enabled`, run today's TTA procedure. `aci_correct` applies only when
   `t > 2 · window_size`.
   - The TTA head may reuse an embedding computed once, or recompute it every iteration as today.
     Both give identical numbers, because every other weight is frozen and dropout is off in eval
     mode.
   - Otherwise `q̂n = model(batch)`.
3. Record the entries `(t, q̂n, ŷn)` in `history`, and update `q_adj` for `locs`.
4. Return `{r: scaler.inverse(r, q̂n[r], is_score=True)}` as float64.

**How the pipeline adapter will use it** (not built here):
- one `NCC` per (target, horizon);
- `t` = the position in the pass's timeline;
- `feedback` = its `new_feedback` reduced to one row per task: truth, score, and optionally
  `external_miss` from `~covered` per stream, mapped to NCC's alphas, with `None` for α = 1;
- the intervals come from its score adapter's `construct_quantiles`, with `adjustment = q̂[α]` per
  stream and the α = 1 entry dropped;
- it refuses scores that aren't symmetric (one score per task).

### 3.5 Paper driver (`src/ncc/experiments/run.py`)
```python
def run_experiment(exp: Experiment, seed: int, device: str | None = None) -> Path:
    cfg = exp.ncc if device is None else replace(exp.ncc, device=device)
    pred = pickle_load(paths.base_pred(exp.base_pred_id))['base_pred']
    locations = list(pred[0][0].keys())                     # pickle order
    yhat[t][r] = float(pred[t][0][r][exp.horizon - 1]); y[t][r] = float(pred[t][1][r][0][exp.horizon - 1])
    N = len(pred); first_test = N - exp.test_steps
    results = {r: {exp.horizon: []} for r in locations}     # pre-initialized, as forecast4eval does
```
`float(...)` makes the values Python floats (float64) even if a pickle stores float32 arrays.
- **Commit 1** keeps today's two-phase flow, in one process:
  1. Seed `random`, `np.random` and `torch.manual_seed` with `seed`.
  2. `data, scaler = build_series(pred, exp.horizon, train_steps=first_test − val_steps)`.
  3. Build the model and train it as `-t` does.
  4. Reseed the same way, build a fresh model, and load the trained `state_dict`.
  5. Run `online.calibrate(...)`, which is `forecast4eval` for one horizon.

  The in-process reseed reproduces the random-number sequence of the old two processes exactly.
  (That holds on CPU; on accelerators, see section 3.4, "Scope".)
- **Commit 2:**
  - `cfg = replace(cfg, n_warmup=first_test − exp.horizon + 1)`, then build `NCC(cfg, seed)`.
  - For each `t` in `range(min(N, exp.max_steps + 1))`, with `s = t − exp.horizon`:

    ```python
    q = ncc.step(t, {r: yhat[t][r] for r in locations},
                 [Feedback(s, r, y[s][r], abs(y[s][r] − yhat[s][r])) for r in locations] if s >= 0 else [])
    ```
  - For `t ≥ first_test`, save `(q[r], np.array([yhat[t][r]]), np.array([y[t][r]]))`, float64, in
    the section 2 results layout.
  - The paper's `horizon` counts steps ahead from 1, so its truth becomes visible `horizon` steps
    later. That equals the pipeline's `h + 1`.
  - With this `n_warmup`, warm-up fires exactly at `t = first_test`. The first feedback arrives at
    `t = horizon`, for origin 0, which satisfies the input checks.
  - This driver is for reproducing the paper only. Deriving `n_warmup` from `N` isn't prefix-stable,
    and doesn't need to be.
- **Both commits** write `results/<exp_id>_<seed>.pkl` and `models/<exp_id>_<seed>.pt` (the final
  `state_dict`). The losses pickle is no longer written, since nothing reads it and the losses are
  logged. Commit 1 saves float32 arrays, as `4daa066` does. Commit 2 saves float64 arrays and the raw
  `y_hat`/`y`. The layout is unchanged, and `evaluate` reads both.

`experiments/paths.py`: `ROOT = Path(__file__).resolve().parents[3]` (the repository root, with an
editable install), plus:
- `base_pred(id)` → `ROOT/results/base_pred/saved_pred_<id>.pickle`;
- `results(exp_id, seed)` → `ROOT/results/<exp_id>_<seed>.pkl`;
- `weights(exp_id, seed)` → `ROOT/models/<exp_id>_<seed>.pt`;
- `csv_dir()` → `ROOT/results/csvs`;
- `clip_idx(dataset)` → `ROOT/data/<dataset>_clip_idx.pkl`;
- `configs()` → `ROOT/configs`.

Every function reads the module-level `ROOT` when called, so tests can monkeypatch it.

`experiments/config.py`:
- `load_experiment(exp_id) -> Experiment(exp_id, base_pred_id, horizon, test_steps, max_steps, seed,
  seeds, ncc: NCCConfig)`.
- The dataset name used by `evaluate -c` is read from the base-pred pickle's `params['dataset']`, as
  `eval_cp.load_clip_idx` does today.

`experiments/evaluate.py` is today's `eval_cp.py`, with the same options and outputs:
- `results/csvs/<id>_<seed>.csv`: per region, with median and mean rows on top;
- `results/csvs/<id>_multi_seeds.csv`: per seed, with a mean row on top.

`eval_metrics` returns a dict keyed by `METRIC_NAMES`, which moves to `metrics.py` in the current
order: `rmse, nrmse, mape, cs_score, norm_crps_val, mean_wis, mono, cov_50, cov_95, cov_90,
pi_length_90, dvs, qice` (`eval_cp.py:10`). The CSVs write the columns in that order. The metric
function bodies stay unchanged, including the in-place `mape` NaN side effect.

CLI (`python -m ncc ...`, runnable from any directory):
```
python -m ncc run -i <exp_id> [-s <seed>] [--device <cpu|cuda:0|mps>]
python -m ncc evaluate -i <exp_id> [-s <seed> | -u] [-c] [-o]
```
- `-s` defaults to `seed` in the experiment config.
- `--device` overrides `ncc.device` for that run only; the config files stay unchanged.
- `-u` evaluates every seed in `seeds`.
- `-c` evaluates only the steps in `data/<dataset>_clip_idx.pkl`.
- `-o` sorts the widths so they shrink as alpha grows.

## 4. Decisions and alternatives

| Decision | Rationale | Rejected alternative, and why |
|---|---|---|
| Stateful core with `step()`; the drivers do the I/O | This is the pipeline's contract, and one code path serves both uses | Keeping separate train and calibrate processes: can't be one object per run, and replays the whole file |
| The caller decides visibility; NCC places an origin in its history when its feedback arrives | The lag rule exists in one place (the caller); NCC needs no lag arithmetic. With on-schedule feedback, the windows are identical to the old slot history | Slot history indexed by `s + W + h − 1` inside NCC: a second copy of the lag rule, which can drift from the pipeline's |
| Feedback must be complete and in order, or `step` raises | A truth always exists; enforcing it keeps the history free of gaps and makes caller bugs loud | Tolerating missing feedback: needs gap handling in the window, and hides caller bugs |
| Training misses from NCC's own q̂; online misses from the externally delivered interval when given, otherwise from NCC's own q̂; warm-up replay from own q̂ | Training can only grade its own predictions. Online, what was actually delivered (e.g. a Hub submission) is what the history should reflect. Before warm-up NCC delivered nothing, so there's nothing external of its own to reflect | Always own q̂: ignores what was delivered. Always the delivered interval: impossible in training |
| Warm-up counted in visible scores (`n_warmup`) | Doesn't depend on the horizon, like the pipeline's ACI burn-in. Prefix-stable | The paper's split counted from the end: depends on future steps |
| Train on scored origins `s < t − val_steps` | Causal by construction. Identical to the paper whenever `val_steps ≥ horizon − 1` | The paper's range `[0, t − val_steps)`: uses unobserved truths for 29105 and 29205 |
| Locations fixed by the first step's `centers`; every step must bring a center and (once feedback starts) a feedback row for each, or `step` raises | The base forecaster always forecasts every location, so a missing one is a caller bug. The model is trained across locations, and each location needs a fitted scaler. It also removes all partial-batch and empty-origin cases | Skipping a location without a center: needs gap handling in the history and the batches, and an origin with no centers at all can't be expressed in the feedback rule. Adding late locations: scalers fitted mid-run, and a different set of locations trained together |
| One model (one `NCC` object) per horizon | Every paper config has one horizon, and the old multi-horizon calibration leaks: horizons ran one after another, so later ones started from a model retrained through the test period | A shared model across horizons: leaks, and the model has no horizon input |
| The caller supplies the score | NCC works for any score with one non-negative value per task | Computing `|y − ŷ|` inside NCC: ties it to one score. Scores with one value per alpha (CQR) or with lower and upper parts are not supported |
| Each object owns its random generators, one stream per run, guaranteed on CPU | Several objects in one process stay deterministic | Global generators: objects disturb each other. Owning the accelerator generators too: not needed now, since `device` is a config option |
| Defaults only in `NCCConfig` | One source of truth for defaults | Repeating them in `default.yaml`: the two copies drift apart |
| Metric code kept as is; only the interface changes | Protects the reported numbers | A rewrite on arrays: may change metric values |

## 5. Detailed implementation

### Commit 1: structure (the output must match `4daa066` exactly)

1. **Branch.** `git checkout -b ncc-refactor ncc-cleanup`.
   - This plan file (`docs/ncc-refactor-plan.md`) starts untracked on `ncc-cleanup`. It carries over
     to the new branch and is committed as part of commit 1.
2. **File** `pyproject.toml` (new).
   - **Change:**
     - a setuptools build;
     - `[project] name = "ncc"`, `version = "0.2.0"`, `requires-python = ">=3.10"`,
       `readme = "README.md"`;
     - `dependencies = [numpy, torch, PyYAML, scikit-learn, properscoring]`;
     - `[project.optional-dependencies] test = ["pytest"]`;
     - package discovery under `src`;
     - a pytest marker `slow` registered under `[tool.pytest.ini_options]`, with
       `addopts = "-m 'not slow'"`.
   - **Reason:** install from the repository root. The code uses `X | None` annotations, which need
     Python 3.10.
   - Delete `src/setup.py` and `requirements.txt`.
3. **File** `bin/setup.sh`.
   - **Change:** from `bin/`, run `pip install -e "..[test]"`, then
     `mkdir -p ../results/base_pred ../results/csvs ../models`.
   - **Reason:** can be rerun, and uses the new install location.
4. **Files** `src/ncc/layers.py`, `model.py`, `losses.py`, `series.py`, `history.py`, `train.py`,
   `online.py`, `config.py`, `__init__.py`, `__main__.py`.
   - **Change:** move the code from `src/forecaster/*` as section 3.3 describes.
     - `series.build_series(base_pred, horizon, train_steps) -> (data, scaler)` is the body of
       `End2endCPDataset.e2ecp_seq_data` for one horizon. It takes the already-loaded `base_pred`
       list and returns `data[region] = [(y_hat_n, y_n), ...]`.
     - `online.calibrate(model, data, scaler, cfg, regions, horizon, first_test, max_steps) -> results`
       is `forecast4eval` for one horizon, with `forward_pass` and `tta_forward_pass` merged into one
       prediction path. `horizon` sets the history length `W + N + horizon − 1`, the write slot
       `t + W + horizon − 1` and the results key.
     - `train.fit(model, data, cfg, regions, horizon, train_end, walk_end, schedule) -> LossLog`
       holds `train` and `stage_train`. Per epoch: gradient steps over `[0, train_end)`, then the
       gradient-free walk over `[train_end, walk_end)`. `schedule` is the epochs (single run) or the
       per-stage epochs.
       - Initial training: `train_end = first_test − val_steps`, `walk_end = first_test`, with
         `train.epochs` / the `train.stages` epochs.
       - Retrain at test step t: `train_end = t − val_steps`, `walk_end = t`, with `retrain.epochs` /
         `retrain.stage_epochs`. This is what today's `retrain` gets by shrinking `test_weeks`
         (`e2ecp_utils.py:256-265`).
   - **Important details:**
     - Keep the module construction order and `embed`'s stack/permute exactly.
     - Keep one `np.random.shuffle` per batch and per step, including the gradient-free walk.
     - Keep the `t > 2·window_size` ACI condition.
     - Keep `Batch` and `CalibrationHistory` as today's `Batch`/`RollingSequences` in this commit.
     - Delete `src/forecaster/`.
5. **Files** `src/ncc/experiments/{__init__,paths,config,run,evaluate,metrics,cli}.py`.
   - **Change:** sections 3.2 and 3.5, with the commit 1 flow.
6. **Files** `configs/default.yaml` and `configs/experiments/*.yaml`.
   - **Change:** convert with the key table in section 3.2, then delete `setup/`.
   - **Important details:** verify with a throwaway script that each experiment's effective values
     (YAML merged onto the `NCCConfig` defaults) equal the old `load_all_params` output, after
     mapping the keys. The only allowed differences are the dropped unused keys.
7. **File** `tests/test_ncc.py`: T1 and T7 (section 8).
8. **File** `tests/test_real_data.py`: R3 (section 8).
9. **File** `README.md`.
   - **Change:**
     - update the layout, setup, commands (`python -m ncc run/evaluate`, from any directory,
       `--device`) and the configuration section;
     - fix the retraining sentence: retraining at test step t trains on steps `[0, t − val_steps)`,
       not on "everything observed so far".
   - Keep the experiment table, Drive links, CSV column descriptions, data formats, contact and
     bibtex.
10. **Verify** with R1 and T1/T7, then run R3 (section 8). Then commit: "Restructure into the ncc
    package".

### Commit 2: step-based core

1. **File** `src/ncc/series.py`.
   - **Change:** add `SeriesStore`:
     - `add_centers(t, centers)`;
     - `add_truth(fb, arrived)`;
     - `num_scored_origins()`;
     - `scored_origins()`;
     - `center(r, s)`, `truth(r, s)`, `score(r, s)`, `external_miss(r, s)`, `arrived(r, s)`;
     - `normalized_score(r, s) -> np.float32`, which returns `score / std`. It's the only place a
       score is normalized, so R2 can patch it.
   - `Scaler.fit` sets `std = 1` when it would be 0. Values are stored in float64.
   - Remove `build_series`.
2. **File** `src/ncc/history.py`.
   - **Change:** `CalibrationHistory` becomes the entry list of section 3.4.
     - `record(origin, locations, q̂n, source)`, where `source` is `"train"`, `"replay"` or
       `"online"`. It decides whether `external_miss` is read: only for `"online"`.
     - Entries complete lazily from the store.
     - `window(t, locations) -> Batch`.
   - **Important details:** placeholders and tensor layout exactly as today's `RollingSequences.batch`.
     `qhat_seq` is zeros. Shuffle with the `RandomState` passed in. `Batch` is redefined as in
     section 3.4.
3. **File** `src/ncc/train.py`.
   - **Change:** `fit(model, store, scaler, locations, origins, t, epochs, cfg, np_rng) -> LossLog`
     (section 3.4).
4. **File** `src/ncc/online.py`.
   - **Change:** replace `calibrate` with `Feedback` and `NCC` (section 3.4). `aci_correct`,
     `frozen_except` and the TTA loop stay in this file.
   - **Important details:**
     - The torch random state is loaded and saved around every model build, fit and inference.
     - ACI and TTA get `t` from the caller.
     - The input checks run before any state changes, so a rejected call leaves the object
       unchanged.
5. **File** `src/ncc/online.py` (same file as step 4): `NCC.__init__` raises `ValueError` if
   `n_warmup` is `None` or < 1.
6. **File** `src/ncc/experiments/run.py`: switch to the commit 2 driver (section 3.5) and remove the
   in-process reseed.
7. **File** `tests/test_ncc.py`: add T2–T6 and T8 (section 8). **File** `tests/test_real_data.py`:
   set its `THRESHOLDS` constant to the commit 2 values of R3 (section 8).
8. **File** `README.md`: add a section "Using NCC step by step". It covers:
   - the `NCC.step` interface and `Feedback`;
   - the visibility rules and input checks;
   - warm-up, and the location set fixed by the first step;
   - where misses come from;
   - that commit 2's numbers differ from commit 1's (section 6).
9. **Verify** with R2, T1–T8 and R3, then commit: "Make NCC a step-based method".

End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not push.

## 6. Constraints and invariants
- **Results file layout and CSV layout are unchanged**, so the published results on Drive still
  evaluate. The dtype of the saved arrays changes from float32 to float64 in commit 2.
- **No file I/O in the core.** `src/ncc/*.py` outside `experiments/` must not read or write files,
  and must not import `pickle`, `yaml`, `csv` or `ncc.experiments`. `__main__.py` is the only
  exception.
- **The only error `step` raises on its own is `ValueError`, for the input checks in section 3.4.**
  Every valid input is handled (warm-up returns `{}`). Errors raised by torch itself propagate.
- **Prefix stability (commit 2, CPU).** For any T, running steps `0…T` and `0…T+1` gives identical
  outputs for steps `0…T`.
- **Determinism and isolation (commit 2, CPU).** The same inputs and seed give identical outputs.
  Interleaving the calls of two `NCC` objects gives the same outputs as running each alone.
- **Model construction order:** score, qhat, error and alpha encoders, then fusion, the quantile head
  and the TTA head.
- **Kept on purpose** (latent bugs, to be fixed later; add a `# TODO(paper):` comment at each):
  - the q̂ history is never written, so `qhat_seq` is zeros;
  - no best state is restored;
  - the gradient-free walk is unused, and early stopping reads the training loss;
  - ACI runs only inside TTA;
  - `mape` writes NaN into the targets in place, which affects later metrics.
- **How commit 2's numbers differ from commit 1's:**
  - Slightly, in every config. The score is now given by the caller, in float64, and normalized as
    `score/std`, where it used to be `|ŷn − yn|` on float32 tensors. Training amplifies such
    differences. The saved arrays are also float64 and raw.
  - Clearly, in 29105 and 29205: training now uses only truths that have arrived.

## 7. Edge cases and failure modes
- **A location with constant truth during warm-up:** `std = 1`, and its outputs stay finite.
- **`n_warmup` scored origins are visible but none is before `t − val_steps`:** NCC stays in warm-up
  and returns `{}` until one is.
- **A location missing from `centers`, or a new one appearing:** `ValueError`. The run fails, because
  the base forecaster is assumed to forecast every location at every step.
- **Malformed input** (a skipped `t`, missing, extra, duplicate or out-of-order feedback, non-finite
  values): `ValueError` with the object unchanged. The caller must fix the input.
- **Non-finite gradient:** a warning, and the optimizer step is skipped, as today.
- **An exception inside TTA:** `frozen_except` restores `requires_grad` before it propagates.
- **Negative or non-monotone q̂** (possible after ACI or TTA): returned as is. The caller clamps and
  repairs, and the paper's `evaluate` clamps negatives to 0 (`eval_cp.py:29`).
- **GPU or MPS:** outputs are seeded but may not repeat exactly, and objects aren't isolated. The
  prefix-stability and determinism tests run on CPU only.
- **Runtime:** each pipeline run retrains from warm-up. Accepted, with no caching. Runtime is not a
  goal of this plan: commit 2's per-entry history may be slower than today's tensor slices. Record
  the run times in R3; don't optimize unless asked.
- **`max_steps` smaller than `first_test`:** no test steps are saved (as today), and the results
  file holds empty lists. `evaluate` then raises `ValueError` from `MinMaxScaler` on empty input
  (`weighted_IS`, `metrics.py:132-133`, the first metric `eval_metrics` computes), as today.

## 8. Validation

### R1: commit 1 vs `4daa066` (exact; throwaway harness in the session scratchpad, not committed)
The real base predictions aren't in git. Generate synthetic ones in the section 2 format:
- 60 steps, and 3 locations `A`, `B`, `C`;
- `predictions[r]`: a float64 array of shape (4,);
- `addition_infos[r] = (truth float64 (4,), ...)`;
- all values positive and well away from 0 (e.g. in [30, 140]). R2's relative tolerance needs this:
  commit 1's `y_hat`/`y` are float32 round trips, whose error is relative to the spread, not to the
  value;
- `params = {'dataset': <name>}` for ids 19 (covid), 30 (weather), 35 (smd) and 50 (electricity).

Use experiments 22051, 22052, 29130 and 29205. Together they cover TTA on and off, stages on and off,
`init_err_with_alpha`, and horizons 1 and 4. Shrink the settings so each run takes seconds: device
cpu, epochs 2, stage epochs [1, 1, 1], retrain epochs 1 / [1, 1, 1], `val_steps` 5, `test_steps` 15.

- **Old side:**
  - Create `git worktree add <scratch>/old 4daa066`.
  - Run `e2ecp.py -i <id> -s 1 -t`, then `e2ecp.py -i <id> -s 1`, from a sandbox `a/b/` directory,
    with `PYTHONPATH=<scratch>/old/src`. The sandbox holds `setup/` (with the shrunk settings),
    `results/base_pred/`, `results/csvs/`, `models/` and `data/` two levels up, so the old `../../`
    paths resolve inside it. `data/` holds the sandbox clip files described under "Pass".
- **New side:** a small Python wrapper in the scratchpad sets `ncc.experiments.paths.ROOT` to a
  second sandbox (same contents, with `configs/` instead of `setup/`), then calls
  `ncc.experiments.cli.main(["run", "-i", id, "-s", "1", "--device", "cpu"])`, and likewise for
  `evaluate`. So `cli.main` takes an optional argv list.
- **Pass:**
  - exactly equal arrays (values and dtype) in the results pkl;
  - equal CSVs from `evaluate` with and without `-c` / `-o`.

  For `-c`, write each sandbox's `data/<dataset>_clip_idx.pkl` with `list(range(1, 8, 2))`, because
  there are only 15 test steps (the real clip files index further than that).

### R2: commit 2 vs commit 1 (rough, but diagnostic)
Use the R1 cases plus a 29205 variant with `val_steps` 1, with two harness-only patches that undo the
intended changes:
1. After the first `fit` in `warm_up` returns, reseed `np_rng` and the torch state with the seed, and
   build one throwaway `NCCModel`. This reproduces commit 1's in-process reseed.
2. Replace `SeriesStore.normalized_score` with the old computation,
   `abs(float32((ŷ−mean)/std) − float32((y−mean)/std))`.

**Pass:**
- Every case with `val_steps ≥ horizon − 1` matches commit 1 within `np.allclose(rtol=1e-6,
  atol=0)` on q̂, `y_hat` and `y`, after casting both sides to float64.
- The 29205 variant with `val_steps` 1 differs by more than that. This shows that the causal
  training range is the only remaining change.

Why a tolerance works here: with both patches, the only remaining numeric difference is the final
multiply by `std` (float64 versus float32) and the raw `y_hat`/`y`. Neither feeds back into NCC, so
a real bug shows up as a difference far above 1e-6.

### R3: real data on MPS (`tests/test_real_data.py`, marked `slow`)
- **Data:** before running, download `saved_pred_50.pickle` (electricity) and `saved_pred_19.pickle`
  (covid) from the Drive folder in section 2 into `results/base_pred/`, using the Claude Google
  Drive connector. The test is skipped when a file is missing or `torch.backends.mps.is_available()`
  is false.
- **Command:** `pytest -m slow tests/test_real_data.py` (the default `addopts` excludes slow tests).
- **Isolation:** the test monkeypatches `paths.ROOT` to `tmp_path`, with `results/base_pred`,
  `configs` and `data` symlinked from the repository. So it never overwrites a downloaded paper
  result in `results/` (the README tells users to put them there under the same names).
- **Runs:** `run_experiment(load_experiment(id), seed=1, device="mps")`, then evaluate. The paper
  configs are used unchanged apart from the device.
  - 22052 (electricity, 1 step ahead): evaluated on all test steps (it has no clip file).
  - 22055 (covid, 1 step ahead): evaluated with `-c`, on the steps in `data/covid_clip_idx.pkl`.
  - Unsorted (no `-o`) in both.
  - The metric is `cov_90` from the `mean` row of `results/csvs/<id>_1.csv`.
- **Thresholds after commit 1:** `cov_90 ≥ 0.80` for 22052 and `≥ 0.60` for 22055, held in a
  module constant `THRESHOLDS = {"22052": 0.80, "22055": 0.60}`.
- **If a threshold isn't met after commit 1:** don't change the structure or any parameter. Look for
  an implementation bug:
  1. Run the old code (`4daa066`, with its config device set to `mps` in the old sandbox) on the same
     files, seed and device.
  2. If the old code meets the threshold and the new code doesn't, the refactor is wrong. Find the
     first point where the two diverge (data arrays, scaler statistics, initial weights, per-epoch
     losses, q̂ per step) and fix it.
  3. If both miss it, check the data loading, the horizon index, the scaler, the evaluation (clip
     indices, clamping, metric order) and the dtypes.
  4. If nothing is wrong, commit 1's value becomes the reference for commit 2. Report both numbers
     with the evidence.
- **Threshold after commit 2:** `cov_90 ≥` (commit 1's value, or the stated threshold if commit 1
  met it) `− 0.05`. The margin allows for MPS noise and for the intended changes of section 6. The
  same rule applies if it's missed: debug, don't tune, and report. Commit 2 writes these values into
  `THRESHOLDS` (e.g. `{"22052": 0.75, "22055": 0.55}` if commit 1 met both), with a comment giving
  commit 1's measured values.
- Record the measured `cov_90` values and run times in the final report. They are not written into
  this file.

### Unit tests (`tests/test_ncc.py`, pytest, CPU, synthetic, under 60 s in total)
Use a tiny config: `window_size` 3, hidden sizes 8, one encoder layer, epochs 1, 3 locations, 30
steps, horizon 2 (so feedback for origin s arrives at s + 2), `n_warmup` 10, `val_steps` 0,
`retrain.period` 3, and TTA on.

- **T1: end to end.** `run_experiment` then `evaluate`, with `paths.ROOT` monkeypatched to
  `tmp_path` and the working directory elsewhere. The results pkl, the weights and the CSV exist, and
  the CSV has the `METRIC_NAMES` columns. This checks the layout and that commands run from any
  directory.
- **T2: warm-up.** `step` returns `{}` until `n_warmup` scored origins are visible, then returns
  finite q̂ for every location.
- **T3: prefix stability.** Outputs for steps 0…T are identical when the run stops at T and at T+1.
  The step T+1 inputs use extreme values, so any leak would show.
- **T4: determinism and isolation.** Two runs with the same seed are identical. Interleaving the
  calls of two `NCC` objects (different seeds) gives the same outputs as running each alone.
- **T5: input checks.** Each of these raises `ValueError` and leaves the object usable:
  - a skipped `t`;
  - `centers` missing one location, and `centers` with an extra location;
  - feedback missing one location, feedback for an extra location, and a duplicate feedback row;
  - feedback that skips an origin;
  - two origins in one step;
  - a negative score.

  To show the object is still usable, after the error, the correct call succeeds and matches a run
  without the error.
- **T6: constant location.** A location whose truth is always 0 gives finite q̂.
- **T7: loss log and patience** (commit 1). `LossLog` turns a coverage part above 1e7 into NaN, and
  its epoch total is the sum of the `nanmean`s. `Patience(3)` stops after 3 non-improving epochs,
  resets on improvement, and follows the section 3.3 NaN behavior. Checked against `EarlyStopping`
  from `4daa066` on the same loss sequences, with the expected values written into the test.
- **T8: where misses come from.** Two runs are identical except for `external_miss`:
  - If they differ only on origins before `warmup_step`, the outputs are identical (warm-up ignores it).
  - If they differ on an online origin o (all True versus all None; pick an o where NCC's own q̂
    covered for at least one alpha and location, so the runs really differ), the outputs are identical up to
    the step before o's feedback arrives, and differ from that step on. The feedback is stored with
    `arrived = t` before `infer(t)` reads the entries with `arrived ≤ t`, so the window at the arrival
    step already shows the forced misses. This matches today's code, where the window at step t
    already contains origin `t − h`.
  - Training ignores `external_miss`: with TTA off (TTA fits the TTA head on the window, so it would
    carry the difference into the weights), `fit_logs` of the two runs are equal.

Location handling is covered by T5 (a missing or extra location in `centers` or in the feedback
raises).

## 9. Implementation order
1. Commit 1, steps 1–10: package, configs, driver, evaluate, README, T1, T7, R1, R3.
2. Commit 2, steps 1–9: store, history, train, `NCC`, driver, T2–T6, T8, R2, R3, README.

## 10. Definition of done
- `ncc-refactor` exists with exactly two new commits on top of `4daa066`, not pushed.
- `pip install -e ".[test]"` works in `epi4cast`. `python -m ncc run` and `python -m ncc evaluate`
  work from any directory.
- R1 passes exactly after commit 1, and R2 as specified after commit 2.
- `pytest tests/` passes on CPU (slow tests excluded by default).
- R3 is run after each commit on MPS. It either meets its thresholds, or the investigation of
  section 8 finds no implementation bug. The measured values and evidence are reported.
- No `src/forecaster/`, `setup/`, `src/setup.py` or `requirements.txt` remain.
- Nothing in `src/ncc/` outside `experiments/` does file I/O or imports `ncc.experiments`. Check with
  `grep -rnE "import (pickle|yaml|csv)|open\(|ncc\.experiments" src/ncc --include='*.py'`; the only
  matches may be under `src/ncc/experiments/` and in `src/ncc/__main__.py`.
- The README describes the new commands, the step interface, the visibility rules and input checks,
  and the corrected retraining description.
