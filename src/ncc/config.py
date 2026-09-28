"""NCC configuration: frozen dataclasses that hold the single source of defaults.

Build a config from a plain (e.g. YAML-parsed) dict with `NCCConfig.from_dict`. Fields without a
default are required.
"""
from dataclasses import MISSING, dataclass, field, fields


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 64                  # size of each encoding and of the fused embedding
    score_encoder_hidden_dim: int = 64    # GRU hidden size of the (y_hat, y, score) history encoder
    score_encoder_num_layers: int = 2     # GRU layers of the score encoder
    qhat_encoder_hidden_dim: int = 32     # GRU hidden size of the q-hat history encoder
    qhat_encoder_num_layers: int = 4      # GRU layers of the q-hat encoder
    error_encoder_hidden_dim: int = 32    # hidden size of the miss history encoder
    alpha_encoder_hidden_dim: int = 32    # hidden size of the alpha encoder
    q_fc_hidden_dim: int = 128            # hidden size of the quantile head


@dataclass(frozen=True)
class LossWeights:
    quantile: float = 0.01                # weight of the quantile (pinball) loss
    coverage: float = 1                   # weight of the coverage loss
    efficiency: float = 0.00108           # weight of the efficiency (width) loss
    monotonicity: float = 1               # weight of the monotonicity loss; used in every stage


@dataclass(frozen=True)
class Stage:
    epochs: int                           # epochs of this stage
    weights: tuple                        # (quantile, coverage, efficiency) loss weights of this stage


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    lr: float = 0.0001                    # Adam learning rate
    gamma: float = 0.01                   # learning rate decay factor, applied every 50 epochs
    epochs: int = 500                     # epochs of the initial training (ignored with stages)
    K: float                              # temperature of the sigmoid in the coverage and efficiency losses
    weights: LossWeights = field(default_factory=LossWeights)   # loss weights (only monotonicity is used with stages)
    stages: tuple | None = None           # tuple of Stage: train in stages, each with its own epochs and weights


@dataclass(frozen=True)
class RetrainConfig:
    period: int = 5                       # retrain every `period` issued steps (steps that return q-hats)
    epochs: int = 50                      # epochs of a retrain without stages
    stage_epochs: tuple = (10, 10, 1)     # epochs per stage of a retrain with stages


@dataclass(frozen=True)
class TTAConfig:
    enabled: bool = False                 # test-time adaptation of q-hats that are not monotone in alpha
    lr: float = 0.0005                    # Adam learning rate of the TTA head
    reg_factor: float = 1                 # weight of the RMS of the TTA output in the TTA loss
    aci_lr: float = 0.01                  # step size of the ACI correction (which only runs inside TTA)


@dataclass(frozen=True, kw_only=True)
class NCCConfig:
    alphas: tuple = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)  # target miscoverage rates, strictly ascending
    device: str = "cpu"                   # torch device of the model and the losses
    window_size: int                      # length of the calibration history the model reads
    val_steps: int                        # training at step t uses the scored origins s < t - val_steps
    init_err_with_alpha: bool = False     # start the miss history at alpha instead of 1
    n_warmup: int | None = None           # scored origins NCC waits for before its first training; required by NCC()
    model: ModelConfig = field(default_factory=ModelConfig)        # model sizes
    train: TrainConfig                    # initial training
    retrain: RetrainConfig = field(default_factory=RetrainConfig)  # retraining during the test steps
    tta: TTAConfig = field(default_factory=TTAConfig)              # test-time adaptation

    def __post_init__(self):
        if any(a >= b for a, b in zip(self.alphas, self.alphas[1:])):
            raise ValueError(f"alphas must be strictly ascending, got {self.alphas}")
        stages = self.train.stages
        if stages is not None and len(self.retrain.stage_epochs) != len(stages):
            raise ValueError(f"retrain.stage_epochs has {len(self.retrain.stage_epochs)} entries, "
                             f"but train.stages has {len(stages)}")

    @classmethod
    def from_dict(cls, d):
        """Builds the config from a nested dict. Lists become tuples; a None value counts as missing.

        Raises ValueError on an unknown key, a missing required key, alphas that are not strictly
        ascending, or retrain.stage_epochs whose length differs from train.stages.
        """
        return _build(cls, d, "ncc", {
            "alphas": lambda v, where: tuple(v),
            "model": lambda v, where: _build(ModelConfig, v, where),
            "train": _train_config,
            "retrain": lambda v, where: _build(RetrainConfig, v, where, {"stage_epochs": lambda s, w: tuple(s)}),
            "tta": lambda v, where: _build(TTAConfig, v, where),
        })


def _train_config(d, where):
    def stages(v, where):
        return tuple(_build(Stage, s, f"{where}[{i}]", {"weights": lambda w, _: tuple(w)}) for i, s in enumerate(v))
    return _build(TrainConfig, d, where, {
        "weights": lambda v, where: _build(LossWeights, v, where),
        "stages": stages,
    })


def _build(cls, d, where, converters=None):
    """cls(**d) after checking the keys of d and converting the values named in converters."""
    converters = converters or {}
    d = {key: value for key, value in (d or {}).items() if value is not None}
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(d) - names)
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {', '.join(unknown)}")
    for f in fields(cls):
        if f.default is MISSING and f.default_factory is MISSING and f.name not in d:
            raise ValueError(f"missing required key {where}.{f.name}")
    kwargs = {key: converters[key](value, f"{where}.{key}") if key in converters else value
              for key, value in d.items()}
    return cls(**kwargs)
