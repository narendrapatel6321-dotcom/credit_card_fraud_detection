"""
Pipeline configuration.

Defines all configurable paths, model hyperparameters,
training settings, and business cost parameters used
throughout the fraud detection pipeline.
"""

import logging
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger: logging.Logger = logging.getLogger(__name__)


class PipelineConfig(BaseSettings):
    """
    Central configuration shared by every pipeline component.

    Values can be overridden through environment variables
    or constructor arguments.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # File-system paths (relative — override per-environment)

    DATA_PATH: Path = Path("data/creditcard.csv")
    """Path to the raw Kaggle European Credit Card Fraud CSV file."""

    MODEL_SAVE_PATH: Path = Path("artifacts/model.json")
    """Path where the trained XGBoost booster is serialised to disk."""

    TRANSFORMER_STATS_PATH: Path = Path("artifacts/transformer_stats.json")
    """Path to saved transformer statistics."""

    SHAP_PLOT_PATH: Path = Path("artifacts/shap_summary.png")
    """Path where the SHAP feature-importance summary plot is saved."""
    
    # XGBoost hyperparameter search bounds (consumed by Optuna)

    XGB_MAX_DEPTH_MIN: int = Field(default=3, gt=0)
    """Minimum tree depth explored during Optuna hyperparameter search."""

    XGB_MAX_DEPTH_MAX: int = Field(default=10, gt=0)
    """Maximum tree depth explored during Optuna hyperparameter search."""

    XGB_LR_MIN: float = Field(default=0.005, gt=0.0)
    """Minimum learning rate (eta) explored during Optuna search."""

    XGB_LR_MAX: float = Field(default=0.3, gt=0.0)
    """Maximum learning rate (eta) explored during Optuna search."""

    XGB_SUBSAMPLE_MIN: float = Field(default=0.5, gt=0.0, le=1.0)
    """Minimum row-subsample ratio explored during Optuna search."""

    XGB_SUBSAMPLE_MAX: float = Field(default=1.0, gt=0.0, le=1.0)
    """Maximum row-subsample ratio explored during Optuna search."""

    XGB_COLSAMPLE_MIN: float = Field(default=0.5, gt=0.0, le=1.0)
    """Minimum column-subsample-by-tree ratio explored during Optuna search."""

    XGB_COLSAMPLE_MAX: float = Field(default=1.0, gt=0.0, le=1.0)
    """Maximum column-subsample-by-tree ratio explored during Optuna search."""

    XGB_GAMMA_MIN: float = Field(default=0.0, ge=0.0)
    """Minimum minimum-split-loss (gamma) explored during Optuna search."""

    XGB_GAMMA_MAX: float = Field(default=5.0, ge=0.0)
    """Maximum minimum-split-loss (gamma) explored during Optuna search."""

    XGB_REG_ALPHA_MIN: float = Field(default=1e-8, gt=0.0)
    """Minimum L1 regularisation alpha explored during Optuna search."""

    XGB_REG_ALPHA_MAX: float = Field(default=1.0, gt=0.0)
    """Maximum L1 regularisation alpha explored during Optuna search."""

    XGB_REG_LAMBDA_MIN: float = Field(default=1e-8, gt=0.0)
    """Minimum L2 regularisation lambda explored during Optuna search."""

    XGB_REG_LAMBDA_MAX: float = Field(default=1.0, gt=0.0)
    """Maximum L2 regularisation lambda explored during Optuna search."""

    XGB_N_ESTIMATORS_MIN: int = Field(default=100, gt=0)
    """Minimum boosting rounds explored during Optuna search."""

    XGB_N_ESTIMATORS_MAX: int = Field(default=1000, gt=0)
    """Maximum boosting rounds explored during Optuna search."""

    XGB_MIN_CHILD_WEIGHT_MIN: int = Field(default=1, gt=0)
    """Minimum value of ``min_child_weight`` explored during Optuna search.
    Controls minimum sum of instance weights required in a child node — critical
    for imbalanced datasets where small leaves overfit to rare fraud patterns."""

    XGB_MIN_CHILD_WEIGHT_MAX: int = Field(default=20, gt=0)
    """Maximum value of ``min_child_weight`` explored during Optuna search."""

    # Training / tracking / split parameters

    OPTUNA_N_TRIALS: int = Field(default=50, gt=0)
    """Number of Optuna optimisation trials to execute during model search."""

    EARLY_STOPPING_ROUNDS: int = Field(default=50, gt=0)
    """XGBoost early-stopping patience: halt if validation metric does not
    improve for this many consecutive rounds."""

    VALIDATION_SPLIT: float = Field(default=0.15, gt=0.0, lt=1.0)
    """Fraction of data reserved for the validation set during training."""

    TEST_SPLIT: float = Field(default=0.15, gt=0.0, lt=1.0)
    """Fraction of data reserved as the held-out test set for final evaluation."""

    TIME_PERIOD_SECONDS: float = Field(default=86400.0, gt=0.0)
    """Denominator for cyclical time encoding (seconds in one 24-hour period)."""

    RANDOM_STATE: int = 42
    """Global random seed propagated to train/test splits and XGBoost."""

    # Asymmetric business cost parameters

    COST_FALSE_NEGATIVE: float = Field(default=100.0, gt=0.0)
    """Financial penalty (USD) for each missed fraud transaction (Type II error).
    Reflects actual fraud loss absorbed by the institution."""

    COST_FALSE_POSITIVE: float = Field(default=5.0, gt=0.0)
    """Financial penalty (USD) for each legitimate transaction incorrectly
    blocked (Type I error).  Reflects operational cost of manual review."""

    # Probability threshold grid for cost-matrix sweep

    THRESHOLD_GRID_START: float = Field(default=0.001, gt=0.0, lt=1.0)
    """Lower bound of the probability threshold grid searched by the
    CostMatrixEvaluator."""

    THRESHOLD_GRID_END: float = Field(default=0.999, gt=0.0, lt=1.0)
    """Upper bound of the probability threshold grid searched by the
    CostMatrixEvaluator."""

    THRESHOLD_GRID_STEPS: int = Field(default=1000, gt=1)
    """Number of evenly-spaced threshold candidates in the grid search."""

    # Cross-field validation

    @model_validator(mode="after")
    def _validate_cross_field_constraints(self) -> "PipelineConfig":
        """Catch configuration combinations that are individually valid but
        collectively nonsensical, so the pipeline fails fast at startup
        rather than producing silently wrong results mid-run.
        """
        if self.VALIDATION_SPLIT + self.TEST_SPLIT >= 1.0:
            raise ValueError(
                f"VALIDATION_SPLIT ({self.VALIDATION_SPLIT}) + TEST_SPLIT "
                f"({self.TEST_SPLIT}) must be < 1.0 to leave a non-empty "
                "training partition."
            )

        if self.THRESHOLD_GRID_START >= self.THRESHOLD_GRID_END:
            raise ValueError(
                f"THRESHOLD_GRID_START ({self.THRESHOLD_GRID_START}) must be "
                f"< THRESHOLD_GRID_END ({self.THRESHOLD_GRID_END})."
            )

        min_max_pairs: list[tuple[str, str]] = [
            ("XGB_MAX_DEPTH_MIN", "XGB_MAX_DEPTH_MAX"),
            ("XGB_LR_MIN", "XGB_LR_MAX"),
            ("XGB_SUBSAMPLE_MIN", "XGB_SUBSAMPLE_MAX"),
            ("XGB_COLSAMPLE_MIN", "XGB_COLSAMPLE_MAX"),
            ("XGB_GAMMA_MIN", "XGB_GAMMA_MAX"),
            ("XGB_REG_ALPHA_MIN", "XGB_REG_ALPHA_MAX"),
            ("XGB_REG_LAMBDA_MIN", "XGB_REG_LAMBDA_MAX"),
            ("XGB_N_ESTIMATORS_MIN", "XGB_N_ESTIMATORS_MAX"),
            ("XGB_MIN_CHILD_WEIGHT_MIN", "XGB_MIN_CHILD_WEIGHT_MAX"),
        ]
        for min_field, max_field in min_max_pairs:
            min_val = getattr(self, min_field)
            max_val = getattr(self, max_field)
            if min_val >= max_val:
                raise ValueError(
                    f"{min_field} ({min_val}) must be < {max_field} ({max_val})."
                )

        return self
