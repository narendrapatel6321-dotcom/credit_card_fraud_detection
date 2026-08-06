""" Layer 4: Model Training.

Provides FraudModelTrainer, responsible for:

- Stratified train/validation/test splitting
- Optuna hyperparameter optimization
- Final XGBoost training
- Model persistence
- Probability prediction """

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from config import PipelineConfig

logger: logging.Logger = logging.getLogger(__name__)


class _OptunaPruningCallback(xgb.callback.TrainingCallback):
     """Reports validation PR-AUC to Optuna for trial pruning."""
    def __init__(
        self,
        trial: optuna.Trial,
        metric_name: str,
        eval_set_name: str = "val",
    ) -> None:
        """Initialise the callback for a single Optuna trial."""
        
        super().__init__()
        self._trial: optuna.Trial = trial
        self._metric_name: str = metric_name
        self._eval_set_name: str = eval_set_name

    def after_iteration(
        self,
        model: xgb.Booster,
        epoch: int,
        evals_log: dict[str, dict[str, list[float] | list[tuple[float, float]]]],
    ) -> bool:
        """Report the latest validation score and prune if requested."""
        
        set_log = evals_log.get(self._eval_set_name, {})
        raw_scores = set_log.get(self._metric_name, [])
        if raw_scores:
            last_score = raw_scores[-1]
            score_value: float = (
                float(last_score[0])
                if isinstance(last_score, tuple)
                else float(last_score)
            )
            self._trial.report(score_value, step=epoch)
            if self._trial.should_prune():
                raise optuna.TrialPruned()
        return False


@dataclass
class TrainTestSplit:
    """ Container for train, validation and test splits. """
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


class FraudModelTrainer:
    """ Handles data splitting, hyperparameter optimization, training, model saving and inference. """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialise the trainer in an unoptimised state.

        Args:
            config: Instantiated ``PipelineConfig`` object.
        """
        self.config: PipelineConfig = config
        self.best_params_: dict[str, Any] | None = None
        self.best_val_aucpr_: float | None = None
        self.best_n_estimators_: int | None = None
        logger.info("FraudModelTrainer initialised.")

    def _create_dmatrices(
    self,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[xgb.DMatrix, xgb.DMatrix]:
    """Create training and validation DMatrices."""

    feature_names = X_train.columns.tolist()

    dtrain: xgb.DMatrix = xgb.DMatrix(
                X_train.values.astype(np.float32),
                label=y_train.values.astype(np.int32),
                feature_names=feature_names,
            )
    dval: xgb.DMatrix = xgb.DMatrix(
                X_val.values.astype(np.float32),
                label=y_val.values.astype(np.int32),
                feature_names=feature_names,
            )

    return dtrain, dval
    
    def split(self, df: pd.DataFrame) -> TrainTestSplit:
        """
        Split the dataset into stratified train,
        validation and test partitions.
        Args:
            df: Engineered DataFrame produced by FraudFeatureTransformer.
                Must contain a Class column (int32, 0 / 1).

        Returns:
            TrainTestSplit dataclass with six fields: X_train, X_val,
            X_test, y_train, y_val, y_test.

        Raises:
            KeyError: If the Class column is absent from df.
            ValueError: If stratification fails due to too few positive samples
                in one of the partitions.
            RuntimeError: On any unexpected splitting error.
        """
        try:
            if "Class" not in df.columns:
                raise KeyError(
                    "'Class' column not found. Available columns: %s"
                    % df.columns.tolist()
                )

            y: pd.Series = df["Class"]
            X: pd.DataFrame = df.drop(columns=["Class"])

            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X,
                y,
                test_size=self.config.TEST_SPLIT,
                random_state=self.config.RANDOM_STATE,
                stratify=y,
            )

            val_fraction: float = self.config.VALIDATION_SPLIT / (
                1.0 - self.config.TEST_SPLIT
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval,
                y_trainval,
                test_size=val_fraction,
                random_state=self.config.RANDOM_STATE,
                stratify=y_trainval,
            )

            logger.info(
                "Data split complete.\n"
                "  Train : %d rows | %d fraud (%.4f%%)\n"
                "  Val   : %d rows | %d fraud (%.4f%%)\n"
                "  Test  : %d rows | %d fraud (%.4f%%)",
                len(X_train),
                int(y_train.sum()),
                float(y_train.mean()) * 100,
                len(X_val),
                int(y_val.sum()),
                float(y_val.mean()) * 100,
                len(X_test),
                int(y_test.sum()),
                float(y_test.mean()) * 100,
            )
            return TrainTestSplit(
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
            )

        except KeyError:
            logger.error("'Class' column missing in split().")
            raise
        except ValueError as exc:
            logger.error(
                "Stratified split failed — likely too few positive samples "
                "in a partition: %s",
                exc,
            )
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error in FraudModelTrainer.split(): %s",
                exc,
                exc_info=True,
            )
            raise RuntimeError(
                "FraudModelTrainer.split() failed with an unexpected error."
            ) from exc


    def _build_objective(
        self,
        dtrain: xgb.DMatrix,
        dval: xgb.DMatrix,
        scale_pos_weight: float,
    ) -> Callable[[optuna.Trial], float]:
        """ Create the Optuna objective function used during hyperparameter optimization. """
        
        def objective(trial: optuna.Trial) -> float:
            params: dict[str, Any] = {
                "objective": "binary:logistic",
                "eval_metric": ["aucpr", "logloss"],
                "max_depth": trial.suggest_int(
                    "max_depth",
                    self.config.XGB_MAX_DEPTH_MIN,
                    self.config.XGB_MAX_DEPTH_MAX,
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    self.config.XGB_LR_MIN,
                    self.config.XGB_LR_MAX,
                    log=True,
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    self.config.XGB_SUBSAMPLE_MIN,
                    self.config.XGB_SUBSAMPLE_MAX,
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    self.config.XGB_COLSAMPLE_MIN,
                    self.config.XGB_COLSAMPLE_MAX,
                ),
                "gamma": trial.suggest_float(
                    "gamma",
                    self.config.XGB_GAMMA_MIN,
                    self.config.XGB_GAMMA_MAX,
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha",
                    self.config.XGB_REG_ALPHA_MIN,
                    self.config.XGB_REG_ALPHA_MAX,
                    log=True,
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    self.config.XGB_REG_LAMBDA_MIN,
                    self.config.XGB_REG_LAMBDA_MAX,
                    log=True,
                ),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight",
                    self.config.XGB_MIN_CHILD_WEIGHT_MIN,
                    self.config.XGB_MIN_CHILD_WEIGHT_MAX,
                ),
                "scale_pos_weight": scale_pos_weight,
                "tree_method": "hist",
                "seed": self.config.RANDOM_STATE,
                "verbosity": 0,
                "nthread": -1,
            }

            evals_result: dict[str, dict[str, list[float]]] = {}
            callbacks: list[Any] = [
                xgb.callback.EarlyStopping(
                    rounds=self.config.EARLY_STOPPING_ROUNDS,
                    metric_name="aucpr",
                    maximize=True,
                    save_best=True,
                ),
                _OptunaPruningCallback(trial, metric_name="aucpr"),
            ]

            xgb.train(
                params,
                dtrain,
                num_boost_round=self.config.XGB_N_ESTIMATORS_MAX,
                evals=[(dtrain, "train"), (dval, "val")],
                callbacks=callbacks,
                evals_result=evals_result,
            )

            val_aucpr_list: list[float] = evals_result["val"]["aucpr"]
            val_logloss_list: list[float] = evals_result["val"]["logloss"]
            best_iteration: int = int(np.argmax(val_aucpr_list))
            val_aucpr: float = float(val_aucpr_list[best_iteration])
            val_logloss: float = float(val_logloss_list[best_iteration])

            trial.set_user_attr("best_iteration", best_iteration)
            trial.set_user_attr("val_logloss", val_logloss)

            logger.debug(
                "Trial %d: val_aucpr=%.4f | val_logloss=%.6f | best_iter=%d",
                trial.number,
                val_aucpr,
                val_logloss,
                best_iteration,
            )
            return val_aucpr

        return objective


    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        scale_pos_weight: float,
    ) -> dict[str, Any]:
        """Run Optuna TPE search over XGBoost hyperparameter bounds.

        Builds DMatrix objects once, then dispatches ``OPTUNA_N_TRIALS`` trial
        evaluations.  Each trial suggests hyperparameters, trains XGBoost with
        early stopping on validation PR-AUC, and reports the best score back to
        Optuna.  After the study completes, best params and statistics are
        stored as instance attributes.

        Args:
            X_train: Feature matrix for the training partition (post-engineering).
            y_train: Binary target vector for training.
            X_val: Feature matrix for the validation partition.
            y_val: Binary target vector for validation.
            scale_pos_weight: Negative-to-positive class ratio.

        Returns:
            Dict of best tuned XGBoost hyperparameters (max_depth, learning_rate,
            subsample, colsample_bytree, gamma, reg_alpha, reg_lambda).

        Raises:
            RuntimeError: On any unrecoverable error during the Optuna study.
        """
        try:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            logger.info(
            "Starting Optuna hyperparameter optimization (%d trials).",
            self.config.OPTUNA_N_TRIALS )
            
            dtrain, dval = self._create_dmatrices(X_train,y_train,X_val,y_val)

            objective: Callable[[optuna.Trial], float] = self._build_objective(
                dtrain, dval, scale_pos_weight
            )

            study: optuna.Study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.config.RANDOM_STATE),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=20,
                ),
            )
            study.optimize(
                objective,
                n_trials=self.config.OPTUNA_N_TRIALS,
                show_progress_bar=False,
            )

            best_trial: optuna.trial.FrozenTrial = study.best_trial
            self.best_params_ = dict(best_trial.params)
            self.best_val_aucpr_ = float(best_trial.value or 0.0)
            self.best_n_estimators_ = (
                int(
                    best_trial.user_attrs.get(
                        "best_iteration",
                        self.config.XGB_N_ESTIMATORS_MAX - 1,
                    )
                )
                + 1
            )

            logger.info(
                "Optuna search complete.\n"
                "  Best val PR-AUC  : %.4f\n"
                "  Best val LogLoss : %.6f\n"
                "  Best n_estimators: %d\n"
                "  Best params      : %s",
                self.best_val_aucpr_,
                float(best_trial.user_attrs.get("val_logloss", float("nan"))),
                self.best_n_estimators_,
                self.best_params_,
            )
            return dict(self.best_params_)

        except Exception as exc:
            logger.error(
                "Unexpected error in FraudModelTrainer.optimize(): %s",
                exc,
                exc_info=True,
            )
            raise RuntimeError(
                "FraudModelTrainer.optimize() failed with an unexpected error."
            ) from exc


    def train_final(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        scale_pos_weight: float,
    ) -> xgb.Booster:
        """ Train the final XGBoost model using the best hyperparameters found by Optuna. """
        try:
            if self.best_params_ is None:
                raise RuntimeError(
                    "train_final() called before optimize(). "
                    "Call optimize() first to obtain best hyperparameters."
                )
                
            params: dict[str, Any] = {
                **self.best_params_,
                "objective": "binary:logistic",
                "eval_metric": ["aucpr", "logloss"],
                "scale_pos_weight": scale_pos_weight,
                "tree_method": "hist",
                "seed": self.config.RANDOM_STATE,
                "verbosity": 0,
                "nthread": -1,
            }

            dtrain, dval = self._create_dmatrices(X_train,y_train,X_val,y_val)

            evals_result: dict[str, dict[str, list[float]]] = {}
            callbacks: list[Any] = [
                xgb.callback.EarlyStopping(
                    rounds=self.config.EARLY_STOPPING_ROUNDS,
                    metric_name="aucpr",
                    maximize=True,
                    save_best=True,
                )
            ]

            logger.info(
                "Training final production model.\n"
                "  Max rounds      : %d\n"
                "  Early stopping  : %d rounds\n"
                "  Tuned params    : %s",
                self.config.XGB_N_ESTIMATORS_MAX,
                self.config.EARLY_STOPPING_ROUNDS,
                self.best_params_,
            )

            booster: xgb.Booster = xgb.train(
                params,
                dtrain,
                num_boost_round=self.config.XGB_N_ESTIMATORS_MAX,
                evals=[(dtrain, "train"), (dval, "val")],
                callbacks=callbacks,
                evals_result=evals_result,
            )

            val_aucpr_list: list[float] = evals_result["val"]["aucpr"]
            val_logloss_list: list[float] = evals_result["val"]["logloss"]
            best_iteration: int = int(np.argmax(val_aucpr_list))
            best_val_aucpr: float = float(val_aucpr_list[best_iteration])
            best_val_logloss: float = float(val_logloss_list[best_iteration])

            logger.info(
                "Final model training complete.\n"
                "  Best iteration  : %d\n"
                "  Val PR-AUC      : %.4f\n"
                "  Val LogLoss     : %.6f",
                best_iteration,
                best_val_aucpr,
                best_val_logloss,
            )
            return booster

        except RuntimeError:
            logger.error("train_final() called before optimize().")
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error in FraudModelTrainer.train_final(): %s",
                exc,
                exc_info=True,
            )
            raise RuntimeError(
                "FraudModelTrainer.train_final() failed with an unexpected error."
            ) from exc


    def save_model(self, booster: xgb.Booster) -> None:
        """Save the trained XGBoost model."""
        try:
            self.config.MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(str(self.config.MODEL_SAVE_PATH))
            logger.info("Model serialised to: %s", self.config.MODEL_SAVE_PATH)
        except Exception as exc:
            logger.error(
                "Failed to save model to %s: %s",
                self.config.MODEL_SAVE_PATH,
                exc,
                exc_info=True,
            )
            raise RuntimeError(
                "FraudModelTrainer.save_model() failed with an unexpected error."
            ) from exc

    def predict_proba(
        self,
        booster: xgb.Booster,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray: 
        """ Predict fraud probabilities for the given feature matrix."""
        try:
            feature_names: list[str] | None = (
                X.columns.tolist() if isinstance(X, pd.DataFrame) else None
            )
            data: np.ndarray = (
                X.values.astype(np.float32)
                if isinstance(X, pd.DataFrame)
                else np.asarray(X, dtype=np.float32)
            )
            dmat: xgb.DMatrix = xgb.DMatrix(data, feature_names=feature_names)
            probabilities: np.ndarray = booster.predict(dmat)

            logger.debug(
                "predict_proba: %d samples | range [%.4f, %.4f] | mean %.4f",
                len(probabilities),
                float(probabilities.min()),
                float(probabilities.max()),
                float(probabilities.mean()),
            )
            return probabilities

        except Exception as exc:
            logger.error(
                "Unexpected error in FraudModelTrainer.predict_proba(): %s",
                exc,
                exc_info=True,
            )
            raise RuntimeError(
                "FraudModelTrainer.predict_proba() failed with an unexpected error."
            ) from exc
