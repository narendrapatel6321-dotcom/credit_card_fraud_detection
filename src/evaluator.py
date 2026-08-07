"""
Layer 5: Evaluation and Explainability.

Provides two components for post-training analysis:

- CostMatrixEvaluator: Finds the probability threshold that minimises
  business cost using an asymmetric cost matrix.
- ExplainabilityEngine: Generates SHAP-based feature importance and
  explanation artefacts for the trained XGBoost model.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.config import PipelineConfig

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ThresholdSearchResult:
    """
    Results returned by CostMatrixEvaluator.find_optimal_threshold().

    Stores the optimal decision threshold, confusion matrix statistics,
    classification metrics, and the full threshold-cost curve.
    """
    optimal_threshold: float
    minimum_total_cost: float
    
    true_positive_count: int
    true_negative_count: int
    false_positive_count: int
    false_negative_count: int
    
    precision: float
    recall: float
    f1_score: float
    
    threshold_grid: np.ndarray
    cost_curve: np.ndarray

@dataclass
class ExplainabilityResult:
    """ Results returned by ExplainabilityEngine.explain().
    Contains the computed SHAP values, feature names, generated plot location, and aggregated feature importance scores.
    """
    shap_values: np.ndarray
    feature_names: list[str]
    plot_path: Path
    mean_abs_shap: pd.Series = field(default_factory=pd.Series)

# CostMatrixEvaluator

class CostMatrixEvaluator:
    """ Optimise the classification threshold using the configured business cost matrix. """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialise the evaluator."""
        self.config: PipelineConfig = config
        logger.info(
            "CostMatrixEvaluator initialised. FN cost: $%.0f | FP cost: $%.0f | "
            "Grid: %.2f–%.2f (%d steps)",
            config.COST_FALSE_NEGATIVE,
            config.COST_FALSE_POSITIVE,
            config.THRESHOLD_GRID_START,
            config.THRESHOLD_GRID_END,
            config.THRESHOLD_GRID_STEPS,
        )

    def find_optimal_threshold(
        self,
        y_true: pd.Series | np.ndarray,
        y_proba: np.ndarray,
    ) -> ThresholdSearchResult:
        """Find the probability threshold that minimises total business cost.
        Evaluates the configured threshold grid using the supplied fraud
        probabilities and returns the threshold with the lowest overall cost.

        Args:
            y_true: Ground-truth binary labels.
            y_proba: Predicted fraud probabilities.
        Returns:
            ThresholdSearchResult containing the optimal threshold,
            evaluation metrics, and the complete threshold-cost curve.
        Raises:
            ValueError: If the inputs are invalid.
            RuntimeError: If threshold optimisation fails unexpectedly."""
        
        try:
            labels: np.ndarray = np.asarray(y_true, dtype=np.int32).ravel()
            proba: np.ndarray = np.asarray(y_proba, dtype=np.float32).ravel()

            if labels.shape[0] != proba.shape[0]:
                raise ValueError(
                    f"y_true length {labels.shape[0]} != y_proba length "
                    f"{proba.shape[0]}."
                )
            if float(proba.min()) < 0.0 or float(proba.max()) > 1.0:
                raise ValueError(
                    "y_proba contains values outside [0, 1]. Ensure the model "
                    "output is a probability."
                )

            threshold_grid: np.ndarray = np.linspace(
                self.config.THRESHOLD_GRID_START,
                self.config.THRESHOLD_GRID_END,
                self.config.THRESHOLD_GRID_STEPS,
                dtype=np.float64,
            )

            fn_costs: float = float(self.config.COST_FALSE_NEGATIVE)
            fp_costs: float = float(self.config.COST_FALSE_POSITIVE)

            # Vectorized sweep: broadcast thresholds (n_thresholds, 1) against
            # probabilities (1, n_samples) to produce a binary prediction matrix
            # of shape (n_thresholds, n_samples) in a single numpy operation.
            # This replaces a Python loop and is ~100× faster on large datasets.
            preds_matrix: np.ndarray = (
                proba[np.newaxis, :] >= threshold_grid[:, np.newaxis]
            ).astype(np.int8)

            labels_row: np.ndarray = labels[np.newaxis, :]
            fn_counts: np.ndarray = np.sum(
                (preds_matrix == 0) & (labels_row == 1), axis=1
            ).astype(np.float64)
            fp_counts: np.ndarray = np.sum(
                (preds_matrix == 1) & (labels_row == 0), axis=1
            ).astype(np.float64)
            cost_curve: np.ndarray = fn_counts * fn_costs + fp_counts * fp_costs

            best_idx: int = int(np.argmin(cost_curve))
            optimal_threshold: float = float(threshold_grid[best_idx])
            minimum_cost: float = float(cost_curve[best_idx])

            # Recompute full confusion matrix at the optimal threshold
            best_preds: np.ndarray = (proba >= optimal_threshold).astype(np.int32)
            tp: int = int(np.sum((best_preds == 1) & (labels == 1)))
            tn: int = int(np.sum((best_preds == 0) & (labels == 0)))
            fp: int = int(np.sum((best_preds == 1) & (labels == 0)))
            fn: int = int(np.sum((best_preds == 0) & (labels == 1)))

            precision: float = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall: float = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1: float = (
                2.0 * precision * recall / (precision + recall)
                if (precision + recall) > 0.0
                else 0.0
            )

            logger.info(
                "Threshold search complete.\n"
                "  Optimal threshold : %.4f\n"
                "  Minimum cost      : $%.2f\n"
                "  TP=%d  TN=%d  FP=%d  FN=%d\n"
                "  Precision=%.4f  Recall=%.4f  F1=%.4f",
                optimal_threshold, minimum_cost, 
                tp, tn, fp, fn,
                precision, recall, f1, )

            return ThresholdSearchResult(
                optimal_threshold=optimal_threshold,
                minimum_total_cost=minimum_cost,
                false_negative_count=fn,
                false_positive_count=fp,
                true_positive_count=tp,
                true_negative_count=tn,
                precision=precision,
                recall=recall,
                f1_score=f1,
                threshold_grid=threshold_grid,
                cost_curve=cost_curve,
            )

        except (ValueError, RuntimeError):
            raise
        except Exception as exc:
            logger.exception("Unexpected error during threshold grid search.")
            raise RuntimeError(
                f"CostMatrixEvaluator.find_optimal_threshold failed: {exc}"
            ) from exc

    def evaluate_at_threshold(
        self,
        y_true: pd.Series | np.ndarray,
        y_proba: np.ndarray,
        threshold: float,
    ) -> dict[str, Any]:
        """Evaluate model performance at a specified probability threshold.

        Args:
            y_true: Ground-truth binary labels.
            y_proba: Predicted fraud probabilities.
            threshold: Decision threshold to evaluate.
        Returns:
            Dictionary containing the confusion matrix, business cost,
            and classification metrics at the specified threshold.
        Raises:
            ValueError: If the threshold or inputs are invalid.
            RuntimeError: If evaluation fails unexpectedly.
        """
        try:
            if not (0.0 < threshold < 1.0):
                raise ValueError(f"threshold must be in (0, 1), got {threshold}.")

            labels: np.ndarray = np.asarray(y_true, dtype=np.int32).ravel()
            proba: np.ndarray = np.asarray(y_proba, dtype=np.float32).ravel()

            if labels.shape[0] != proba.shape[0]:
                raise ValueError(
                    f"y_true length {labels.shape[0]} != y_proba length "
                    f"{proba.shape[0]}."
                )

            preds: np.ndarray = (proba >= threshold).astype(np.int32)
            tp: int = int(np.sum((preds == 1) & (labels == 1)))
            tn: int = int(np.sum((preds == 0) & (labels == 0)))
            fp: int = int(np.sum((preds == 1) & (labels == 0)))
            fn: int = int(np.sum((preds == 0) & (labels == 1)))

            total_cost: float = fn * float(
                self.config.COST_FALSE_NEGATIVE
            ) + fp * float(self.config.COST_FALSE_POSITIVE)
            precision: float = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall: float = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1: float = (
                2.0 * precision * recall / (precision + recall)
                if (precision + recall) > 0.0
                else 0.0
            )

            metrics: dict[str, Any] = {
                "threshold": threshold,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "total_cost": total_cost,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }

            logger.info(
                "Evaluation at threshold=%.4f: cost=$%.2f " "TP=%d TN=%d FP=%d FN=%d",
                threshold,
                total_cost,
                tp, tn, fp, fn,
            )
            return metrics

        except ValueError:
            raise
        except Exception as exc:
            logger.exception("Unexpected error in evaluate_at_threshold.")
            raise RuntimeError(
                f"CostMatrixEvaluator.evaluate_at_threshold failed: {exc}"
            ) from exc

# ExplainabilityEngine

class ExplainabilityEngine:
    """ Generate SHAP-based explanations for a trained XGBoost model. """
    
    def __init__(self, config: PipelineConfig) -> None:
        """ Initialise the explainability engine. """
        self.config: PipelineConfig = config
        logger.info(
            "ExplainabilityEngine initialised. SHAP plot path: %s",
            config.SHAP_PLOT_PATH,
        )

    def explain(
        self,
        booster: xgb.Booster,
        X_test: pd.DataFrame,
    ) -> ExplainabilityResult:
        """
        Compute SHAP values and generate the feature-importance summary plot.

        Args:
            booster: Trained XGBoost booster.
            X_test: Feature matrix used for explanation.

        Returns:
            ExplainabilityResult containing the SHAP values, feature names,
            summary plot path, and mean absolute SHAP values.

        Raises:
            ValueError: If X_test is empty.
            RuntimeError: If SHAP computation fails unexpectedly. """
        try:
            if X_test.empty:
                raise ValueError(
                    "X_test must contain at least one row for SHAP computation."
                )

            feature_names: list[str] = list(X_test.columns)
            logger.info(
                "Computing SHAP values for %d samples × %d features.",
                len(X_test),
                len(feature_names),
            )

            explainer: shap.TreeExplainer = shap.TreeExplainer(booster)

            X_array: np.ndarray = X_test.to_numpy(dtype=np.float32)
            shap_values: np.ndarray = np.asarray(
                explainer.shap_values(X_array), dtype=np.float64
            )

            # shap_values may be 3-D (n_samples, n_features, n_classes) for
            # multi-output models; for binary classification XGBoost returns
            # (n_samples, n_features) directly — guard against the edge case.
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

            mean_abs: np.ndarray = np.abs(shap_values).mean(axis=0)
            sort_order: np.ndarray = np.argsort(mean_abs)[::-1]
            sorted_names: list[str] = [feature_names[i] for i in sort_order]
            sorted_values: np.ndarray = mean_abs[sort_order]

            mean_abs_shap: pd.Series = pd.Series(
                data=sorted_values, index=sorted_names, name="mean_abs_shap"
            )

            self._save_summary_plot(sorted_names, sorted_values)

            logger.info(
                "SHAP computation complete. Top-5 features by |SHAP|:\n%s",
                mean_abs_shap.head(5).to_string(),
            )

            return ExplainabilityResult(
                shap_values=shap_values,
                feature_names=feature_names,
                plot_path=self.config.SHAP_PLOT_PATH,
                mean_abs_shap=mean_abs_shap,
            )

        except ValueError:
            raise
        except Exception as exc:
            logger.exception("Unexpected error during SHAP computation.")
            raise RuntimeError(f"ExplainabilityEngine.explain failed: {exc}") from exc

    def _save_summary_plot(
        self,
        feature_names: list[str],
        mean_abs_shap: np.ndarray,
    ) -> None:
        """
        Save a horizontal bar chart of mean absolute SHAP values.

        Args:
            feature_names: Feature names sorted by importance.
            mean_abs_shap: Mean absolute SHAP values for each feature.

        Raises:
            RuntimeError: If the plot cannot be saved. """
        try:
            max_features: int = 20
            display_names: list[str] = feature_names[:max_features]
            display_values: np.ndarray = mean_abs_shap[:max_features]

            fig_height: float = max(6.0, len(display_names) * 0.45)
            fig, ax = plt.subplots(figsize=(10.0, fig_height))

            y_pos: np.ndarray = np.arange(len(display_names), dtype=np.float64)
            ax.barh(y_pos, display_values[::-1], align="center", color="#1f77b4")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(display_names[::-1], fontsize=9)
            ax.set_xlabel("Mean |SHAP value|", fontsize=10)
            ax.set_title(
                "Feature Importance — Mean Absolute SHAP (Test Set)", fontsize=11
            )
            ax.invert_xaxis()
            fig.tight_layout()

            plot_path: Path = self.config.SHAP_PLOT_PATH
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
            plt.close(fig)

            logger.info("SHAP summary plot saved to: %s", plot_path)

        except Exception as exc:
            logger.exception("Failed to save SHAP summary plot.")
            raise RuntimeError(
                f"ExplainabilityEngine._save_summary_plot failed: {exc}"
            ) from exc
