"""
Layer 5: Cost-Matrix Evaluator and Explainability Engine.

Provides two independent classes that operate on a trained XGBoost booster and
a held-out test partition:

    ``CostMatrixEvaluator``
        Sweeps a fine probability-threshold grid and selects the decision
        boundary that minimises total financial loss using asymmetric costs:
            - False Negative (missed fraud)  = $100 per transaction
            - False Positive (blocked legit) = $5  per transaction

    ``ExplainabilityEngine``
        Computes SHAP TreeExplainer values for the booster and produces a
        summary bar plot saved to ``config.SHAP_PLOT_PATH``.  The raw SHAP
        values matrix is also returned so callers can build custom
        visualisations or include them in regulatory reports.

Both classes accept a ``PipelineConfig`` at construction time so they share the
same cost parameters and file-system paths as every other pipeline layer.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from config import PipelineConfig

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ThresholdSearchResult:
    """Immutable record produced by ``CostMatrixEvaluator.find_optimal_threshold()``.

    Attributes:
        optimal_threshold: Probability cut-off that minimises total cost.
        minimum_total_cost: Total financial loss at ``optimal_threshold``.
        false_negative_count: Number of missed frauds at ``optimal_threshold``.
        false_positive_count: Number of blocked legitimate txns at that threshold.
        true_positive_count: Correctly identified frauds at that threshold.
        true_negative_count: Correctly passed legitimate txns at that threshold.
        precision: TP / (TP + FP) at ``optimal_threshold``; 0.0 if denominator
            is zero.
        recall: TP / (TP + FN) at ``optimal_threshold``; 0.0 if denominator is
            zero.
        f1_score: Harmonic mean of precision and recall; 0.0 if denominator is
            zero.
        threshold_grid: Full array of threshold candidates that were evaluated.
        cost_curve: Total cost at every threshold in ``threshold_grid``.
    """

    optimal_threshold: float
    minimum_total_cost: float
    false_negative_count: int
    false_positive_count: int
    true_positive_count: int
    true_negative_count: int
    precision: float
    recall: float
    f1_score: float
    threshold_grid: np.ndarray
    cost_curve: np.ndarray


@dataclass
class ExplainabilityResult:
    """Immutable record produced by ``ExplainabilityEngine.explain()``.

    Attributes:
        shap_values: 2-D array of shape ``(n_samples, n_features)`` containing
            the SHAP contribution of each feature for every test sample.
        feature_names: Ordered list of feature column names corresponding to
            axis-1 of ``shap_values``.
        plot_path: Absolute path where the SHAP summary PNG was saved.
        mean_abs_shap: Series mapping feature name → mean |SHAP| across all
            test samples, sorted descending.
    """

    shap_values: np.ndarray
    feature_names: list[str]
    plot_path: Path
    mean_abs_shap: pd.Series = field(default_factory=pd.Series)


# ---------------------------------------------------------------------------
# CostMatrixEvaluator
# ---------------------------------------------------------------------------


class CostMatrixEvaluator:
    """Threshold optimiser driven by asymmetric business costs.

    The evaluator iterates a uniform grid of probability thresholds between
    ``config.THRESHOLD_GRID_START`` and ``config.THRESHOLD_GRID_END``.  At
    each candidate threshold it classifies every test-set prediction, computes
    the confusion matrix, and calculates:

        total_cost = (FN_count × COST_FALSE_NEGATIVE)
                   + (FP_count × COST_FALSE_POSITIVE)

    The threshold that produces the lowest ``total_cost`` is returned as the
    production decision boundary.

    Attributes:
        config: Validated ``PipelineConfig`` supplying cost constants and grid
            resolution.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialise the evaluator with pipeline configuration.

        Args:
            config: Instantiated ``PipelineConfig`` object.
        """
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
        """Sweep the probability grid and return the cost-minimising threshold.

        At each threshold candidate the method:
            1. Converts continuous probabilities to binary predictions.
            2. Derives TP, TN, FP, FN from element-wise comparison with
               ``y_true``.
            3. Computes ``total_cost = FN × COST_FN + FP × COST_FP``.
        The threshold with the lowest total cost is selected.

        Args:
            y_true: Ground-truth binary labels (0 = legitimate, 1 = fraud).
                Accepts both ``pd.Series`` and ``np.ndarray``; internally
                converted to a 1-D int32 NumPy array.
            y_proba: Model output probabilities in [0, 1], shape (n_samples,).

        Returns:
            ``ThresholdSearchResult`` populated with the optimal threshold,
            its confusion-matrix statistics, derived classification metrics,
            and the full cost curve for downstream plotting.

        Raises:
            ValueError: If ``y_true`` and ``y_proba`` have different lengths or
                if ``y_proba`` contains values outside [0, 1].
            RuntimeError: If the grid search fails for any unexpected reason.
        """
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
                optimal_threshold,
                minimum_cost,
                tp,
                tn,
                fp,
                fn,
                precision,
                recall,
                f1,
            )

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
        """Compute confusion matrix and cost at a user-supplied threshold.

        Useful for auditing a threshold that was determined externally (e.g.
        from a previous run) without re-running the full grid search.

        Args:
            y_true: Ground-truth binary labels, shape (n_samples,).
            y_proba: Model output probabilities in [0, 1], shape (n_samples,).
            threshold: Decision boundary to apply; must be in (0, 1).

        Returns:
            Dictionary with keys: ``threshold``, ``tp``, ``tn``, ``fp``,
            ``fn``, ``total_cost``, ``precision``, ``recall``, ``f1_score``.

        Raises:
            ValueError: If ``threshold`` is not in (0, 1) or array lengths
                differ.
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

            result: dict[str, Any] = {
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
                tp,
                tn,
                fp,
                fn,
            )
            return result

        except ValueError:
            raise
        except Exception as exc:
            logger.exception("Unexpected error in evaluate_at_threshold.")
            raise RuntimeError(
                f"CostMatrixEvaluator.evaluate_at_threshold failed: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# ExplainabilityEngine
# ---------------------------------------------------------------------------


class ExplainabilityEngine:
    """SHAP-based feature importance calculator for regulatory explainability.

    Uses ``shap.TreeExplainer`` which operates directly on the internal tree
    structure of the XGBoost booster — no sampling or approximation is required
    for tree models.  The resulting SHAP values represent the exact Shapley
    contribution of each feature to every individual prediction.

    The summary bar plot shows mean absolute SHAP values aggregated across all
    provided test samples and is saved as a PNG to ``config.SHAP_PLOT_PATH``.

    Attributes:
        config: Validated ``PipelineConfig`` supplying the SHAP plot output
            path.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialise the engine with pipeline configuration.

        Args:
            config: Instantiated ``PipelineConfig`` object.
        """
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
        """Compute SHAP values and persist the feature-importance summary plot.

        Workflow:
            1. Instantiate ``shap.TreeExplainer(booster)`` — leverages the
               exact tree structure; no background dataset required.
            2. Call ``explainer.shap_values(X_test)`` — returns a 2-D array
               (n_samples, n_features) of per-sample Shapley contributions.
            3. Compute ``mean_abs_shap = |shap_values|.mean(axis=0)`` and rank
               features descending by importance.
            4. Render a horizontal bar chart via ``matplotlib`` and save to
               ``config.SHAP_PLOT_PATH``, creating parent directories if absent.

        Args:
            booster: Trained ``xgb.Booster`` whose trees are explained.
            X_test: Feature matrix used for explanation; must share the same
                column order as the training data seen by the booster.

        Returns:
            ``ExplainabilityResult`` containing the raw SHAP matrix, ordered
            feature names, absolute plot path, and the mean |SHAP| series.

        Raises:
            ValueError: If ``X_test`` is empty.
            RuntimeError: Wraps any ``shap`` or ``matplotlib`` error.
        """
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
        """Render a horizontal bar chart of mean absolute SHAP values and save.

        Displays up to 20 features (most important first) on the y-axis with
        mean |SHAP| on the x-axis.  The figure is saved as a PNG at
        ``config.SHAP_PLOT_PATH``; parent directories are created automatically.

        Args:
            feature_names: Feature names sorted by descending mean |SHAP|.
            mean_abs_shap: Corresponding mean absolute SHAP values (same
                order as ``feature_names``).

        Raises:
            RuntimeError: If the plot cannot be saved (e.g. permission error).
        """
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
