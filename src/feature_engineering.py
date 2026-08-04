"""
Layer 3: Advanced Feature Transformation.

Exposes ``FraudFeatureTransformer``, a stateful sklearn-style transformer that
prepares the raw cleaned DataFrame for XGBoost training.  Five transformations
are applied:

    1. **Cyclical Time Encoding** — The raw ``Time`` column (elapsed seconds)
       is projected onto a unit circle using sine and cosine at frequency
       ``1 / TIME_PERIOD_SECONDS``.  This preserves the periodic structure of
       daily transaction rhythms without imposing an arbitrary linear ordering.
       Formula:
           Time_sin = sin(2π × Time / TIME_PERIOD_SECONDS)
           Time_cos = cos(2π × Time / TIME_PERIOD_SECONDS)
       The original ``Time`` column is dropped after encoding.

    2. **Hour-of-Day Bucket** — An integer ``Time_hour`` (0–23) is derived from
       ``Time % 86400 // 3600``.  Cyclical encoding is smooth but XGBoost cannot
       split on periodicity; a discrete hour bucket lets the model learn clean
       "2 AM vs 2 PM" boundaries directly.

    3. **Amount Outlier Clipping** — The 99.9th-percentile of ``Amount`` is
       learned from the training partition and used to clip extreme outlier
       transactions before scaling.  This prevents a handful of very large
       legitimate transactions from collapsing the IQR scale for the majority
       of sub-$200 transactions.

    4. **Robust Scale Normalisation** — The heavily right-skewed ``Amount``
       column is normalised using median and IQR learned exclusively from the
       training split to prevent data leakage:
           Amount_scaled = (clip(Amount, upper=99.9p) − median) / IQR
       The original ``Amount`` column is dropped after scaling.

    5. **Log-Amount Feature** — ``Amount_log1p = log(1 + Amount)`` computed on
       the raw (pre-clip) value.  Provides the model with a complementary
       magnitude signal that naturally compresses extreme outliers.

    6. **Round-Amount Flag** — ``Amount_is_round = 1`` when ``Amount`` is a
       whole number.  Fraud transactions disproportionately use round values
       (ATM test charges, exact amounts); this boolean feature costs nothing and
       is directly interpretable.

The transformer follows a strict fit → transform split.  ``fit()`` learns
``amount_median_``, ``amount_iqr_``, and ``amount_clip_upper_`` from the
provided DataFrame; all subsequent ``transform()`` calls apply those stored
statistics.  Calling ``transform()`` before ``fit()`` raises ``RuntimeError``.

Fitted statistics can be persisted to / loaded from a JSON artefact via
``save_state()`` / ``load_state()``, enabling the inference API to reconstruct
a fitted transformer without access to the training data.
"""

import json
import logging
import math
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

from config import PipelineConfig

logger: logging.Logger = logging.getLogger(__name__)


class FraudFeatureTransformer:
    """Stateful feature transformer for the fraud detection pipeline.

    Encapsulates cyclical time encoding, hour-of-day bucketing, Amount outlier
    clipping, robust Amount normalisation, log-Amount, and a round-amount
    boolean flag.  Follows a scikit-learn-style fit / transform API so the
    transformer is fitted on the training split only, preventing label-leakage
    from the validation or test partitions.

    Attributes:
        config: Validated ``PipelineConfig`` supplying ``TIME_PERIOD_SECONDS``.
        amount_median_: Median of ``Amount`` learned during ``fit()``.
        amount_iqr_: Interquartile range (Q75 − Q25) of ``Amount`` learned
            during ``fit()``.
        amount_clip_upper_: 99.9th-percentile of ``Amount`` learned during
            ``fit()``.  Used to clip extreme outlier transactions before
            robust scaling.  Defaults to ``inf`` (no clipping) until fitted.
        is_fitted_: Flag set to ``True`` after a successful ``fit()`` call.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialise the transformer in an unfitted state.

        Args:
            config: Instantiated ``PipelineConfig`` object.  Only
                ``TIME_PERIOD_SECONDS`` is consumed by this layer.
        """
        self.config: PipelineConfig = config
        self.amount_median_: float = 0.0
        self.amount_iqr_: float = 1.0
        self.amount_clip_upper_: float = float("inf")
        self.is_fitted_: bool = False
        logger.info(
            "FraudFeatureTransformer initialised. Time period: %.0f seconds.",
            config.TIME_PERIOD_SECONDS,
        )

    def fit(self, df: pd.DataFrame) -> Self:
        """Learn robust scaling statistics and clipping bound from the Amount column.

        Computes and stores the median, IQR, and 99.9th-percentile of ``Amount``
        from the supplied DataFrame.  Should be called only on the training
        partition.

        Args:
            df: DataFrame containing at least an ``Amount`` column.  The
                ``Class`` column may be present; it is ignored here.

        Returns:
            ``self``, enabling method chaining (``transformer.fit(df).transform(df)``).

        Raises:
            KeyError: If ``Amount`` column is absent from ``df``.
            ValueError: If the computed IQR is zero (degenerate distribution).
        """
        try:
            if "Amount" not in df.columns:
                raise KeyError(
                    "'Amount' column not found. Available columns: %s"
                    % df.columns.tolist()
                )

            amount_series: pd.Series = df["Amount"].astype(np.float64)
            self.amount_median_ = float(amount_series.median())
            q25: float = float(amount_series.quantile(0.25))
            q75: float = float(amount_series.quantile(0.75))
            self.amount_iqr_ = q75 - q25
            self.amount_clip_upper_ = float(amount_series.quantile(0.999))

            if self.amount_iqr_ == 0.0:
                raise ValueError(
                    "IQR of 'Amount' is zero — cannot perform robust scaling. "
                    "Check for a constant or near-constant Amount distribution."
                )

            self.is_fitted_ = True
            logger.info(
                "FraudFeatureTransformer fitted.\n"
                "  Amount median     : %.6f\n"
                "  Amount Q25        : %.6f\n"
                "  Amount Q75        : %.6f\n"
                "  Amount IQR        : %.6f\n"
                "  Amount 99.9th pct : %.6f",
                self.amount_median_,
                q25,
                q75,
                self.amount_iqr_,
                self.amount_clip_upper_,
            )
            return self

        except KeyError:
            logger.error("'Amount' column missing during fit().")
            raise
        except ValueError as exc:
            logger.error("Degenerate Amount distribution in fit(): %s", exc)
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error in FraudFeatureTransformer.fit(): %s",
                exc,
                exc_info=True,
            )
            raise RuntimeError(
                "FraudFeatureTransformer.fit() failed with an unexpected error."
            ) from exc

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transformations using fitted statistics.

        Requires the transformer to have been fitted first via ``fit()``.
        Operates on a copy of ``df`` so the caller's DataFrame is not mutated.

        Column changes:
            - ``Time``   → dropped; ``Time_sin``, ``Time_cos``, ``Time_hour``
              added (float32, float32, int32).
            - ``Amount`` → dropped; ``Amount_scaled`` (float32),
              ``Amount_log1p`` (float32), ``Amount_is_round`` (int32) added.

        Args:
            df: DataFrame containing ``Time`` and ``Amount`` columns (plus any
                PCA components V1–V28 and optional ``Class`` column).

        Returns:
            Transformed DataFrame with ``Time`` and ``Amount`` replaced by
            their engineered counterparts.

        Raises:
            RuntimeError: If called before ``fit()``.
            KeyError: If ``Time`` or ``Amount`` columns are absent.
        """
        try:
            if not self.is_fitted_:
                raise RuntimeError(
                    "FraudFeatureTransformer.transform() called before fit(). "
                    "Call fit() on the training partition first."
                )

            for required_col in ("Time", "Amount"):
                if required_col not in df.columns:
                    raise KeyError(
                        "Required column '%s' not found. Available: %s"
                        % (required_col, df.columns.tolist())
                    )

            out: pd.DataFrame = df.copy()
            out = self._encode_cyclical_time(out)
            out = self._engineer_amount(out)

            logger.info(
                "FraudFeatureTransformer transform complete. "
                "Output shape: %s | Columns: %s",
                out.shape,
                out.columns.tolist(),
            )
            return out

        except RuntimeError:
            logger.error("transform() called on unfitted FraudFeatureTransformer.")
            raise
        except KeyError as exc:
            logger.error("Column missing during transform(): %s", exc)
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error in FraudFeatureTransformer.transform(): %s",
                exc,
                exc_info=True,
            )
            raise RuntimeError(
                "FraudFeatureTransformer.transform() failed with an unexpected error."
            ) from exc

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``df`` and immediately return the transformed result.

        Convenience wrapper equivalent to ``fit(df).transform(df)``.  Use this
        on the training partition only; call ``transform()`` standalone on
        validation and test partitions.

        Args:
            df: Training-partition DataFrame with ``Time``, ``Amount``, and
                PCA component columns.

        Returns:
            Transformed DataFrame as returned by ``transform()``.
        """
        logger.info("fit_transform() called — fitting then transforming in one pass.")
        return self.fit(df).transform(df)

    def save_state(self, path: Path) -> None:
        """Persist fitted transformer statistics to a JSON artefact.

        Serialises ``amount_median_``, ``amount_iqr_``, ``amount_clip_upper_``,
        and ``time_period_seconds`` so the inference API can reconstruct a
        fully fitted transformer at startup without any training data.

        Args:
            path: Absolute path for the output JSON file.  Parent directories
                are created automatically.

        Raises:
            RuntimeError: If called before ``fit()`` or on any I/O error.
        """
        try:
            if not self.is_fitted_:
                raise RuntimeError(
                    "save_state() called on an unfitted transformer. "
                    "Call fit() before saving."
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            state: dict[str, float] = {
                "amount_median": self.amount_median_,
                "amount_iqr": self.amount_iqr_,
                "amount_clip_upper": self.amount_clip_upper_,
                "time_period_seconds": self.config.TIME_PERIOD_SECONDS,
            }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
            logger.info("Transformer state saved to: %s", path)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.exception("Failed to save transformer state.")
            raise RuntimeError(
                f"FraudFeatureTransformer.save_state() failed: {exc}"
            ) from exc

    def load_state(self, path: Path) -> Self:
        """Load fitted transformer statistics from a JSON artefact.

        Restores ``amount_median_``, ``amount_iqr_``, and
        ``amount_clip_upper_`` from the file and sets ``is_fitted_`` to
        ``True``, making the transformer ready to call ``transform()``
        immediately.

        Args:
            path: Absolute path to the JSON artefact written by
                ``save_state()``.

        Returns:
            ``self``, enabling method chaining.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            RuntimeError: On any JSON parsing or key-missing error.
        """
        try:
            if not path.exists():
                raise FileNotFoundError(
                    f"Transformer stats artefact not found at: {path}. "
                    "Run the training pipeline first."
                )
            with open(path, "r", encoding="utf-8") as fh:
                state: dict[str, float] = json.load(fh)
            self.amount_median_ = float(state["amount_median"])
            self.amount_iqr_ = float(state["amount_iqr"])
            self.amount_clip_upper_ = float(state["amount_clip_upper"])
            self.is_fitted_ = True
            logger.info(
                "Transformer state loaded from: %s\n"
                "  amount_median     : %.6f\n"
                "  amount_iqr        : %.6f\n"
                "  amount_clip_upper : %.6f",
                path,
                self.amount_median_,
                self.amount_iqr_,
                self.amount_clip_upper_,
            )
            return self
        except FileNotFoundError:
            raise
        except Exception as exc:
            logger.exception("Failed to load transformer state from: %s", path)
            raise RuntimeError(
                f"FraudFeatureTransformer.load_state() failed: {exc}"
            ) from exc

    def _encode_cyclical_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Project ``Time`` onto a unit circle and add an hour-of-day bucket.

        Produces three columns from the raw elapsed-seconds value:

            - ``Time_sin`` / ``Time_cos`` — continuous signals capturing the
              24-hour periodicity without an artificial boundary.
            - ``Time_hour`` — integer 0–23 derived from
              ``(Time % 86400) // 3600``.  Gives XGBoost a direct axis on
              which to cut "business hours vs night-time" fraud patterns.

        All three output columns are stored as their natural dtypes (float32,
        float32, int32).  The original ``Time`` column is dropped.

        Args:
            df: DataFrame containing a numeric ``Time`` column.

        Returns:
            DataFrame with ``Time`` replaced by ``Time_sin``, ``Time_cos``,
            and ``Time_hour``.
        """
        time_f64: pd.Series = df["Time"].astype(np.float64)
        angle: pd.Series = 2.0 * math.pi * time_f64 / self.config.TIME_PERIOD_SECONDS

        df["Time_sin"] = np.sin(angle).astype(np.float32)
        df["Time_cos"] = np.cos(angle).astype(np.float32)
        df["Time_hour"] = ((time_f64 % 86400.0) // 3600.0).astype(np.int32)
        df = df.drop(columns=["Time"])

        logger.debug(
            "Cyclical time encoding applied. "
            "Time_sin range: [%.4f, %.4f] | "
            "Time_cos range: [%.4f, %.4f] | "
            "Time_hour range: [%d, %d]",
            float(df["Time_sin"].min()),
            float(df["Time_sin"].max()),
            float(df["Time_cos"].min()),
            float(df["Time_cos"].max()),
            int(df["Time_hour"].min()),
            int(df["Time_hour"].max()),
        )
        return df

    def _engineer_amount(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive all Amount-based features and drop the raw column.

        Produces three columns from the raw ``Amount`` value:

            - ``Amount_log1p`` — ``log(1 + Amount)`` computed on the original
              pre-clip value.  Naturally compresses the extreme right tail and
              provides the model with a complementary magnitude signal that is
              robust to outliers without requiring fitted statistics.
            - ``Amount_is_round`` — ``1`` when ``Amount`` is a whole number,
              ``0`` otherwise.  Fraud transactions disproportionately use round
              values (e.g., $100, $500 ATM test charges).
            - ``Amount_scaled`` — Robust-scaled value:
              ``(clip(Amount, upper=amount_clip_upper_) − amount_median_) / amount_iqr_``.
              Clipping is applied first to prevent extreme outlier transactions
              from distorting the IQR scale for the majority of sub-$200
              transactions.

        The original ``Amount`` column is dropped after all three derived
        columns have been computed.

        Args:
            df: DataFrame containing a numeric ``Amount`` column.

        Returns:
            DataFrame with ``Amount`` replaced by ``Amount_log1p``,
            ``Amount_is_round``, and ``Amount_scaled``.
        """
        amount_f64: pd.Series = df["Amount"].astype(np.float64)

        df["Amount_log1p"] = np.log1p(amount_f64).astype(np.float32)

        df["Amount_is_round"] = ((amount_f64.round(0) == amount_f64)).astype(np.int32)

        amount_clipped: pd.Series = amount_f64.clip(upper=self.amount_clip_upper_)
        df["Amount_scaled"] = (
            (amount_clipped - self.amount_median_) / self.amount_iqr_
        ).astype(np.float32)

        df = df.drop(columns=["Amount"])

        logger.debug(
            "Amount feature engineering applied.\n"
            "  Amount_log1p   range: [%.4f, %.4f]\n"
            "  Amount_is_round freq: %.4f\n"
            "  Amount_scaled  range: [%.4f, %.4f] | mean: %.4f | std: %.4f",
            float(df["Amount_log1p"].min()),
            float(df["Amount_log1p"].max()),
            float(df["Amount_is_round"].mean()),
            float(df["Amount_scaled"].min()),
            float(df["Amount_scaled"].max()),
            float(df["Amount_scaled"].mean()),
            float(df["Amount_scaled"].std()),
        )
        return df
