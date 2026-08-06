""" Layer 3: Feature Engineering.

Provides FraudFeatureTransformer, a stateful scikit-learn-style
transformer used during training and inference.

Transformations:
- Cyclical time encoding
- Hour-of-day feature
- Robust amount scaling
- Log amount feature
- Round amount indicator

The transformer learns scaling statistics during fit() and
reuses them during transform() to prevent data leakage. """

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
    """ Stateful feature transformer for the fraud detection pipeline.

    Learns feature-engineering statistics during fit() and applies
    the same transformations during inference to ensure consistency
    and prevent data leakage. """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialise the transformer."""
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
        """Learn Amount scaling statistics from the training data.
        Args:
            df: DataFrame containing at least an Amount column.  The
                Class column may be present; it is ignored here.
        Returns:
            self, enabling method chaining (transformer.fit(df).transform(df)).
        Raises:
            KeyError: If Amount column is absent from df.
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
        
        Args:
            df: DataFrame containing Time and Amount columns (plus any
                PCA components V1–V28 and optional Class column).

        Returns:
            Transformed DataFrame with Time and Amount replaced by
            their engineered counterparts.

        Raises:
            RuntimeError: If called before fit().
            KeyError: If Time or Amount columns are absent.
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
        """Fit the transformer and immediately transform the data."""
        
        logger.info("fit_transform() called — fitting then transforming in one pass.")
        return self.fit(df).transform(df)

    def save_state(self, path: Path) -> None:
       """Persist fitted transformer statistics to JSON."""
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
       """Load fitted transformer statistics from JSON."""
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
        """Encode the Time feature into cyclical and hour-based features."""
        
        time_f64: pd.Series = df["Time"].astype(np.float64)
        angle: pd.Series = 2.0 * math.pi * time_f64 / self.config.TIME_PERIOD_SECONDS

        df["Time_sin"] = np.sin(angle).astype(np.float32)
        df["Time_cos"] = np.cos(angle).astype(np.float32)
        df["Time_hour"] = ((time_f64 % 86400.0) // 3600.0).astype(np.int32)
        df = df.drop(columns=["Time"])

        return df

    def _engineer_amount(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Amount-derived features and remove the raw Amount column."""
        amount_f64: pd.Series = df["Amount"].astype(np.float64)

        df["Amount_log1p"] = np.log1p(amount_f64).astype(np.float32)

        df["Amount_is_round"] = ((amount_f64.round(0) == amount_f64)).astype(np.int32)

        amount_clipped: pd.Series = amount_f64.clip(upper=self.amount_clip_upper_)
        df["Amount_scaled"] = (
            (amount_clipped - self.amount_median_) / self.amount_iqr_
        ).astype(np.float32)

        df = df.drop(columns=["Amount"])

        return df
