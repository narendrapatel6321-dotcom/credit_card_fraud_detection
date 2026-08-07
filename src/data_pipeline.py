"""
Layers 1 & 2: Data Ingestion and Integrity.

Provides two components for preparing the raw credit card fraud dataset:
- FraudDataLoader: Loads, validates, cleans, and downcasts the dataset.
- ImbalanceAnalyzer: Computes class distribution statistics and the
  XGBoost scale_pos_weight value.
"""

import logging
from typing import Any
 
import numpy as np
import pandas as pd
 
from config import PipelineConfig
 
logger: logging.Logger = logging.getLogger(__name__)
 
 
class FraudDataLoader:
    """ Load and prepare the raw credit card fraud dataset for training. """
 
    def __init__(self, config: PipelineConfig) -> None:
        """ Initialise the loader with a validated pipeline configuration. """
        self.config: PipelineConfig = config
        logger.info(
            "FraudDataLoader initialised. Target data path: %s", config.DATA_PATH
        )
 
    def load(self) -> pd.DataFrame:
        """
        Load the dataset, apply integrity checks, and optimise memory usage.

        Returns:
           Cleaned DataFrame ready for feature engineering.

        Raises:
          FileNotFoundError: If the input CSV cannot be found.
          ValueError: If the cleaned dataset is empty.
          RuntimeError: If an unexpected loading error occurs.
         """
        try:
            logger.info("Loading raw CSV from: %s", self.config.DATA_PATH)
            df: pd.DataFrame = pd.read_csv(self.config.DATA_PATH)
            logger.info(
                "Raw CSV loaded. Shape: %s. Columns: %s",
                df.shape,
                df.columns.tolist(),
            )
 
            df = self._handle_integrity(df)
            df = self._downcast_memory(df)
 
            if df.empty:
                raise ValueError(
                    "DataFrame is empty after integrity processing. "
                    "Verify the source CSV at: %s" % self.config.DATA_PATH
                )
 
            logger.info(
                "Data loading complete. Final shape: %s. Memory usage: %.2f MB",
                df.shape,
                df.memory_usage(deep=True).sum() / (1024**2),
            )
            return df
 
        except FileNotFoundError:
            logger.error("Data file not found at path: %s", self.config.DATA_PATH)
            raise
        except ValueError:
            logger.error("DataFrame is empty after integrity processing.")
            raise
        except Exception as exc:
            logger.error("Unexpected error during data loading: %s", exc, exc_info=True)
            raise RuntimeError(
                "FraudDataLoader.load() failed with an unexpected error."
            ) from exc
 
    def _handle_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove invalid and duplicate rows from the dataset.

        Applies integrity checks for missing values, infinite values,
        negative Amount/Time values, and duplicate records.
        Args:
            df: Raw dataset loaded from CSV.
        Returns:
            Cleaned DataFrame with integrity issues removed.
        """
        initial_rows: int = len(df)
        has_class: bool = "Class" in df.columns
 
        if has_class:
            initial_fraud: int = int((df["Class"] == 1).sum())
            logger.info(
                "Raw data: %d rows | %d fraud (%.4f%%)",
                initial_rows,
                initial_fraud,
                initial_fraud / max(initial_rows, 1) * 100,
            )
 
        # replace inf with NaN and audit per column
        df = df.replace([np.inf, -np.inf], np.nan)
        nan_per_col: pd.Series = df.isnull().sum()
        affected_cols: pd.Series = nan_per_col[nan_per_col > 0]
        if not affected_cols.empty:
            logger.info(
                "Columns containing NaN / Inf values (row counts):\n%s",
                affected_cols.to_string(),
            )
        else:
            logger.info("No NaN or Inf values detected in any column.")
 
        # flag and neutralise negative Amount/Time (corrupted rows;
        # should never occur in this dataset). Converted to NaN so they are
        # removed by the dropna() step below and counted in its logging.
        negative_amount_mask: pd.Series = (
            df["Amount"] < 0 if "Amount" in df.columns else pd.Series(False, index=df.index)
        )
        negative_time_mask: pd.Series = (
            df["Time"] < 0 if "Time" in df.columns else pd.Series(False, index=df.index)
        )
        negative_mask: pd.Series = negative_amount_mask | negative_time_mask
 
        if negative_mask.any():
            logger.warning(
                "Found %d rows with negative Amount/Time — treating as corrupted, "
                "will be dropped in NaN cleanup step.",
                int(negative_mask.sum()),
            )
            df.loc[negative_mask, ["Amount", "Time"]] = np.nan
        else:
            logger.info("No negative Amount or Time values detected.")
 
        # drop NaN rows
        df = df.dropna()
        after_nan: int = len(df)
        nan_rows_dropped: int = initial_rows - after_nan
        logger.info(
            "NaN rows dropped: %d (%.4f%% of raw data).",
            nan_rows_dropped,
            nan_rows_dropped / max(initial_rows, 1) * 100,
        )
        if has_class:
            fraud_post_nan: int = int((df["Class"] == 1).sum())
            logger.info(
                "Fraud rows after NaN drop: %d (%.4f%% of remaining data).",
                fraud_post_nan,
                fraud_post_nan / max(after_nan, 1) * 100,
            )
 
        # drop exact duplicates
        df = df.drop_duplicates()
        after_dedup: int = len(df)
        dup_rows_dropped: int = after_nan - after_dedup
        logger.info(
            "Duplicate rows dropped: %d (%.4f%% of post-NaN data).",
            dup_rows_dropped,
            dup_rows_dropped / max(after_nan, 1) * 100,
        )
        if has_class:
            fraud_post_dedup: int = int((df["Class"] == 1).sum())
            logger.info(
                "Fraud rows after dedup: %d (%.4f%% of remaining data).",
                fraud_post_dedup,
                fraud_post_dedup / max(after_dedup, 1) * 100,
            )
 
        logger.info(
            "Integrity check complete. Rows: %d → %d (net removed: %d).",
            initial_rows,
            after_dedup,
            initial_rows - after_dedup,
        )
        return df.reset_index(drop=True)
 
    def _downcast_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce DataFrame memory usage by downcasting numeric columns.
        Args:
             df: Input DataFrame.
        Returns:
             Memory-optimised DataFrame. """
     
        float64_cols: list[str] = [
            col for col in df.columns if df[col].dtype == np.float64
        ]
        df[float64_cols] = df[float64_cols].astype(np.float32)
        logger.info(
            "Downcast %d float64 columns → float32: %s",
            len(float64_cols),
            float64_cols,
        )
 
        if "Class" in df.columns:
            df["Class"] = df["Class"].astype(np.int32)
            logger.info("Column 'Class' cast to int32.")
 
        return df
 
 
class ImbalanceAnalyzer:
    """ Analyse class imbalance and compute training statistics. """
 
    def __init__(self, config: PipelineConfig) -> None:
       """ Initialise the imbalance analyzer. """
        self.config: PipelineConfig = config
        logger.info("ImbalanceAnalyzer initialised.")
 
    def analyze(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Compute class distribution statistics for the target column.
        Args:
             target: Binary fraud labels.
        Returns:
             Dictionary containing class counts, fraud prevalence,
             and the recommended XGBoost scale_pos_weight value.
        Raises:
             ValueError: If the target labels are invalid. """
        try:
            if "Class" not in df.columns:
                raise KeyError(
                    "'Class' column not found. DataFrame columns: %s"
                    % df.columns.tolist()
                )
 
            value_counts: pd.Series = df["Class"].value_counts()
            negative_count: int = int(value_counts.get(0, 0))
            positive_count: int = int(value_counts.get(1, 0))
            total_rows: int = len(df)
 
            if positive_count == 0:
                raise ValueError(
                    "No positive (fraud) samples found in 'Class' column. "
                    "Cannot compute scale_pos_weight."
                )
 
            negative_frequency: float = negative_count / total_rows
            positive_frequency: float = positive_count / total_rows
            imbalance_ratio: float = negative_count / positive_count
            scale_pos_weight: float = self.calculate_scale_pos_weight(df)
 
            stats: dict[str, Any] = {
                "total_rows": total_rows,
                "negative_count": negative_count,
                "positive_count": positive_count,
                "negative_frequency": negative_frequency,
                "positive_frequency": positive_frequency,
                "imbalance_ratio": imbalance_ratio,
                "scale_pos_weight": scale_pos_weight,
            }
 
            logger.info(
                "Imbalance analysis complete.\n"
                "  Total rows         : %d\n"
                "  Legitimate (0)     : %d (%.4f%%)\n"
                "  Fraud (1)          : %d (%.4f%%)\n"
                "  Imbalance ratio    : %.2f : 1\n"
                "  scale_pos_weight   : %.4f",
                total_rows,
                negative_count,
                negative_frequency * 100,
                positive_count,
                positive_frequency * 100,
                imbalance_ratio,
                scale_pos_weight,
            )
            return stats
 
        except KeyError:
            logger.error("'Class' column missing from DataFrame.")
            raise
        except ValueError as exc:
            logger.error("Cannot compute imbalance stats: %s", exc)
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error in ImbalanceAnalyzer.analyze(): %s",
                exc,
                exc_info=True,
            )
            raise RuntimeError(
                "ImbalanceAnalyzer.analyze() failed with an unexpected error."
            ) from exc
 
    def calculate_scale_pos_weight(self, df: pd.DataFrame) -> float:
        """ Derive the XGBoost scale_pos_weight from class counts. """
        try:
            negative_count: int = int((df["Class"] == 0).sum())
            positive_count: int = int((df["Class"] == 1).sum())
 
            if positive_count == 0:
                raise ZeroDivisionError(
                    "Positive class count is zero — scale_pos_weight is undefined."
                )
 
            weight: float = negative_count / positive_count
            logger.debug(
                "scale_pos_weight computed: %d / %d = %.4f",
                negative_count,
                positive_count,
                weight,
            )
            return weight
 
        except KeyError:
            logger.error("'Class' column missing; cannot compute scale_pos_weight.")
            raise
        except ZeroDivisionError as exc:
            logger.error("Division by zero in scale_pos_weight: %s", exc)
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error in calculate_scale_pos_weight(): %s",
                exc,
                exc_info=True,
            )
            raise RuntimeError(
                "calculate_scale_pos_weight() failed with an unexpected error."
            ) from exc
