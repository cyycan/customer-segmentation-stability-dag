"""
RFM Feature Engineering — Customer Segmentation Pipeline
Cleans raw transaction data and computes Recency, Frequency, Monetary metrics.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Online Retail data:
      - Drop rows with missing CustomerID
      - Remove returns (negative quantities)
      - Parse InvoiceDate
      - Compute TotalPrice
    """
    original_rows = len(df)
    df = df.dropna(subset=["CustomerID"]).copy()
    df = df[df["Quantity"] > 0]
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    logger.info(
        f"Cleaning: {original_rows:,} → {len(df):,} rows "
        f"({original_rows - len(df):,} removed)"
    )
    return df


def compute_rfm(df: pd.DataFrame, reference_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Compute RFM features for each customer.

    Args:
        df: Cleaned transactions DataFrame.
        reference_date: Snapshot date for Recency calculation.
                        Defaults to max(InvoiceDate) + 1 day.

    Returns:
        DataFrame indexed by CustomerID with columns:
        Recency, Frequency, Monetary
    """
    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )

    logger.info(
        f"RFM computed for {len(rfm):,} customers. "
        f"Reference date: {reference_date.date()}"
    )
    return rfm


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign R/F/M quintile scores (1–5) and build composite RFM_Score.
    Higher scores = better customer value.
    """
    rfm = rfm.copy()
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    )
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])
    rfm["RFM_Segment"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )
    rfm["RFM_Score"] = (
        rfm[["R_Score", "F_Score", "M_Score"]].astype(int).sum(axis=1)
    )
    logger.info("RFM scoring complete.")
    return rfm


def get_rfm_diagnostics(rfm: pd.DataFrame) -> dict:
    """Return descriptive statistics for MLflow logging."""
    return {
        "rfm_customers": len(rfm),
        "recency_mean": round(rfm["Recency"].mean(), 2),
        "recency_median": round(rfm["Recency"].median(), 2),
        "frequency_mean": round(rfm["Frequency"].mean(), 2),
        "frequency_median": round(rfm["Frequency"].median(), 2),
        "monetary_mean": round(rfm["Monetary"].mean(), 2),
        "monetary_median": round(rfm["Monetary"].median(), 2),
        "rfm_score_mean": round(rfm["RFM_Score"].mean(), 2),
    }
