"""
Data Loader — Customer Segmentation Pipeline
Loads the Online Retail dataset from a URL or local CSV.
"""

import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

DATASET_URL = (
    "https://raw.githubusercontent.com/Umesh-01/"
    "Customer-Segmentation-Dataset/main/Online%20Retail.csv"
)


def load_online_retail(source: str = "url", local_path: str = None) -> pd.DataFrame:
    """
    Load the Online Retail dataset.

    Args:
        source: 'url' to download from GitHub, 'local' to load from local_path.
        local_path: Path to local CSV file (required when source='local').

    Returns:
        Raw DataFrame with all original columns.
    """
    if source == "url":
        logger.info("Downloading Online Retail dataset from GitHub...")
        df = pd.read_csv(DATASET_URL, encoding="ISO-8859-1")
    elif source == "local":
        if not local_path or not os.path.exists(local_path):
            raise FileNotFoundError(f"Local data file not found: {local_path}")
        logger.info(f"Loading dataset from {local_path}...")
        df = pd.read_csv(local_path, encoding="ISO-8859-1")
    else:
        raise ValueError(f"Unknown source: {source}. Use 'url' or 'local'.")

    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return basic data quality diagnostics."""
    return {
        "total_rows": len(df),
        "missing_customer_ids": df["CustomerID"].isna().sum(),
        "negative_quantities": (df["Quantity"] < 0).sum(),
        "unique_customers": df["CustomerID"].nunique(),
        "date_min": str(df["InvoiceDate"].min()) if "InvoiceDate" in df.columns else None,
        "date_max": str(df["InvoiceDate"].max()) if "InvoiceDate" in df.columns else None,
    }
