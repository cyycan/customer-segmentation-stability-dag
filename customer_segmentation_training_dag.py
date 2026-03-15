"""
customer_segmentation_training_dag.py
──────────────────────────────────────
Airflow DAG — Customer Segmentation Training Pipeline

Schedule: Monthly (1st of each month at 00:00 UTC)
Tasks:
  1. load_data
  2. clean_and_rfm
  3. elbow_analysis
  4. run_clustering
  5. visualise_clusters
  6. log_to_mlflow
  7. export_segments

Inter-task communication uses Parquet files written to /tmp/seg_pipeline/
and paths passed via XCom (avoids Airflow's 48KB XCom limit).
"""

import os
import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# ─── Pipeline modules ─────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pipeline.data_loader import load_online_retail, get_data_summary
from pipeline.rfm_engineering import (
    clean_transactions, compute_rfm, score_rfm, get_rfm_diagnostics
)
from pipeline.clustering import (
    scale_features, run_elbow_analysis, plot_elbow,
    run_kmeans, run_gmm,
    plot_pca_clusters, plot_tsne_clusters,
    get_cluster_profiles,
)
from pipeline.mlflow_logger import log_training_run

# ─── Configuration ────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
N_CLUSTERS = int(os.getenv("SEG_N_CLUSTERS", "4"))
OUTPUT_DIR = os.getenv("SEG_OUTPUT_DIR", "/tmp/seg_pipeline")
PIPELINE_DIR = os.path.join(OUTPUT_DIR, "artifacts")

logger = logging.getLogger(__name__)

default_args = {
    "owner": "data-science",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# ─── Task functions ───────────────────────────────────────────────────────────

def task_load_data(**context):
    """Load dataset and push diagnostics + Parquet path to XCom."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_online_retail(source="url")
    summary = get_data_summary(df)
    logger.info(f"Data summary: {summary}")

    parquet_path = os.path.join(OUTPUT_DIR, "raw_data.parquet")
    df.to_parquet(parquet_path, index=False)

    context["ti"].xcom_push(key="raw_parquet_path", value=parquet_path)
    context["ti"].xcom_push(key="data_summary", value=summary)
    return parquet_path


def task_clean_and_rfm(**context):
    """Clean transactions and compute scored RFM features."""
    raw_path = context["ti"].xcom_pull(key="raw_parquet_path", task_ids="load_data")
    df = pd.read_parquet(raw_path)

    df_clean = clean_transactions(df)
    rfm = compute_rfm(df_clean)
    rfm = score_rfm(rfm)
    diagnostics = get_rfm_diagnostics(rfm)

    rfm_path = os.path.join(OUTPUT_DIR, "rfm.parquet")
    rfm.to_parquet(rfm_path)

    context["ti"].xcom_push(key="rfm_parquet_path", value=rfm_path)
    context["ti"].xcom_push(key="rfm_diagnostics", value=diagnostics)
    return rfm_path


def task_elbow_analysis(**context):
    """Run elbow method and push distortions + plot path to XCom."""
    rfm_path = context["ti"].xcom_pull(key="rfm_parquet_path", task_ids="clean_and_rfm")
    rfm = pd.read_parquet(rfm_path)

    X_scaled, scaler, feature_cols = scale_features(rfm)
    distortions = run_elbow_analysis(X_scaled)

    os.makedirs(PIPELINE_DIR, exist_ok=True)
    elbow_plot = plot_elbow(distortions, PIPELINE_DIR)

    # Persist scaler for downstream use
    import joblib
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    context["ti"].xcom_push(key="scaled_parquet_path",
                             value=rfm_path)   # rfm already saved
    context["ti"].xcom_push(key="scaler_path", value=scaler_path)
    context["ti"].xcom_push(key="elbow_plot_path", value=elbow_plot)
    context["ti"].xcom_push(key="distortions",
                             value={str(k): v for k, v in distortions.items()})
    return elbow_plot


def task_run_clustering(**context):
    """Run KMeans + GMM and save enriched RFM with cluster labels."""
    rfm_path = context["ti"].xcom_pull(key="rfm_parquet_path", task_ids="clean_and_rfm")
    rfm = pd.read_parquet(rfm_path)

    X_scaled, scaler, feature_cols = scale_features(rfm)

    # ── KMeans ────────────────────────────────────────────────────────────────
    km_labels, km_centroids, km_model, km_sil = run_kmeans(X_scaled, N_CLUSTERS)
    rfm["Cluster_KMeans"] = km_labels

    # ── GMM ───────────────────────────────────────────────────────────────────
    gmm_labels, gmm_model, gmm_sil = run_gmm(X_scaled, N_CLUSTERS)
    rfm["Cluster_GMM"] = gmm_labels

    # ── Cluster profiles ──────────────────────────────────────────────────────
    km_profile = get_cluster_profiles(rfm, "Cluster_KMeans")
    profile_path = os.path.join(OUTPUT_DIR, "cluster_profile.parquet")
    km_profile.to_parquet(profile_path)

    # ── Save enriched RFM ─────────────────────────────────────────────────────
    clustered_path = os.path.join(OUTPUT_DIR, "rfm_clustered.parquet")
    rfm.to_parquet(clustered_path)

    # ── Persist models ────────────────────────────────────────────────────────
    import joblib
    km_path = os.path.join(OUTPUT_DIR, "kmeans_model.pkl")
    gmm_path = os.path.join(OUTPUT_DIR, "gmm_model.pkl")
    centroids_path = os.path.join(OUTPUT_DIR, "kmeans_centroids.npy")
    joblib.dump(km_model, km_path)
    joblib.dump(gmm_model, gmm_path)
    np.save(centroids_path, km_centroids)

    context["ti"].xcom_push(key="clustered_parquet_path", value=clustered_path)
    context["ti"].xcom_push(key="km_model_path", value=km_path)
    context["ti"].xcom_push(key="centroids_path", value=centroids_path)
    context["ti"].xcom_push(key="profile_path", value=profile_path)
    context["ti"].xcom_push(key="kmeans_silhouette", value=km_sil)
    context["ti"].xcom_push(key="gmm_silhouette", value=gmm_sil)
    return clustered_path


def task_visualise_clusters(**context):
    """Generate PCA and t-SNE visualisations for KMeans and GMM clusters."""
    clustered_path = context["ti"].xcom_pull(
        key="clustered_parquet_path", task_ids="run_clustering"
    )
    rfm = pd.read_parquet(clustered_path)
    X_scaled, _, _ = scale_features(rfm)

    plots = []
    plots.append(plot_pca_clusters(
        X_scaled, rfm["Cluster_KMeans"].values,
        "Customer Segments by KMeans — PCA Projection",
        PIPELINE_DIR, "pca_kmeans.png"
    ))
    plots.append(plot_pca_clusters(
        X_scaled, rfm["Cluster_GMM"].values,
        "Customer Segments by GMM — PCA Projection",
        PIPELINE_DIR, "pca_gmm.png"
    ))
    plots.append(plot_tsne_clusters(
        X_scaled, rfm["Cluster_GMM"].values,
        "Customer Segments by GMM — t-SNE Projection",
        PIPELINE_DIR, "tsne_gmm.png"
    ))

    context["ti"].xcom_push(key="plot_paths", value=plots)
    return plots


def task_log_to_mlflow(**context):
    """Consolidate all outputs and log to MLflow."""
    ti = context["ti"]

    data_summary = ti.xcom_pull(key="data_summary", task_ids="load_data")
    rfm_diagnostics = ti.xcom_pull(key="rfm_diagnostics", task_ids="clean_and_rfm")
    km_sil = ti.xcom_pull(key="kmeans_silhouette", task_ids="run_clustering")
    gmm_sil = ti.xcom_pull(key="gmm_silhouette", task_ids="run_clustering")
    profile_path = ti.xcom_pull(key="profile_path", task_ids="run_clustering")
    km_model_path = ti.xcom_pull(key="km_model_path", task_ids="run_clustering")
    plot_paths = ti.xcom_pull(key="plot_paths", task_ids="visualise_clusters")
    elbow_plot = ti.xcom_pull(key="elbow_plot_path", task_ids="elbow_analysis")

    import joblib
    km_model = joblib.load(km_model_path)
    profile = pd.read_parquet(profile_path)

    run_period = context["ds"]  # YYYY-MM-DD execution date

    run_id = log_training_run(
        run_period=run_period,
        data_diagnostics=data_summary,
        rfm_diagnostics=rfm_diagnostics,
        n_clusters=N_CLUSTERS,
        kmeans_model=km_model,
        kmeans_sil=km_sil,
        gmm_sil=gmm_sil,
        cluster_profile=profile,
        artifact_paths=plot_paths + [elbow_plot],
        tracking_uri=MLFLOW_TRACKING_URI,
    )
    context["ti"].xcom_push(key="mlflow_run_id", value=run_id)
    return run_id


def task_export_segments(**context):
    """Export final customer segment assignments to CSV."""
    clustered_path = context["ti"].xcom_pull(
        key="clustered_parquet_path", task_ids="run_clustering"
    )
    rfm = pd.read_parquet(clustered_path)

    output_cols = [
        "Recency", "Frequency", "Monetary",
        "R_Score", "F_Score", "M_Score", "RFM_Score", "RFM_Segment",
        "Cluster_KMeans", "Cluster_GMM",
    ]
    export_cols = [c for c in output_cols if c in rfm.columns]
    export_path = os.path.join(OUTPUT_DIR, f"segments_{context['ds']}.csv")
    rfm[export_cols].to_csv(export_path)
    logger.info(f"Segments exported: {export_path}")
    return export_path


# ─── DAG Definition ───────────────────────────────────────────────────────────

with DAG(
    dag_id="customer_segmentation_training",
    description="Monthly customer segmentation training pipeline with MLflow tracking",
    schedule_interval="0 0 1 * *",   # 1st of every month
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    tags=["segmentation", "mlops", "rfm"],
) as dag:

    t1 = PythonOperator(
        task_id="load_data",
        python_callable=task_load_data,
    )
    t2 = PythonOperator(
        task_id="clean_and_rfm",
        python_callable=task_clean_and_rfm,
    )
    t3 = PythonOperator(
        task_id="elbow_analysis",
        python_callable=task_elbow_analysis,
    )
    t4 = PythonOperator(
        task_id="run_clustering",
        python_callable=task_run_clustering,
    )
    t5 = PythonOperator(
        task_id="visualise_clusters",
        python_callable=task_visualise_clusters,
    )
    t6 = PythonOperator(
        task_id="log_to_mlflow",
        python_callable=task_log_to_mlflow,
    )
    t7 = PythonOperator(
        task_id="export_segments",
        python_callable=task_export_segments,
    )

    t1 >> t2 >> t3 >> t4 >> [t5, t7]
    t5 >> t6
