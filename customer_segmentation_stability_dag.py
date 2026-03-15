"""
customer_segmentation_stability_dag.py
──────────────────────────────────────
Airflow DAG — Cluster Stability Monitoring

Schedule: Weekly (every Monday at 06:00 UTC)
Tasks:
  1. load_current_data       — re-compute RFM for the current period
  2. load_reference_model    — load previous KMeans model + centroids
  3. run_current_clustering  — cluster current data
  4. compute_ari             — label-level stability (ARI)
  5. compute_centroid_shift  — centroid-level drift
  6. compute_silhouette      — cluster quality vs previous run
  7. compute_profile_drift   — mean RFM per cluster vs previous
  8. build_stability_report  — aggregate all signals
  9. log_stability_to_mlflow — push report to MLflow
 10. alert_on_degradation    — log/print actionable alerts

All stability metrics from the documentation are captured:
  - ARI (Adjusted Rand Index)
  - Centroid shift (Euclidean distance)
  - Silhouette score comparison
  - Cluster profile (mean RFM) drift per cluster
"""

import os
import json
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pipeline.data_loader import load_online_retail
from pipeline.rfm_engineering import clean_transactions, compute_rfm, score_rfm
from pipeline.clustering import (
    scale_features, run_kmeans,
    get_cluster_profiles,
)
from pipeline.stability_monitor import (
    compute_ari, interpret_ari,
    compute_centroid_shift,
    evaluate_silhouette, compare_silhouette,
    compute_profile_drift, interpret_profile_drift,
    build_stability_report, save_stability_report,
    plot_stability_history,
)
from pipeline.mlflow_logger import log_stability_run

# ─── Configuration ────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
N_CLUSTERS = int(os.getenv("SEG_N_CLUSTERS", "4"))
OUTPUT_DIR = os.getenv("SEG_OUTPUT_DIR", "/tmp/seg_pipeline")
STABILITY_DIR = os.path.join(OUTPUT_DIR, "stability")
HISTORY_FILE = os.path.join(STABILITY_DIR, "stability_history.json")

logger = logging.getLogger(__name__)

default_args = {
    "owner": "data-science",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


# ─── Helper: history management ──────────────────────────────────────────────

def _load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def _save_history(history: list):
    os.makedirs(STABILITY_DIR, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def _get_previous_run_metadata() -> dict:
    """Return metadata from the most recent stability history entry."""
    history = _load_history()
    if not history:
        return {}
    return history[-1]


# ─── Task functions ───────────────────────────────────────────────────────────

def task_load_current_data(**context):
    """Load and compute RFM for the current monitoring window."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_online_retail(source="url")
    df_clean = clean_transactions(df)
    rfm = compute_rfm(df_clean)
    rfm = score_rfm(rfm)

    rfm_path = os.path.join(OUTPUT_DIR, "stability_current_rfm.parquet")
    rfm.to_parquet(rfm_path)
    context["ti"].xcom_push(key="current_rfm_path", value=rfm_path)
    return rfm_path


def task_load_reference_model(**context):
    """
    Load the reference (previous) KMeans model, centroids, and cluster labels.
    Falls back to fitting a fresh reference model if no saved model exists.
    """
    km_model_path = os.path.join(OUTPUT_DIR, "kmeans_model.pkl")
    centroids_path = os.path.join(OUTPUT_DIR, "kmeans_centroids.npy")
    ref_rfm_path = os.path.join(OUTPUT_DIR, "rfm_clustered.parquet")

    model_exists = os.path.exists(km_model_path) and os.path.exists(centroids_path)
    ref_data_exists = os.path.exists(ref_rfm_path)

    if model_exists and ref_data_exists:
        logger.info("Reference model and data found — loading.")
        context["ti"].xcom_push(key="reference_model_path", value=km_model_path)
        context["ti"].xcom_push(key="reference_centroids_path", value=centroids_path)
        context["ti"].xcom_push(key="reference_rfm_path", value=ref_rfm_path)
        context["ti"].xcom_push(key="reference_available", value=True)
    else:
        logger.warning(
            "No reference model found. This appears to be the first stability run. "
            "Fitting reference model from current data to bootstrap history."
        )
        current_rfm_path = context["ti"].xcom_pull(
            key="current_rfm_path", task_ids="load_current_data"
        )
        rfm = pd.read_parquet(current_rfm_path)
        X_scaled, scaler, _ = scale_features(rfm)
        _, centroids, km_model, sil = run_kmeans(X_scaled, N_CLUSTERS)

        joblib.dump(km_model, km_model_path)
        np.save(centroids_path, centroids)

        # Label the reference data so we can compare next run
        rfm["Cluster_KMeans"] = km_model.predict(X_scaled)
        rfm.to_parquet(ref_rfm_path)

        context["ti"].xcom_push(key="reference_model_path", value=km_model_path)
        context["ti"].xcom_push(key="reference_centroids_path", value=centroids_path)
        context["ti"].xcom_push(key="reference_rfm_path", value=ref_rfm_path)
        context["ti"].xcom_push(key="reference_available", value=False)
    return km_model_path


def task_run_current_clustering(**context):
    """Cluster the current RFM data with fresh KMeans and GMM."""
    current_rfm_path = context["ti"].xcom_pull(
        key="current_rfm_path", task_ids="load_current_data"
    )
    rfm = pd.read_parquet(current_rfm_path)
    X_scaled, scaler, _ = scale_features(rfm)

    km_labels, km_centroids, km_model, km_sil = run_kmeans(X_scaled, N_CLUSTERS)
    rfm["Cluster_KMeans"] = km_labels

    current_clustered_path = os.path.join(OUTPUT_DIR, "stability_current_clustered.parquet")
    rfm.to_parquet(current_clustered_path)

    current_centroids_path = os.path.join(OUTPUT_DIR, "stability_current_centroids.npy")
    np.save(current_centroids_path, km_centroids)

    context["ti"].xcom_push(key="current_clustered_path", value=current_clustered_path)
    context["ti"].xcom_push(key="current_centroids_path", value=current_centroids_path)
    context["ti"].xcom_push(key="current_km_labels", value=km_labels.tolist())
    context["ti"].xcom_push(key="current_silhouette", value=km_sil)
    return current_clustered_path


def task_compute_ari(**context):
    """Compute ARI between reference and current cluster labels."""
    ref_rfm_path = context["ti"].xcom_pull(
        key="reference_rfm_path", task_ids="load_reference_model"
    )
    current_clustered_path = context["ti"].xcom_pull(
        key="current_clustered_path", task_ids="run_current_clustering"
    )

    ref_rfm = pd.read_parquet(ref_rfm_path)
    current_rfm = pd.read_parquet(current_clustered_path)

    # Align on shared CustomerIDs
    common_ids = ref_rfm.index.intersection(current_rfm.index)
    if len(common_ids) < 50:
        logger.warning(
            f"Only {len(common_ids)} common customers found. ARI may not be meaningful."
        )

    labels_ref = ref_rfm.loc[common_ids, "Cluster_KMeans"].values
    labels_cur = current_rfm.loc[common_ids, "Cluster_KMeans"].values

    ari = compute_ari(labels_ref, labels_cur)
    interpretation = interpret_ari(ari)
    logger.info(f"ARI interpretation: {interpretation}")

    context["ti"].xcom_push(key="ari", value=ari)
    context["ti"].xcom_push(key="ari_status", value=interpretation["status"])
    return ari


def task_compute_centroid_shift(**context):
    """Compute Euclidean centroid shift between reference and current KMeans."""
    ref_centroids_path = context["ti"].xcom_pull(
        key="reference_centroids_path", task_ids="load_reference_model"
    )
    current_centroids_path = context["ti"].xcom_pull(
        key="current_centroids_path", task_ids="run_current_clustering"
    )

    old_centroids = np.load(ref_centroids_path)
    new_centroids = np.load(current_centroids_path)

    shift = compute_centroid_shift(old_centroids, new_centroids)
    logger.info(f"Centroid shift: {shift}")

    context["ti"].xcom_push(key="centroid_shift", value=shift)
    return shift


def task_compute_silhouette(**context):
    """Compare current silhouette vs previous run from history."""
    current_clustered_path = context["ti"].xcom_pull(
        key="current_clustered_path", task_ids="run_current_clustering"
    )
    rfm = pd.read_parquet(current_clustered_path)
    X_scaled, _, _ = scale_features(rfm)
    current_labels = rfm["Cluster_KMeans"].values
    current_sil = evaluate_silhouette(X_scaled, current_labels)

    prev_metadata = _get_previous_run_metadata()
    previous_sil = prev_metadata.get("silhouette_current", current_sil)

    comparison = compare_silhouette(previous_sil, current_sil)
    logger.info(f"Silhouette comparison: {comparison}")

    context["ti"].xcom_push(key="silhouette_comparison", value=comparison)
    return comparison


def task_compute_profile_drift(**context):
    """Compare mean RFM per cluster across periods."""
    ref_rfm_path = context["ti"].xcom_pull(
        key="reference_rfm_path", task_ids="load_reference_model"
    )
    current_clustered_path = context["ti"].xcom_pull(
        key="current_clustered_path", task_ids="run_current_clustering"
    )

    ref_rfm = pd.read_parquet(ref_rfm_path)
    cur_rfm = pd.read_parquet(current_clustered_path)

    # Both must have Cluster_KMeans
    if "Cluster_KMeans" not in ref_rfm.columns:
        logger.warning("Reference RFM has no Cluster_KMeans — skipping profile drift.")
        context["ti"].xcom_push(key="profile_flags", value={})
        return {}

    profile_ref = get_cluster_profiles(ref_rfm, "Cluster_KMeans")
    profile_cur = get_cluster_profiles(cur_rfm, "Cluster_KMeans")

    drift_df = compute_profile_drift(profile_ref, profile_cur)
    flags = interpret_profile_drift(drift_df)

    # Persist profile comparison
    os.makedirs(STABILITY_DIR, exist_ok=True)
    drift_path = os.path.join(STABILITY_DIR, "profile_drift.parquet")
    drift_df.to_parquet(drift_path)

    logger.info(f"Profile drift flags: {flags}")
    context["ti"].xcom_push(key="profile_flags", value=flags)
    context["ti"].xcom_push(key="profile_ref_path", value=drift_path)
    return flags


def task_build_stability_report(**context):
    """Assemble all stability signals into a single report."""
    ti = context["ti"]
    ari = ti.xcom_pull(key="ari", task_ids="compute_ari")
    centroid_shift = ti.xcom_pull(key="centroid_shift", task_ids="compute_centroid_shift")
    silhouette_comparison = ti.xcom_pull(
        key="silhouette_comparison", task_ids="compute_silhouette"
    )
    profile_flags = ti.xcom_pull(
        key="profile_flags", task_ids="compute_profile_drift"
    ) or {}

    run_period = context["ds"]
    report = build_stability_report(
        ari=ari,
        centroid_shift=centroid_shift,
        silhouette_comparison=silhouette_comparison,
        profile_flags=profile_flags,
        run_period=run_period,
    )

    # Save to disk
    os.makedirs(STABILITY_DIR, exist_ok=True)
    report_path = save_stability_report(report, STABILITY_DIR, run_period)

    # Update history
    history = _load_history()
    history.append({
        "run_period": run_period,
        "ari": ari,
        "silhouette_current": silhouette_comparison.get("silhouette_current"),
        "centroid_shift_mean": centroid_shift.get("centroid_shift_mean"),
        "overall_status": report["overall_status"],
    })
    _save_history(history)

    context["ti"].xcom_push(key="stability_report", value=report)
    context["ti"].xcom_push(key="stability_report_path", value=report_path)
    return report


def task_log_stability_to_mlflow(**context):
    """Log stability report and plots to the MLflow Stability experiment."""
    ti = context["ti"]
    report = ti.xcom_pull(key="stability_report", task_ids="build_stability_report")
    report_path = ti.xcom_pull(
        key="stability_report_path", task_ids="build_stability_report"
    )
    run_period = context["ds"]

    # Generate stability history plot if we have enough history
    history = _load_history()
    history_plot = plot_stability_history(history, STABILITY_DIR)

    artifact_paths = [p for p in [report_path, history_plot] if p]

    run_id = log_stability_run(
        run_period=run_period,
        stability_report=report,
        artifact_paths=artifact_paths,
        tracking_uri=MLFLOW_TRACKING_URI,
    )
    logger.info(f"Stability run logged to MLflow: {run_id}")
    return run_id


def task_alert_on_degradation(**context):
    """
    Emit structured alerts for any WARNING or CRITICAL signals.
    In production: replace logger calls with PagerDuty / Slack / email hooks.
    """
    report = context["ti"].xcom_pull(
        key="stability_report", task_ids="build_stability_report"
    )
    status = report.get("overall_status", "UNKNOWN")

    divider = "=" * 60
    print(f"\n{divider}")
    print(f"CLUSTER STABILITY ALERT — {report.get('run_period')}")
    print(f"Overall Status: {status}")
    print(divider)

    checks = [
        ("ARI", report.get("ari"), report.get("ari_status")),
        ("Centroid Shift Mean", report.get("centroid_shift_mean"),
         report.get("centroid_shift_status")),
        ("Silhouette Drop", report.get("silhouette_drop"),
         report.get("silhouette_status")),
    ]
    for label, value, check_status in checks:
        icon = "✅" if check_status == "STABLE" else ("⚠️" if check_status == "WARNING" else "🚨")
        print(f"  {icon} {label}: {value:.4f} [{check_status}]")

    print("\n  Cluster Profile Drift:")
    for cluster, flag in report.get("cluster_profile_flags", {}).items():
        icon = "✅" if flag["status"] == "STABLE" else (
            "⚠️" if flag["status"] == "WARNING" else "🚨"
        )
        print(f"    {icon} Cluster {cluster}: max drift={flag['max_drift']:.4f} [{flag['status']}]")

    print(divider)

    if status in ("WARNING", "CRITICAL"):
        logger.warning(
            f"[STABILITY ALERT] Overall status={status}. "
            "Consider reviewing clustering parameters or triggering retraining."
        )
    else:
        logger.info("Stability check passed. No action required.")

    return status


# ─── DAG Definition ───────────────────────────────────────────────────────────

with DAG(
    dag_id="customer_segmentation_stability_monitoring",
    description="Weekly cluster stability monitoring — ARI, centroid shift, silhouette, profile drift",
    schedule_interval="0 6 * * 1",  # Every Monday at 06:00 UTC
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    tags=["segmentation", "stability", "monitoring", "mlops"],
) as dag:

    t1 = PythonOperator(
        task_id="load_current_data",
        python_callable=task_load_current_data,
    )
    t2 = PythonOperator(
        task_id="load_reference_model",
        python_callable=task_load_reference_model,
    )
    t3 = PythonOperator(
        task_id="run_current_clustering",
        python_callable=task_run_current_clustering,
    )
    t4 = PythonOperator(
        task_id="compute_ari",
        python_callable=task_compute_ari,
    )
    t5 = PythonOperator(
        task_id="compute_centroid_shift",
        python_callable=task_compute_centroid_shift,
    )
    t6 = PythonOperator(
        task_id="compute_silhouette",
        python_callable=task_compute_silhouette,
    )
    t7 = PythonOperator(
        task_id="compute_profile_drift",
        python_callable=task_compute_profile_drift,
    )
    t8 = PythonOperator(
        task_id="build_stability_report",
        python_callable=task_build_stability_report,
    )
    t9 = PythonOperator(
        task_id="log_stability_to_mlflow",
        python_callable=task_log_stability_to_mlflow,
    )
    t10 = PythonOperator(
        task_id="alert_on_degradation",
        python_callable=task_alert_on_degradation,
    )

    # load data first, reference model bootstraps from it
    t1 >> t2 >> t3
    # parallel stability checks after clustering
    t3 >> [t4, t5, t6, t7]
    # converge into report
    [t4, t5, t6, t7] >> t8
    # log and alert
    t8 >> [t9, t10]
