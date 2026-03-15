# Customer Segmentation MLOps Pipeline

> Production-ready customer segmentation system with Apache Airflow orchestration, MLflow experiment tracking, and automated cluster stability monitoring — built on RFM analysis, KMeans, and GMM clustering.

---

## What's Inside

| Component | Description |
|---|---|
| **Training DAG** | Monthly Airflow pipeline: load → clean → RFM → cluster → visualise → MLflow log → export |
| **Stability DAG** | Weekly monitoring: ARI, centroid shift, silhouette tracking, cluster profile drift |
| **Pipeline modules** | Modular Python: `data_loader`, `rfm_engineering`, `clustering`, `stability_monitor`, `mlflow_logger` |
| **Docker Compose** | One-command stack: Airflow 2.8 + MLflow + PostgreSQL |

---

## Architecture

```
Raw Transactions (Online Retail)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│            Airflow: Training DAG (monthly)              │
│                                                         │
│  load_data → clean_and_rfm → elbow_analysis             │
│      → run_clustering (KMeans + GMM)                    │
│          → visualise_clusters → log_to_mlflow           │
│              → export_segments                          │
│                                                         │
│  (MLflow logs params, metrics, models, artifacts)       │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│         Airflow: Stability DAG (weekly)                 │
│                                                         │
│  load_current_data → load_reference_model               │
│      → run_current_clustering                           │
│          → [compute_ari | compute_centroid_shift |      │
│              compute_silhouette | compute_profile_drift] │
│          → build_stability_report                       │
│          → [log_stability_to_mlflow | alert_on_degradation] │
└─────────────────────────────────────────────────────────┘
         │
         ▼
  MLflow UI (port 5001) — two experiments:
    • Customer_Segmentation_RFM
    • Customer_Segmentation_Stability
```

---

## Stability Monitoring Framework

All stability checks are based on the team's documented framework:

### 1. Adjusted Rand Index (ARI)
Label-level consistency between clustering runs.

| ARI Range | Status | Action |
|---|---|---|
| ≥ 0.80 | ✅ STABLE | No action |
| 0.60 – 0.80 | ⚠️ WARNING | Review RFM distribution, consider re-evaluating k |
| < 0.60 | 🚨 CRITICAL | Re-run elbow analysis, recalibrate clusters |

### 2. Centroid Shift (Euclidean Distance)
Mean shift of KMeans cluster centres between periods.

| Shift | Status | Action |
|---|---|---|
| < 0.5 | ✅ STABLE | No action |
| 0.5 – 1.0 | ⚠️ WARNING | Monitor next period |
| ≥ 1.0 | 🚨 CRITICAL | Trigger retraining |

### 3. Silhouette Score
Absolute drop in silhouette score vs previous run.

| Drop | Status |
|---|---|
| ≤ 0.05 | ✅ STABLE |
| 0.05 – 0.10 | ⚠️ WARNING |
| > 0.10 | 🚨 CRITICAL |

### 4. Cluster Profile Drift
Relative change in mean Recency, Frequency, Monetary per cluster.

| Max % Change | Status |
|---|---|
| < 15% | ✅ STABLE |
| 15% – 30% | ⚠️ WARNING |
| > 30% | 🚨 CRITICAL |

All four signals are logged to **MLflow** (`Customer_Segmentation_Stability` experiment) and combined into an **overall status** (worst signal wins).

---

## Quickstart

### Option A — Docker (recommended)

```bash
git clone https://github.com/your-username/customer-segmentation-pipeline.git
cd customer-segmentation-pipeline

# Start the full stack (Airflow + MLflow + PostgreSQL)
docker compose up -d

# Wait ~60s for services to initialise, then open:
# Airflow:  http://localhost:8080  (admin / admin)
# MLflow:   http://localhost:5001
```

Enable and trigger both DAGs in the Airflow UI:
- `customer_segmentation_training`
- `customer_segmentation_stability_monitoring`

### Option B — Local (conda)

```bash
conda create -n seg_pipeline python=3.10 -y
conda activate seg_pipeline

pip install -r requirements.txt

# Terminal 1 — MLflow server
mlflow server --host 127.0.0.1 --port 5001

# Terminal 2 — Airflow
export AIRFLOW_HOME=$(pwd)
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001

airflow db init
airflow users create --username admin --password admin \
  --firstname Admin --lastname User --role Admin --email admin@example.com
airflow scheduler &
airflow webserver --port 8080
```

---

## MLflow Experiments

### `Customer_Segmentation_RFM` (Training)
Tracked per monthly run:
- **Params**: n_clusters, data diagnostics, RFM stats
- **Metrics**: KMeans silhouette, GMM silhouette, per-cluster mean RFM
- **Artifacts**: elbow plot, PCA/t-SNE visualisations, KMeans centroids (`.npy`)
- **Model**: KMeans model registered as `CustomerSegmentationKMeans`

### `Customer_Segmentation_Stability` (Monitoring)
Tracked per weekly run:
- **Metrics**: ARI, centroid shift (mean + max), silhouette (current + drop)
- **Params**: ARI status, centroid shift status, silhouette status, overall status
- **Artifacts**: stability history trend plot, stability JSON report

---

## Project Structure

```
customer-segmentation-pipeline/
├── dags/
│   ├── customer_segmentation_training_dag.py      # Monthly training DAG
│   ├── customer_segmentation_stability_dag.py     # Weekly stability DAG
│   └── pipeline/                                  # Modular components
│       ├── __init__.py
│       ├── data_loader.py                         # Dataset loading
│       ├── rfm_engineering.py                     # RFM feature computation
│       ├── clustering.py                          # KMeans, GMM, PCA, t-SNE
│       ├── stability_monitor.py                   # ARI, centroid shift, drift
│       └── mlflow_logger.py                       # MLflow logging helpers
├── data/                                          # (gitignored) local CSVs
├── output/                                        # Pipeline artifacts
├── docker-compose.yml                             # Airflow + MLflow + Postgres
├── requirements.txt
└── README.md
```

---

## Marketing Strategy Recommendations (from clusters)

| Cluster | Profile | Recommended Action |
|---|---|---|
| 0 | High Recency, Low Frequency | Win-back campaigns |
| 1 | Low Recency, High Frequency | Loyalty reward programs |
| 2 | High Monetary, High Frequency | VIP exclusive offers |
| 3 | Low Monetary, High Recency | Discount-focused campaigns |

---

## Design Decisions

**Parquet for XCom** — Airflow's XCom limit is 48KB. All DataFrames are serialised to Parquet and only paths passed via XCom.

**Stability bootstrapping** — On first run of the stability DAG, no reference model exists. The DAG auto-fits a reference model from the current data to bootstrap the comparison history.

**Parallel stability checks** — ARI, centroid shift, silhouette, and profile drift run in parallel Airflow tasks and converge at `build_stability_report`.

**Configurable thresholds** — All warning/critical thresholds live in `pipeline/stability_monitor.py::THRESHOLDS` and can be tuned per-environment via that dict.

---

## Dataset

[Give Me Some Credit — Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit) / Online Retail UCI dataset loaded automatically from GitHub.

---

## Author

Built on top of the Customer Segmentation for Marketing notebook, extended with full MLOps instrumentation and the cluster stability monitoring framework.
