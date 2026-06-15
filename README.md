# Modeling Global Influenza Spread via Mobility Networks and Climate Factors

A big-data and machine-learning pipeline that models the global spread of seasonal influenza by fusing three heterogeneous data sources — **gridded climate data (ERA5)**, **epidemiological surveillance (WHO FluNet)**, and the **global air-travel network (OpenFlights)** — into a unified spatio-temporal forecasting system.

The project quantifies the lag between climate anomalies and flu outbreaks, ranks epidemic hubs, computes mobility-based transmission risk scores, detects regional outbreak communities, and forecasts case counts two weeks ahead (`t+2`). Results are served through an interactive **FluScope** dashboard.

---

## Table of Contents

- [Overview](#Overview)
- [Architecture](#architecture)
- [Data Sources](#data-sources)
- [Tech Stack](#tech-stack)
- [Project Implementation](#project-implementation)
- [Results](#results)
- [Limitations & Future Work](#limitations--future-work)
- [Credits](#credits)

---

## Overview

Seasonal influenza spread is driven largely by two forces: **climate conditions** (temperature, humidity, precipitation) and **human mobility** between countries. Most prior systems use intrinsic time-series models (ARIMA/SARIMA) on past cases alone, or ignore spatial mobility entirely. This project treats the problem as a spatio-temporal one and is built around **streaming and approximation algorithms** so it runs within modest resource limits (Kaggle, 16–30 GB RAM).

Key deliverables:

- **Quantified lag effect** — climate anomaly → flu outbreak delay.
- **Epidemic hub ranking** — most influential countries in the travel network.
- **Mobility-based risk score** — exposure based on neighbor cases weighted by flight routes.
- **Regional outbreak communities** — country clusters with correlated outbreak patterns.
- **Improved forecasts** — case-count prediction at `t+2` weeks.

---

## Architecture

The system is a four-module pipeline; each module's Parquet outputs feed the next.

```
ERA5 + FluNet  ──▶ [M1] Big-Data Ingestion (Spark MapReduce)
                        │  climate_weekly.parquet · flunet_clean.parquet
                        ▼
                   [M2] Climate Anomaly + MinHash/LSH/Bloom Filter
                        │  climate_anomaly_features.parquet
OpenFlights ──────▶ [M3] Mobility Graph (PageRank/TSPR, Louvain, dynamic risk)
                        │  graph_metrics.parquet · mobility_features.parquet · mobility_edges.parquet
                        ▼
                   [M4] Master Feature Table ──▶ Models (Ridge / GBM / A3T-GCN)
                        ▼
                   FluScope Dashboard (Streamlit)
```

### Module breakdown

| Module | Purpose | Core algorithms |
|--------|---------|-----------------|
| **M1 — Ingestion** | Aggregate ~125 GB of ERA5 grid data (2001–2025) to country×week; clean FluNet | Spark MapReduce, Count-Min Sketch, Reservoir Sampling, Country Raster `O(1)` lookup, area-weighted aggregation |
| **M2 — Climate Anomaly** | Cluster countries by climate-anomaly patterns; build a fast existence index | Z-score anomalies, k-shingling, MinHash (128 hashes), LSH banding (b=16, r=8), Bloom Filter |
| **M3 — Mobility Network** | Model the global air-travel graph; score dynamic epidemic risk | PageRank / Topic-Sensitive PageRank, Louvain, Girvan-Newman, weighted diffusion |
| **M4 — Modeling** | Forecast `t+2` cases; compare models on a shared time split | Ridge, Gradient Boosting / XGBoost, A3T-GCN (PyTorch Geometric Temporal); Welford online stats, Flajolet-Martin, Johnson-Lindenstrauss |

---

## Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| **ERA5** (Copernicus CDS) | Global gridded reanalysis: temperature, dew point, precipitation (~130 GB raw) | [Copernicus CDS](https://cds.climate.copernicus.eu/) |
| **WHO FluNet** | Weekly influenza-positive specimen counts per country | [WHO FluNet](https://www.who.int/tools/flunet) |
| **OpenFlights** | `airports.dat` + `routes.dat` — global commercial flight routes | [OpenFlights](https://github.com/jpatokal/openflights) |
| **Natural Earth** | Admin-0 country boundaries (50m) for rasterization | [Natural Earth](https://www.naturalearthdata.com/) |

All sources are normalized to a common reference frame: **Country (ISO3) × ISO Week**.

---

## Tech Stack

- **Big data / processing:** Apache Spark (PySpark), xarray, cfgrib, PyArrow (Parquet), pandas, NumPy
- **Geospatial:** geopandas, rasterio, pycountry, Natural Earth shapefiles
- **Graph / network:** NetworkX, SciPy sparse matrices
- **Machine learning:** scikit-learn (Ridge, Gradient Boosting), XGBoost, PyTorch, PyTorch Geometric Temporal (A3T-GCN)
- **Visualization / demo:** Matplotlib, Seaborn, Streamlit

---

## Project Implementation:
This project is builded on Kaggle with very limited hardware resources. Can be easily implemented by running each module (notebook) in order (1-4) (need to run module 1 multiple times if data involve many years)


## Results

Models are compared on a shared temporal split — **train < 2018 | val 2018–2019 | test ≥ 2020** — scored in raw case-count units.

| Model | Val RMSE | MAE | Test RMSE | R² | F1 |
|-------|---------:|----:|----------:|---:|---:|
| **ridge_enhanced** | 301.6 | 57.8 | 552.4 | **0.802** | 0.751 |
| gbm_enhanced (XGBoost) | 324.1 | 65.7 | 734.9 | 0.650 | **0.817** |
| gnn_a3tgcn (A3T-GCN) | 561.4 | 128.5 | 1034.8 | 0.263 | 0.598 |
| gbm_baseline | 804.7 | 157.9 | 1198.9 | 0.014 | 0.432 |
| ridge_baseline | 808.5 | 164.9 | 1208.2 | −0.002 | 0.426 |

**Highlights**

- The regularized linear model (`ridge_enhanced`) achieves the best regression metrics (R², MAE, RMSE), partly because it extrapolates smoothly through the post-2020 regime shift.
- XGBoost wins on outbreak **classification** (highest F1), where flexible decision boundaries help.
- A3T-GCN establishes a principled spatio-temporal baseline but underperforms here due to data sparsity and a static graph that conflicts with the COVID-era collapse in air travel.

**Data-processing milestone:** ERA5 (2001–2025, ~125 GB) compressed to compact Parquet files with aggregation error MAE ≈ 0.004 °C and correlation ≈ 0.999994.

---

## Limitations & Future Work

- **Data heterogeneity & noise** — FluNet is sparse for developing countries; the mobility graph is limited to commercial flights.
- **Outbreak counting** — Flajolet-Martin estimates carry minor error; consider more advanced methods.
- **Climate–flu link** — global linear correlation is weak; explore lagged cross-correlation and per-region analysis.
- **Modeling** — a dynamic mobility graph (tracking the 2020 collapse) and denser per-country reporting could let spatio-temporal GNNs surpass tabular models.
- **Productionization** — wrap the pipeline as an MLOps workflow that re-runs forecasts as new weekly data arrives.

---

## Credits

Built as a Mining of Massive Datasets project. Data courtesy of Copernicus (ERA5), WHO (FluNet), OpenFlights, and Natural Earth. Algorithmic foundations follow *Mining of Massive Datasets* (Leskovec, Rajaraman, Ullman).

> Contributors: Dang Tuan Phong, Nguyen The Khiem, Tran Minh Hieu and Nguyen Bach Hai Dang
