<div align="center">

# 🌊 OceanEmbed

### *Seeing the ocean's depths through its surface.*

**Satellite-Embedding-Based Deep Learning for Subsurface Ocean Temperature Reconstruction**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

*Smart India Hackathon (SIH 2026) · Problem Statement ID: 26066 · Team: Bug Dealers · Theme: Disaster Management · Software Category*

</div>

---

## 1. 🌀 Project Overview

The ocean conceals its most critical dynamics beneath the surface. While sea surface temperature (SST) is easily captured by satellite radiometers, upper-ocean heat content (OHC), thermocline depth, and barrier layers—the fundamental drivers of rapid tropical cyclone intensification and marine heatwaves—reside at depth. Direct observations via autonomous ARGO profiling floats are sparse, slow to sample (10-day cycle), and leave massive spatio-temporal observation voids.

**OceanEmbed** solves this challenge by leveraging continuous satellite surface observation sequences. Instead of predicting depth temperature as an isolated point regression, OceanEmbed learns a compact, physically structured **512-dimensional continuous latent ocean embedding** from daily surface fields (SST, SSS, SLA, Wind-U, Wind-V) across a 31-day temporal window and 32×32 spatial neighborhood. This representation is decoded into a continuous vertical temperature profile across **15 standardized ocean depths (0 to 1000 m)**, accompanied by rigorous **Gaussian predictive uncertainty ($\pm 1.645\sigma$, 90% confidence envelope)** and soft latent regime context ($K=4$).

```
   ☀️  Satellite Surface Observations (SST, SSS, SLA, Wind-U, Wind-V)
          │  Shape: (B, T=31, C=5, H=32, W=32)
          ▼
   🧠  Ocean Spatial Feature Extractor (2D CNN)
          │
          ▼
   ⏳  Spatiotemporal Dynamics (ConvLSTM + CBAM Attention)
          │
          ▼
   🌐  512-D Ocean Embedding  ←── Reusable Physical Latent Representation
          │
          ├────────────────────────────────┬──────────────────────────────┐
          ▼                                ▼                              ▼
   🔬  Depth Decoder             🌐  K=4 Latent Regimes         📉  Uncertainty Head
          │                                                               │
          └───────────────────────────────┬───────────────────────────────┘
                                          ▼
                      📊  15-Depth Profile: Mean T(z) ± 1.645σ
```

---

## 2. 🏆 SIH Problem Statement

- **Hackathon:** Smart India Hackathon 2026
- **Problem Statement ID:** 26066
- **Theme:** Disaster Management
- **Category:** Software
- **Team Name:** Bug Dealers
- **Mission:** Deliver an operational software system capable of reconstructing subsurface thermal structure in the Bay of Bengal (5°N–23°N, 80°E–100°E) to enhance cyclone intensification warning, marine disaster situational awareness, and extreme event monitoring.

---

## 3. 🏗️ System Architecture

OceanEmbed is engineered as an end-to-end modular pipeline consisting of:
1. **Data Ingestion Layer**: Reproducible acquisition scripts for Copernicus Marine (CMEMS) satellite rasters and in-situ ARGO float NetCDF profiles, paired with a deterministic synthetic simulation pipeline.
2. **Preprocessing & Regridding Engine**: Adapters converting multi-source rasters or synthetic fixtures into standardized 5D tensor sequences `(B, 31, 5, 32, 32)`.
3. **Core Deep Learning Engine (OceanEmbed V2)**: Spatiotemporal ConvLSTM with CBAM attention, 512-D bottleneck, and depth-conditioned continuous decoder.
4. **Production FastAPI Serving Layer**: High-performance RESTful API with Pydantic validation, singleton model lifecycle, and explicit `data_mode` routing.
5. **Interactive Scientific Dashboard**: Streamlit frontend featuring an interactive Leaflet/Folium map of the Bay of Bengal, vertical Plotly profile plots, uncertainty ribbons, latent regime context, and CSV data export.

---

## 4. 🔬 OceanEmbed V2 Neural Specifications

| Component | Specification | Description |
|---|---|---|
| **Input Tensor** | `(B, 31, 5, 32, 32)` | 31 consecutive daily surface fields across 5 channels on a 32×32 grid (~0.25° resolution, ~8°×8° patch). |
| **Surface Channels** | 5 physical variables | Sea Surface Temperature (SST), Sea Surface Salinity (SSS), Sea Level Anomaly (SLA), Eastward Wind (Wind-U), Northward Wind (Wind-V). |
| **Spatial Encoder** | 2D CNN (32 → 64 filters) | Convolutional layers extracting local spatial gradients, frontal boundaries, and mesoscale eddy signatures. |
| **Temporal Encoder** | Bidirectional ConvLSTM | Models 31-day synoptic and seasonal memory, atmospheric forcing lags, and upper-ocean mixing timescales. |
| **Attention Mechanism** | CBAM (Channel + Spatial) | Sequentially applies Channel Attention (inter-variable coupling) and Spatial Attention (focusing on mesoscale eddy cores). |
| **Latent Bottleneck** | **512-dimensional vector** | Dense continuous representation of the localized 3D ocean state. |
| **Latent Regime Head** | Softmax ($K = 4$ classes) | Unsupervised soft probability distribution over 4 distinct hydrographic regimes in the Bay of Bengal. |
| **Climatology Integration** | Residual Formulation | $T(z) = T_{\text{clim}}(z) + \Delta T(z)$, stabilizing deep ocean predictions. |
| **Depth Decoder** | Continuous MLP + FiLM | Predicts temperature anomalies across depth levels conditioned on the 512-D embedding and target depth values. |
| **Uncertainty Head** | Gaussian Log-Variance $\log\sigma^2$ | Outputs depth-wise predictive uncertainty, optimized via Gaussian Negative Log-Likelihood (GNLL). |
| **Target Depths** | 15 standard depths | `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` meters. |
| **Total Parameters** | 337,896 parameters | Compact, memory-efficient (~1.4 MB checkpoint), suitable for edge deployment and fast CPU serving. |

---

## 5. 🌊 Data Pipeline & Input Formulation

The model operates on a spatiotemporal neighborhood centered at target latitude $\phi_0$ and longitude $\lambda_0$:
- **Spatial Grid:** 32×32 points spaced at $\Delta = 0.25^\circ$, spanning approximately $7.75^\circ \times 7.75^\circ$.
- **Temporal Sequence:** 31 consecutive daily time steps centered on the target date ($t_{-15}, \dots, t_0, \dots, t_{+15}$).
- **Standard Normalization:** Input channels are normalized using stored training dataset statistics:
  $$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

---

## 6. 🧪 Deterministic Synthetic Simulation Mode

OceanEmbed provides a deterministic synthetic dataset generator (`src/data/generate_spatiotemporal_demo.py`) that executes 100% offline without external network or API keys:
- Synthesizes realistic Bay of Bengal oceanographic phenomena: latitudinal thermal gradients ($29.5^\circ\text{C}$ to $26.5^\circ\text{C}$), river discharge salinity freshening in the northern bay ($<32\text{ PSU}$), mesoscale cyclonic and anticyclonic eddies (SLA anomalies $\pm 0.15\text{ m}$), and seasonal monsoon wind reversals.
- Seeded deterministically using location coordinates and Julian day of year, ensuring absolute reproducibility across tests and demonstrations.

---

## 7. 🎯 Real ARGO In-Situ Observation Pipeline

- **Acquisition Script:** `scripts/download_argo.py`
  - Targets the Bay of Bengal bounding box (5°N–23°N, 80°E–100°E).
  - Integrates with `argopy` and GDAC ERDDAP endpoints.
  - Configurable date range, output directory (`data/raw/argo/`), and profile count limit.
- **Ingestion & Validation Engine:** `src/api/services/argo_service.py`
  - Inspects all `.nc` files in `data/raw/argo/`.
  - Strictly validates presence of: `latitude`, `longitude`, `time`, `depth/pressure`, and `temperature`.
  - Rejects corrupt or incomplete profiles safely.
  - Normalizes profiles for independent validation and dashboard query.
  - **Graceful Offline Status:** If no real ARGO files exist, honestly reports `available: False`, `files_found: 0`, and disables in-situ overlays without fabricating fake observations.

---

## 8. 🛰️ Copernicus Marine (CMEMS) Satellite Pipeline

- **Acquisition Script:** `scripts/download_cmems.py`
  - Automates subset downloading via the `copernicusmarine` API.
  - Queries physical variables: `thetao` (SST), `so` (SSS), `zos` (SLA), `usi`/`vsi` (Wind-U/V).
  - Uses environment variables `CMEMS_USERNAME` and `CMEMS_PASSWORD` (never hardcoded or committed).
  - Configurable temporal range and output directory (`data/raw/satellite/`).
- **Real-Data Preprocessing Adapter:** `src/data/real_satellite_preprocessing.py`
  - Crops regional rasters to 32×32 spatial patches around target coordinates.
  - Harmonizes variable aliases across different satellite providers (e.g. `thetao`, `analysed_sst`, `sst`).
  - Aligns longitude coordinates ($0\dots360^\circ \to -180\dots180^\circ$).
  - Implements a documented missing-value strategy (temporal linear interpolation + spatial nearest-neighbor fill, failing explicitly if missing ratio exceeds 40%).
- **Current Operational State:** `CMEMS: READY FOR REAL-DATA CONFIGURATION`.

---

## 9. ⚡ FastAPI Production Serving & REST Endpoints

The API is served with **FastAPI** (`src/api/main.py`) with full OpenAPI documentation at `/docs`:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | `GET` | API root metadata, version, and navigation links. |
| `/health` | `GET` | System health, model availability, and honest data mode reporting. |
| `/model/status` | `GET` | Inspection of checkpoint metadata, parameter counts, and depth levels. |
| `/predict` | `POST` | Primary inference endpoint for subsurface reconstruction. |
| `/data/argo/status` | `GET` | Ingestion status of real ARGO float NetCDF files. |
| `/data/argo/profiles` | `GET` | Query ingested ARGO profiles within radius of given coordinate. |
| `/data/satellite/status` | `GET` | Satellite provider status (CMEMS vs synthetic demo). |

---

## 10. 📊 Interactive Scientific Dashboard

Built with **Streamlit** (`dashboard/app.py`), the dashboard offers an intuitive interface for operational oceanographers:
- **Geographic Map:** Interactive Folium map centered on the Bay of Bengal with a highlighted domain bounding box. Click directly on the ocean to select coordinates.
- **Data Source Control:** Explicit radio toggle: `Synthetic Demo (Active)` vs `Real Satellite (Not Configured / Disabled)`.
- **Honest Mode Badges:** Prominently displays:
  - `DEMO MODE — SYNTHETIC DATA` (when running synthetic simulation), or
  - `REAL SATELLITE INPUT — MODEL TRAINED ON SYNTHETIC DEVELOPMENT DATA` (when running real rasters).
- **Vertical Profile Chart:** Interactive Plotly vertical curve with an inverted depth axis ($0\dots1000\text{ m}$) and a shaded 90% Gaussian predictive interval ribbon ($\pm 1.645\sigma$).
- **Climatology Residual Decomposition:** Inspect background climatology prior vs predicted anomaly $\Delta T(z)$.
- **Regime Context:** Soft probability distribution bar chart across $K=4$ latent regimes.
- **512-D Latent State:** Metrics and vector inspection.
- **CSV Data Export:** One-click download of profile table (`depth_m`, `temperature_c`, `sigma_c`, `lower_90_c`, `upper_90_c`).
- **ARGO Verification Card:** Displays nearest ARGO float profile when real NetCDF data is present; cleanly displays `STATUS: NOT AVAILABLE` when unconfigured.

---

## 11. 🐳 Containerized Deployment (Docker & Compose)

```dockerfile
# Production multi-stage Docker build for CPU serving
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Run container:
```bash
docker build -t oceanembed-api .
docker run -d -p 8000:8000 --name oceanembed-api oceanembed-api
# Or via Compose:
docker-compose up -d
```

*(Note: Docker CLI is not installed on the current development host; Docker runtime testing is reported as SKIPPED — Docker CLI unavailable).*

---

## 12. 💻 Installation & Setup

### Requirements
- **OS:** Linux, macOS (Apple Silicon supported), Windows
- **Python:** 3.10 to 3.12 (Python 3.11 recommended)

### Quick Setup
```bash
# Clone repository
git clone https://github.com/sylbornfurtado19/ocean-embed.git
cd ocean-embed

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate synthetic development dataset
python src/data/generate_spatiotemporal_demo.py --samples 800 --seed 42

# Train OceanEmbed V2 model (takes ~2 minutes on CPU)
python src/train_oceanembed.py --epochs 3 --batch_size 16
```

---

## 13. ⚙️ Configuration & Environment (`.env`)

Copy `.env.example` to `.env`:
```bash
OCEAN_DATA_MODE=synthetic                  # 'synthetic' or 'real'
OCEANEMBED_CHECKPOINT=checkpoints/oceanembed_v2.pt
ARGO_DATA_DIR=data/raw/argo
SATELLITE_DATA_DIR=data/raw/satellite
CMEMS_USERNAME=                            # Copernicus Marine user (optional)
CMEMS_PASSWORD=                            # Copernicus Marine password (optional)
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

---

## 14. 🧪 Automated Testing Suite (58 Tests)

OceanEmbed features 58 automated tests passing across three comprehensive test suites:
```bash
pytest -v
```

```
tests/test_phase4_dashboard.py ........ [21 passed]
tests/test_phase5_api.py .............. [22 passed]
tests/test_phase6_robustness.py ....... [15 passed]
======================= 58 passed in 31.29s =======================
```

---

## 15. 📡 Example API Requests & Responses

### Synthetic Prediction Request
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "latitude": 14.0,
       "longitude": 88.0,
       "date": "2023-02-15",
       "data_mode": "synthetic"
     }'
```

### Response
```json
{
  "latitude": 14.0,
  "longitude": 88.0,
  "date": "2023-02-15",
  "depths": [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0],
  "temperatures": [28.7564, 27.691, 26.5223, 24.1645, 21.9418, 18.0167, 14.1902, 11.4721, 9.5287, 8.1635, 6.5554, 5.509, 5.3915, 5.4816, 5.586],
  "sigma": [0.2108, 0.2185, 0.2246, 0.2335, 0.2398, 0.2489, 0.2506, 0.2346, 0.2171, 0.2014, 0.1786, 0.1504, 0.1212, 0.1056, 0.091],
  "lower_90": [28.4096, 27.3316, 26.1528, 23.7804, 21.5473, 17.6073, 13.7779, 11.0862, 9.1716, 7.8322, 6.2616, 5.2616, 5.1921, 5.3079, 5.4363],
  "upper_90": [29.1032, 28.0504, 26.8918, 24.5486, 22.3363, 18.4261, 14.6025, 11.858, 9.8858, 8.4948, 6.8492, 5.7564, 5.5909, 5.6553, 5.7357],
  "regime_probs": [0.1444, 0.1652, 0.3561, 0.3343],
  "embedding": [0.1205, -0.0412, "... 512 dimensions ..."],
  "inference_latency_ms": 12.31,
  "data_mode": "synthetic_demo"
}
```

### Strict Real-Mode Enforcement (No Silent Fallback)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "latitude": 14.0,
       "longitude": 88.0,
       "date": "2023-02-15",
       "data_mode": "real"
     }'
```
Returns **HTTP 503 Service Unavailable**:
```json
{
  "detail": "Real satellite data pipeline is not configured. No satellite raster files were found in data/raw/satellite. In adherence to SIH scientific honesty rules, real-mode inference cannot proceed without verified observations.",
  "error_code": "HTTP_503",
  "timestamp": "2026-09-11T14:45:00.000000Z"
}
```

---

## 16. ⚠️ Scientific Limitations

1. **Training Distribution**: The current neural weights were trained on a deterministic synthetic spatiotemporal simulation. They must not be claimed as operational GLORYS reanalysis predictions.
2. **Missing Real In-Situ Tuning**: Because real ARGO profiles were not ingested into training, real-world operational accuracy has not yet been established.
3. **Domain Boundary**: Validated strictly for the Bay of Bengal domain (5°N–23°N, 80°E–100°E). Out-of-bounds predictions are rejected.
4. **Resolution**: Operates on a ~0.25° horizontal spatial grid (~28 km). Sub-mesoscale phenomena (<10 km) cannot be resolved.

---

## 17. 🔍 Current Real-Data Status & Readiness

- **Real ARGO Acquisition:** Script implemented (`scripts/download_argo.py`). Status: `UNAVAILABLE (argopy/network awaiting live deployment)`.
- **Real ARGO Ingestion & Validation:** Implemented and verified in `src/api/services/argo_service.py`.
- **Real CMEMS Acquisition:** Script implemented (`scripts/download_cmems.py`). Status: `READY FOR REAL-DATA CONFIGURATION (credentials required)`.
- **Real Satellite Preprocessing Adapter:** Implemented and unit-tested in `src/data/real_satellite_preprocessing.py`.
- **Operating Mode:** `synthetic_demo` (Active, verified, and 100% reproducible offline).

---

## 18. 🔬 Scientific Validation Status

> [!IMPORTANT]
> **Strict Scientific Transparency Notice**:
> - **Synthetic Development Validation:** ✅ **VERIFIED**. OceanEmbed V2 achieves validation GNLL of -1.0259 on synthetic held-out spatial patches. Uncertainty intervals are mathematically consistent with Gaussian confidence bounds.
> - **Real Satellite Observation Ingestion:** ⏳ **READY FOR REAL-DATA CONFIGURATION**. Downloader and preprocessing adapter are implemented, tested, and ready to ingest rasters once credentials and live feeds are attached.
> - **Real ARGO Float Ingestion:** ⏳ **READY FOR REAL-DATA CONFIGURATION**. NetCDF discovery, profile validation, and coordinate matching are implemented. Real-world validation metrics are cleanly withheld until authentic floats are ingested.
> - **Real-World Predictive Accuracy:** ❌ **NOT ESTABLISHED**. The model has not been fine-tuned on real observations; no claims of real-time operational ocean forecast skill are made.

---

## 19. ⚡ Performance Benchmark Results

Measured on macOS (Apple Silicon host CPU execution) using `scripts/benchmark.py`:

| Metric | Measured Value | Note |
|---|---|---|
| **Model Load Time** | **443.15 ms** | Checkpoint loaded from disk to RAM. |
| **Pure Model Inference Latency (Mean)** | **12.31 ms** | Min: 11.00 ms, Max: 14.08 ms (15 runs). |
| **Full HTTP Request Latency (Mean)** | **14.16 ms** | Min: 13.06 ms, Max: 15.48 ms (15 runs). |
| **FastAPI / Serialization Overhead** | **1.85 ms** | Pydantic validation + HTTP routing. |
| **Memory Footprint (RSS)** | **195.36 MB** | Lightweight memory footprint. |

---

## 20. 🏆 SIH Demonstration Workflow

1. **Launch API Backend**:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```
2. **Launch Interactive Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```
3. **Walkthrough for SIH Judges**:
   - **Step 1:** Observe the `DEMO MODE — SYNTHETIC DATA` badge in the header, demonstrating scientific honesty.
   - **Step 2:** Click a location on the Bay of Bengal map (e.g. 14.0°N, 88.0°E).
   - **Step 3:** Review coordinate boundaries (5°N–23°N, 80°E–100°E).
   - **Step 4:** Click **RECONSTRUCT PROFILE**. The model executes forward inference in ~12 ms on CPU.
   - **Step 5:** Inspect the 15-depth temperature curve with its downward depth axis (0 to 1000 m).
   - **Step 6:** Inspect the shaded 90% Gaussian predictive uncertainty interval ($\pm 1.645\sigma$).
   - **Step 7:** Examine the $K=4$ latent regime context distribution.
   - **Step 8:** Inspect the Climatology Residual Decomposition ($\Delta T(z)$ vs $T_{\text{clim}}(z)$).
   - **Step 9:** Inspect the continuous 512-D Ocean Embedding representation.
   - **Step 10:** Click **Download Profile CSV** to export reconstructed data.
   - **Step 11:** Show the ARGO Verification card with honest `STATUS: NOT AVAILABLE` notice.
   - **Step 12:** Open OpenAPI documentation at `http://localhost:8000/docs` and send a test request.
   - **Step 13:** Demonstrate strict error handling by requesting `data_mode="real"` and observing clean HTTP 503 rejection.

---

<div align="center">

**Built with 🌊 by Team Bug Dealers**  
*Problem Statement: Disaster Management (ID: 26066) · Software*

</div>
