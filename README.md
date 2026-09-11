<div align="center">

# 🌊 OceanEmbed

### *Seeing the ocean's depths through its surface.*

**Satellite-Embedding-Based Deep Learning for Subsurface Ocean Temperature Reconstruction**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

*Team Bug Dealers · Disaster Management Theme · Software Category*

</div>

---

## 🌀 What is this?

The ocean doesn't hide its secrets on the surface — it hides them **beneath** it. Sea surface temperature tells only half the story; the real signal for marine heatwaves, cyclone intensification, and ocean heat content lives at depth, where direct measurements (ARGO floats) are sparse, slow, and expensive to gather.

**OceanEmbed** flips the problem: instead of predicting temperature directly, it learns a compact, reusable **512-dimensional ocean-state embedding** from daily satellite surface fields — SST, SSS, SSH/SLA, and wind — and decodes that embedding into a full **15-depth temperature profile, complete with uncertainty**.

No buoy required. No waiting for a float to surface. Just satellites, and a model that's learned what the ocean is *doing* underneath.

```
   ☀️  Satellite Surface Data
         │
         ▼
   🧠  Ocean Encoder + Attention
         │
         ▼
   🌐  512-D Ocean Embedding  ←── the reusable part
         │
         ▼
   🔬  Depth-Aware Decoder
         │
         ▼
   📊  15-Depth Temperature + Uncertainty
```

---

## ⚡ Why it's different

| | |
|---|---|
| 🧬 **Reusable, not one-off** | The embedding isn't just a regression byproduct — it's a representation other downstream tasks can plug into. |
| 🗺️ **Regime-aware, no labels** | Learns Arabian Sea vs. Bay of Bengal ocean-state context implicitly — nobody hand-coded a region flag. |
| 🎯 **Uncertainty, not just a number** | Every depth gets a mean *and* a confidence — because knowing what the model doesn't know matters. |
| 🕵️ **Leak-proof validation** | Spatial / temporal / spatiotemporal holdouts, not a random split pretending to be trustworthy. |

---

## 🏗️ Architecture

```
Multi-source Surface Data → Ocean Encoder (CNN/ConvLSTM) → CBAM Attention
        → 512-D Ocean Embedding → Regime Context + Climatology Prior
        → Depth-Conditioned Decoder → 15-Depth Temp + Uncertainty Head
```

**Inputs:** SST · SSS · SSH/SLA · Wind-U/V · Lat/Lon/Time
**Training target:** GLORYS reanalysis (0.25° grid, 15 depths)
**Validation:** Independent ARGO float profiles

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `xarray` / `NumPy` · `NetCDF/Zarr` · `FastAPI` · `React/Next.js` · `Streamlit` · `Docker`

---

## 🗺️ Development Roadmap

- [x] **Phase 1** — Data pipeline & regridding
- [x] **Phase 2** — Encoder–decoder training & uncertainty baseline
- [x] **Phase 3** — Core OceanEmbed architecture (CNN-ConvLSTM-CBAM + 512-D Embedding)
- [x] **Phase 4** — Interactive scientific dashboard & inference integration
- [x] **Phase 5** — Production FastAPI backend, Docker, ARGO/satellite ingestion layers & automated tests

---

## 📦 Project Structure

```
ocean-embed/
├── data/            # raw/processed GLORYS, ARGO NetCDF (gitignored)
├── src/
│   ├── config.py    # centralized environment configuration
│   ├── data/        # download, regrid, preprocessing, demo spatiotemporal generator
│   ├── models/      # baseline & OceanEmbed V2 architectures
│   ├── api/         # FastAPI production serving layer
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── dependencies.py
│   │   ├── routes/  # health, inference, data routes
│   │   └── services/# model, satellite, and argo services
│   ├── train.py     # baseline training
│   ├── train_oceanembed.py # OceanEmbed V2 training
│   ├── eval.py      # canonical evaluation entrypoint
│   └── eval_pipeline.py # depth-wise metrics, calibration, benchmark comparison
├── dashboard/       # Streamlit interactive scientific demonstration
│   ├── app.py       # main dashboard entrypoint
│   └── components/  # modular map, plots, metrics, regime, profile table
├── tests/           # automated test suites (Phase 4 dashboard + Phase 5 API)
├── configs/         # per-experiment yaml configs
├── Dockerfile       # containerized production deployment
├── docker-compose.yml
├── .env.example     # environment configuration template
└── requirements.txt
```

---

## 🚀 Quickstart

### Supported Environment
- **Python**: 3.10 to 3.12 recommended. (Windows note: enable Long Paths in Registry or use short paths if installing PyTorch).

```bash
git clone https://github.com/sylbornfurtado19/ocean-embed.git
cd ocean-embed

# Create and activate virtual environment
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Phase 1: Generate deterministic demo sample dataset
python src/data/prepare.py --config configs/bay_of_bengal.yaml --demo-sample
```

> [!WARNING]
> **Synthetic Development Fixture Notice**:
> The `--demo-sample` mode creates a deterministic, physically-plausible mathematical approximation of surface-subsurface relationships in `data/processed/bay_of_bengal.npz` (2,000 samples across 15 standard depths). It is **not** real satellite or in-situ ocean observations and must **not** be claimed as scientifically validated GLORYS/ARGO measurements.

```bash
# Phase 2: Baseline Model Training
# Train v0 MSE baseline (saves to checkpoints/v0_baseline.pt)
python src/train.py --config configs/bay_of_bengal.yaml --model_version v0

# Train v1 Gaussian uncertainty baseline (saves to checkpoints/v1_uncertainty.pt)
python src/train.py --config configs/bay_of_bengal.yaml --model_version v1_uncertainty

# Phase 3: Core OceanEmbed Architecture (Spatiotemporal ConvLSTM + CBAM + 512-D Embedding)
# 1. Generate 31-day spatiotemporal patches (B, 31, 5, 32, 32)
python src/data/generate_spatiotemporal_demo.py --samples 800 --seed 42

# 2. Train OceanEmbed V2 (saves to checkpoints/oceanembed_v2.pt)
python src/train_oceanembed.py --epochs 3 --batch_size 16

# 3. Run canonical evaluation and baseline comparison
python src/eval.py --model_version oceanembed_v2
python src/eval.py --model_version compare_all

# Phase 4: Interactive Scientific Dashboard
streamlit run dashboard/app.py
```

### Phase 4 — Interactive Scientific Dashboard

**Dashboard Mode: SYNTHETIC DEMO DATA**
*Current predictions are generated from deterministic synthetic development data for software verification and prototype presentation. They do not represent live satellite observations or operational forecasts.*

- **Dashboard Controls**:
  - **Bay of Bengal Domain**: Centered on 5°N–23°N, 80°E–100°E.
  - **Map Selection**: Click directly inside the Bay of Bengal domain to select coordinates or manually edit numeric inputs.
  - **Date Selector**: Central prediction date converts into a 31-day spatiotemporal window (day -15 to day +15).
  - **Reconstruct Profile Button**: Runs cached OceanEmbed V2 inference on the deterministic 31-day patch.
  - **Temperature vs Depth Profile**: Interactive Plotly vertical profile with inverted depth axis (0 to 1000 m across 15 official depths).
  - **Uncertainty Envelope**: 90% Gaussian predictive interval ($\pm 1.645\sigma$) with depth-dependent spread.
  - **Latent Regime Context**: Differentiable probability distribution across $K=4$ soft latent regimes.
  - **Climatology Decomposition**: Inspect synthetic climatology prior + predicted anomaly $\Delta T(z) = T(z)$.
  - **512-D Ocean Embedding**: Compact latent metrics (norm, mean, std) and inspection vector.
  - **CSV Export**: Download profile data table (`depth_m`, `temperature_c`, `sigma_c`, `lower_90_c`, `upper_90_c`).
  - **ARGO Verification Status**: Gracefully displays `STATUS: NOT AVAILABLE` when in-situ NetCDF files are absent, without fabricating fake observations.

### 🏆 SIH Demo Flow (Judging Workflow)

1. **Launch Dashboard**: Run `streamlit run dashboard/app.py`.
2. **Select Location**: Click an ocean point on the Bay of Bengal Folium map (e.g. 14.0°N, 88.0°E).
3. **Select Date**: Pre-populated to a valid demo date or choose a custom date.
4. **Click "RECONSTRUCT PROFILE"**: Executes cached OceanEmbed V2 inference in ~10–25 ms on CPU.
5. **Inspect 0–1000 m Profile**: View vertical temperature curve with downward depth axis across 15 official depths.
6. **Inspect Uncertainty**: Examine the shaded 90% Gaussian predictive interval ribbon ($\pm 1.645\sigma$).
7. **Inspect Regime Context**: View the horizontal bar chart showing learned $K=4$ latent regime probabilities.
8. **Inspect Ocean Embedding**: View the 512-D bottleneck representation.
9. **Export Data**: Click "Download Profile CSV" to download the reconstructed profile data.
10. **ARGO Status**: Observe the honest `STATUS: NOT AVAILABLE` indicator explaining that real in-situ ARGO profiles will be ingested in future operational phases.

```bash
# Phase 5: Production FastAPI Backend & Serving
# Start FastAPI server locally
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Run Phase 5 automated API tests (22/22 tests)
pytest tests/test_phase5_api.py -v
```

### Phase 5 — Production FastAPI Backend & Ingestion Architecture

OceanEmbed exposes a production-oriented RESTful backend built with **FastAPI** and containerized with **Docker**, designed to serve subsurface ocean reconstructions to downstream meteorological, cyclone monitoring, and marine-spatial systems.

> [!IMPORTANT]
> **Data Mode Integrity**:
> Real satellite and ARGO data must be configured separately; **synthetic demo mode remains the active default** for reproducible testing and hackathon demonstration. Endpoints report `synthetic_demo` honestly when real observational feeds are unconfigured.

- **FastAPI Endpoints**:
  - `GET /` — Root metadata and navigation.
  - `GET /health` — Service health, model availability, and real data integration status.
  - `GET /model/status` — Comprehensive metadata directly inspected from the loaded OceanEmbed V2 checkpoint (embedding dimensions, depth levels, domain bounds).
  - `POST /predict` — Core inference endpoint. Accepts `{"latitude": 14.5, "longitude": 88.0, "date": "2023-06-15"}` and returns 15-depth vertical temperatures, uncertainty ($\pm 1.645\sigma$), $K=4$ regime probabilities, and 512-D ocean embedding.
  - `GET /data/argo/status` — In-situ ARGO NetCDF discovery status in `data/raw/argo`.
  - `GET /data/satellite/status` — Satellite provider status (CMEMS provider adapter vs synthetic demo fixture).
  - `GET /data/argo/profiles` — In-situ profile search by coordinate and radius.
  - `GET /docs` & `GET /redoc` — Interactive OpenAPI / Swagger documentation.

- **Example `/predict` Request**:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "latitude": 14.0,
       "longitude": 88.0,
       "date": "2023-02-15"
     }'
```

- **Example Response**:
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
  "inference_latency_ms": 14.2,
  "data_mode": "synthetic_demo"
}
```

- **Containerized Deployment (Docker & Compose)**:
```bash
# Build and run Docker container on CPU
docker build -t oceanembed-api .
docker run -d -p 8000:8000 --name oceanembed-api oceanembed-api

# Or using Docker Compose
docker-compose up -d
```

- **Environment Configuration (`.env`)**:
Copy `.env.example` to `.env` to configure runtime settings:
```bash
OCEAN_DATA_MODE=synthetic                  # 'synthetic' or 'real'
OCEANEMBED_CHECKPOINT=checkpoints/oceanembed_v2.pt
ARGO_DATA_DIR=data/raw/argo
SATELLITE_DATA_DIR=data/raw/satellite
CMEMS_USERNAME=                            # Copernicus Marine user
CMEMS_PASSWORD=                            # Copernicus Marine password
API_HOST=0.0.0.0
API_PORT=8000
```

> [!NOTE]
> **Independent ARGO Validation Status**:
> Currently **NOT AVAILABLE**. Real ARGO NetCDF files are not yet downloaded in `data/raw/argo`. The evaluation engine gracefully skips in-situ verification without fabricating data. Real in-situ ARGO profiles and satellite grids will be integrated in subsequent operational phases.


---

## 🎯 Impact

Marine heatwave monitoring · Upper-ocean heat content · Stratification assessment · Fisheries & ocean-resource planning · Data-assimilation support

> *Supports situational awareness for extreme-weather and marine-risk assessment — not a substitute for direct disaster prediction.*

---

<div align="center">

**Built with 🌊 by Team Bug Dealers**

*Problem Statement: Disaster Management · Software*

</div>
