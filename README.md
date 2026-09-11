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

- [ ] **Phase 1** — Data pipeline & regridding
- [ ] **Phase 2** — Encoder–decoder training
- [ ] **Phase 3** — ARGO validation & uncertainty calibration
- [ ] **Phase 4** — Dashboard integration

---

## 📦 Project Structure

```
ocean-embed/
├── data/            # raw/processed GLORYS, ARGO (gitignored)
├── src/
│   ├── data/        # download, regrid, preprocessing
│   ├── models/      # encoder/decoder versions (v0 → v5)
│   ├── train.py
│   └── eval.py
├── notebooks/       # exploration
├── dashboard/        # FastAPI + Streamlit/React demo
├── configs/          # per-experiment yaml configs
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

# Install Phase 1 dependencies
pip install numpy pyyaml

# Phase 1: Generate deterministic demo sample dataset
python src/data/prepare.py --config configs/bay_of_bengal.yaml --demo-sample
```

> [!WARNING]
> **Synthetic Development Fixture Notice**:
> The `--demo-sample` mode creates a deterministic, physically-plausible mathematical approximation of surface-subsurface relationships in `data/processed/bay_of_bengal.npz` (2,000 samples across 15 standard depths). It is **not** real satellite or in-situ ocean observations and must **not** be claimed as scientifically validated GLORYS/ARGO measurements.

Once deep learning dependencies (`torch`) are installed:
```bash
# Train v0 MSE baseline
python src/train.py --config configs/bay_of_bengal.yaml --model_version v0

# Train v1 Gaussian uncertainty baseline
python src/train.py --config configs/bay_of_bengal.yaml --model_version v1_uncertainty
```

---

## 🎯 Impact

Marine heatwave monitoring · Upper-ocean heat content · Stratification assessment · Fisheries & ocean-resource planning · Data-assimilation support

> *Supports situational awareness for extreme-weather and marine-risk assessment — not a substitute for direct disaster prediction.*

---

<div align="center">

**Built with 🌊 by Team Bug Dealers**

*Problem Statement: Disaster Management · Software*

</div>
