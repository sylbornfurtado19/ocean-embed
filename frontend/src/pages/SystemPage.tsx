import React from 'react';
import type { ArgoStatusResponse, HealthResponse, ModelStatusResponse, SatelliteStatusResponse } from '../types';
import { CpuIcon, DatabaseIcon, InfoIcon } from '../components/common/Icons';

interface SystemPageProps {
  health: HealthResponse | null;
  modelStatus: ModelStatusResponse | null;
  argoStatus: ArgoStatusResponse | null;
  satelliteStatus: SatelliteStatusResponse | null;
}

export const SystemPage: React.FC<SystemPageProps> = ({
  health,
  modelStatus,
  argoStatus,
  satelliteStatus,
}) => {
  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
      {/* Header */}
      <div className="compact-hero" style={{ marginBottom: '24px' }}>
        <div className="hero-main">
          <h1>System Architecture & Operational Runtime</h1>
          <p>
            Hardware deployment, neural model specifications, FastAPI endpoint contracts, and telemetry ingestion statuses.
          </p>
        </div>
        <div className="hero-specs">
          <div className="spec-item">
            <span className="spec-label">Backend API</span>
            <span className="spec-val">FastAPI 2.0.0</span>
          </div>
          <div className="spec-item">
            <span className="spec-label">Inference Target</span>
            <span className="spec-val">PyTorch CPU Runtime</span>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '24px' }}>
        {/* Model Architecture Card */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <CpuIcon size={16} style={{ color: '#0284c7' }} />
              <span>OceanEmbed V2 Architecture</span>
            </div>
            <span style={{ fontSize: '0.6875rem', fontWeight: 600, color: '#10b981', textTransform: 'uppercase' }}>
              Loaded
            </span>
          </div>
          <div className="card-body">
            <table className="data-table">
              <tbody>
                <tr>
                  <td style={{ color: '#64748b' }}>Model Version</td>
                  <td style={{ fontWeight: 600 }}>{modelStatus?.version || 'oceanembed_v2'}</td>
                </tr>
                <tr>
                  <td style={{ color: '#64748b' }}>Encoder Architecture</td>
                  <td>CNN + ConvLSTM + CBAM</td>
                </tr>
                <tr>
                  <td style={{ color: '#64748b' }}>Latent Embedding Dimension</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{modelStatus?.embedding_dim || 512}-D</td>
                </tr>
                <tr>
                  <td style={{ color: '#64748b' }}>Latent Mixture Regimes</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>K = {modelStatus?.num_regimes || 4}</td>
                </tr>
                <tr>
                  <td style={{ color: '#64748b' }}>Vertical Target Levels</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{modelStatus?.num_depths || 15} standard depths</td>
                </tr>
                <tr>
                  <td style={{ color: '#64748b' }}>Checkpoint File</td>
                  <td style={{ fontSize: '0.6875rem', wordBreak: 'break-all' }}>
                    {modelStatus?.checkpoint_path || 'checkpoints/oceanembed_v2.pt'}
                  </td>
                </tr>
                <tr>
                  <td style={{ color: '#64748b' }}>Checkpoint Size</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    {modelStatus?.checkpoint_size_mb.toFixed(2) || '2.84'} MB
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Observation Data Status */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <DatabaseIcon size={16} style={{ color: '#0284c7' }} />
              <span>Data Subsystems & Telemetry</span>
            </div>
          </div>
          <div className="card-body">
            <table className="data-table">
              <tbody>
                <tr>
                  <td style={{ color: '#64748b' }}>Active Data Mode</td>
                  <td style={{ fontWeight: 600, color: '#0284c7' }}>
                    {health?.mode === 'real' ? 'Real Satellite Observations' : 'Synthetic Development Data'}
                  </td>
                </tr>
                <tr>
                  <td style={{ color: '#64748b' }}>Satellite Provider</td>
                  <td>{satelliteStatus?.provider || 'Synthetic Simulation (Phase 1-6)'}</td>
                </tr>
                <tr>
                  <td style={{ color: '#64748b' }}>Surface Input Channels</td>
                  <td style={{ fontSize: '0.75rem' }}>
                    {(modelStatus?.surface_channels || ['sst', 'sss', 'sla', 'wind_u', 'wind_v', 'heat_flux']).join(', ')}
                  </td>
                </tr>
                <tr>
                  <td style={{ color: '#64748b' }}>In-Situ ARGO NetCDF Status</td>
                  <td>
                    <span style={{ color: argoStatus?.available ? '#10b981' : '#64748b', fontWeight: 600 }}>
                      {argoStatus?.available ? 'Available' : 'Not Configured'}
                    </span>
                  </td>
                </tr>
                <tr>
                  <td style={{ color: '#64748b' }}>ARGO Profiles Found</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{argoStatus?.profiles_available ?? 0}</td>
                </tr>
                <tr>
                  <td style={{ color: '#64748b' }}>ARGO Directory Path</td>
                  <td style={{ fontSize: '0.6875rem' }}>data/raw/argo/</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* API Endpoints Contract */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <InfoIcon size={16} style={{ color: '#0284c7' }} />
            <span>FastAPI Operational Endpoints</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Base URL: http://localhost:8000</div>
        </div>
        <div className="card-body" style={{ padding: 0 }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Method</th>
                <th>Endpoint Route</th>
                <th>Description</th>
                <th>Typical Latency</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><span style={{ color: '#15803d', fontWeight: 700 }}>GET</span></td>
                <td style={{ fontWeight: 600 }}>/health</td>
                <td>System and data subsystem availability check</td>
                <td>&lt; 2 ms</td>
              </tr>
              <tr>
                <td><span style={{ color: '#15803d', fontWeight: 700 }}>GET</span></td>
                <td style={{ fontWeight: 600 }}>/model/status</td>
                <td>Model checkpoint metadata, dimensions, and supported domain</td>
                <td>&lt; 3 ms</td>
              </tr>
              <tr>
                <td><span style={{ color: '#0284c7', fontWeight: 700 }}>POST</span></td>
                <td style={{ fontWeight: 600 }}>/predict</td>
                <td>Forward pass inference for 15-depth profile, intervals, and regimes</td>
                <td>12–25 ms (CPU)</td>
              </tr>
              <tr>
                <td><span style={{ color: '#15803d', fontWeight: 700 }}>GET</span></td>
                <td style={{ fontWeight: 600 }}>/data/argo/status</td>
                <td>Real ARGO NetCDF discovery and geographic coverage</td>
                <td>&lt; 5 ms</td>
              </tr>
              <tr>
                <td><span style={{ color: '#15803d', fontWeight: 700 }}>GET</span></td>
                <td style={{ fontWeight: 600 }}>/data/satellite/status</td>
                <td>Satellite observation provider status and variables</td>
                <td>&lt; 2 ms</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
