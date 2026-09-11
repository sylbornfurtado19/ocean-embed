import React from 'react';
import { ActivityIcon, CompassIcon, LayersIcon } from '../components/common/Icons';

export const OverviewPage: React.FC = () => {
  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto' }}>


      {/* Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '32px' }}>
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <ActivityIcon size={16} style={{ color: '#0284c7' }} />
              <span>The Subsurface Ocean Problem</span>
            </div>
          </div>
          <div className="card-body" style={{ fontSize: '0.875rem', color: '#475569', lineHeight: 1.6 }}>
            <p style={{ marginBottom: '12px' }}>
              Satellite altimeters and radiometers observe only the skin and upper millimetres of the ocean surface (SST, SSS, SLA). However, extreme weather phenomena—including severe cyclonic storms in the Bay of Bengal—are governed by subsurface thermal energy reservoirs, specifically Tropical Cyclone Heat Potential (TCHP) down to the 26°C isotherm.
            </p>
            <p>
              Traditional in-situ CTD casts and autonomous profiling floats (ARGO) provide direct measurements but remain sparse in space and time. OceanEmbed bridges this gap by inferring vertical 3D thermal structure (0–1000 m) directly from 2D satellite surface sequences.
            </p>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <LayersIcon size={16} style={{ color: '#0284c7' }} />
              <span>OceanEmbed V2 Architecture</span>
            </div>
          </div>
          <div className="card-body" style={{ fontSize: '0.875rem', color: '#475569', lineHeight: 1.6 }}>
            <ul style={{ paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <li>
                <strong>Spatiotemporal Encoder:</strong> 2D CNN extracting multi-channel surface features coupled with ConvLSTM cells capturing temporal atmospheric-oceanic boundary memory across 31 daily frames.
              </li>
              <li>
                <strong>Spatial-Temporal CBAM Attention:</strong> Dual-channel and spatial attention gates focusing on dynamic baroclinic eddies and frontal boundaries.
              </li>
              <li>
                <strong>Continuous 512-D Embedding:</strong> Dense latent representation encoding oceanographic state.
              </li>
              <li>
                <strong>K=4 Latent Regime Conditioner:</strong> Soft Gaussian Mixture clustering guiding the depth-conditioned decoder.
              </li>
              <li>
                <strong>Calibrated Uncertainty Head:</strong> Predicts local standard deviation σ for honest 90% Gaussian intervals (±1.645σ).
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Target Depths Table */}
      <div className="card" style={{ marginBottom: '32px' }}>
        <div className="card-header">
          <div className="card-title">
            <CompassIcon size={16} style={{ color: '#0284c7' }} />
            <span>Standardized Vertical Depth Levels</span>
          </div>
          <div className="card-subtitle">
            15 canonical vertical levels covering epipelagic and mesopelagic zones
          </div>
        </div>
        <div className="card-body">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '12px', textAlign: 'center' }}>
            {[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000].map((d) => (
              <div
                key={d}
                style={{
                  padding: '12px',
                  backgroundColor: '#f8fafc',
                  border: '1px solid #e2e8f0',
                  borderRadius: '4px',
                }}
              >
                <div style={{ fontSize: '1.125rem', fontWeight: 700, color: '#0284c7', fontFamily: 'var(--font-mono)' }}>
                  {d} m
                </div>
                <div style={{ fontSize: '0.6875rem', color: '#64748b' }}>
                  {d === 0 ? 'Surface Skin' : d <= 100 ? 'Mixed Layer' : d <= 300 ? 'Thermocline' : 'Mesopelagic'}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
