import React from 'react';
import { ActivityIcon, AlertCircleIcon, DatabaseIcon } from '../components/common/Icons';

export const ValidationPage: React.FC = () => {
  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
      {/* Header */}
      <div className="compact-hero" style={{ marginBottom: '24px' }}>
        <div className="hero-main">
          <h1>Model Evaluation & Architectural Benchmarks</h1>
          <p>
            Comparative empirical performance evaluation across Phase 2 baseline models and Phase 3 OceanEmbed V2 deep spatiotemporal architecture.
          </p>
        </div>
        <div className="hero-specs">
          <div className="spec-item">
            <span className="spec-label">Evaluation Domain</span>
            <span className="spec-val">Held-Out Spatial Split</span>
          </div>
          <div className="spec-item">
            <span className="spec-label">Target Metric</span>
            <span className="spec-val">Overall RMSE (°C)</span>
          </div>
        </div>
      </div>

      {/* Mandatory Disclaimer Banner */}
      <div
        style={{
          backgroundColor: '#fffbeb',
          border: '1px solid #fef3c7',
          borderLeft: '4px solid #f59e0b',
          borderRadius: '4px',
          padding: '16px',
          display: 'flex',
          gap: '12px',
          marginBottom: '24px',
        }}
      >
        <AlertCircleIcon size={20} style={{ color: '#d97706', flexShrink: 0, marginTop: '2px' }} />
        <div style={{ fontSize: '0.8125rem', color: '#92400e', lineHeight: 1.5 }}>
          <div style={{ fontWeight: 600, color: '#b45309', marginBottom: '2px' }}>
            SYNTHETIC DEVELOPMENT BENCHMARK NOTICE
          </div>
          The benchmarks reported below were evaluated strictly on held-out spatial test splits of deterministic synthetic simulation data. They validate neural representation capacity and spatial-temporal convergence under controlled physical forcing. These numbers must not be misrepresented as real-world in-situ ARGO operational metrics.
        </div>
      </div>

      {/* Comparative Benchmark Table */}
      <div className="card" style={{ marginBottom: '32px' }}>
        <div className="card-header">
          <div className="card-title">
            <ActivityIcon size={16} style={{ color: '#0284c7' }} />
            <span>Comprehensive Architectural Comparison</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
            Held-out spatial test split (Bay of Bengal)
          </div>
        </div>

        <div className="card-body" style={{ padding: 0 }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Model & Architecture</th>
                <th>Overall RMSE (°C)</th>
                <th>Overall MAE (°C)</th>
                <th>Overall Bias (°C)</th>
                <th>Validation Sample Units</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <div style={{ fontWeight: 600 }}>OceanBaseline V0</div>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Point MLP • Deterministic MSE</div>
                </td>
                <td style={{ color: '#0f172a', fontWeight: 600 }}>1.6756</td>
                <td>0.9557</td>
                <td style={{ color: '#b45309' }}>+0.6545</td>
                <td style={{ color: '#64748b' }}>504 (1D point profiles)</td>
              </tr>
              <tr>
                <td>
                  <div style={{ fontWeight: 600 }}>OceanBaseline V1</div>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Point MLP • Gaussian Uncertainty Head</div>
                </td>
                <td style={{ color: '#0f172a', fontWeight: 600 }}>3.9088</td>
                <td>2.4937</td>
                <td style={{ color: '#ef4444' }}>-0.8485</td>
                <td style={{ color: '#64748b' }}>504 (1D point profiles)</td>
              </tr>
              <tr style={{ backgroundColor: '#f0f9ff' }}>
                <td>
                  <div style={{ fontWeight: 600, color: '#0284c7' }}>OceanEmbed V2 (Proposed)</div>
                  <div style={{ fontSize: '0.75rem', color: '#0369a1' }}>
                    CNN + ConvLSTM + CBAM + 512-D Embedding + K=4 Regimes
                  </div>
                </td>
                <td style={{ color: '#0284c7', fontWeight: 700 }}>0.2366</td>
                <td style={{ color: '#0284c7', fontWeight: 600 }}>0.1872</td>
                <td style={{ color: '#15803d', fontWeight: 600 }}>+0.0649</td>
                <td style={{ color: '#0369a1' }}>207 (31x5x32x32 patches)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Uncertainty Calibration Section */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <DatabaseIcon size={16} style={{ color: '#0284c7' }} />
            <span>Uncertainty Quantification & Calibration</span>
          </div>
        </div>
        <div className="card-body" style={{ fontSize: '0.875rem', color: '#475569', lineHeight: 1.6 }}>
          <p style={{ marginBottom: '12px' }}>
            OceanEmbed V2 incorporates an explicit heteroscedastic uncertainty head parameterized as a Gaussian standard deviation σ at each vertical depth level. This captures both observation noise and depth-dependent baroclinic variability.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginTop: '16px' }}>
            <div style={{ backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '4px', padding: '16px' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#0f172a', textTransform: 'uppercase', marginBottom: '4px' }}>
                90% Coverage Target
              </div>
              <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#0284c7', fontFamily: 'var(--font-mono)' }}>
                ± 1.645 σ
              </div>
              <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '4px' }}>
                Gaussian predictive bounds: Lower = μ - 1.645σ, Upper = μ + 1.645σ
              </div>
            </div>
            <div style={{ backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '4px', padding: '16px' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#0f172a', textTransform: 'uppercase', marginBottom: '4px' }}>
                Operational Status
              </div>
              <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#10b981', fontFamily: 'var(--font-mono)' }}>
                Calibrated
              </div>
              <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '4px' }}>
                NLL loss optimization ensures conservative and strictly positive standard deviations
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
