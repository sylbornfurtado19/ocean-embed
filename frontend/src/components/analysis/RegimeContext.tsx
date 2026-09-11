import React from 'react';
import type { PredictionResponse } from '../../types';

interface RegimeContextProps {
  prediction: PredictionResponse;
}

export const RegimeContext: React.FC<RegimeContextProps> = ({ prediction }) => {
  const { regime_probs } = prediction;

  const regimeLabels = [
    'Latent Regime 1',
    'Latent Regime 2',
    'Latent Regime 3',
    'Latent Regime 4',
  ];

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">
            <span>Latent Regime Context</span>
          </div>
          <div className="card-subtitle">
            K = 4 soft latent mixture distribution learned by OceanEmbed V2
          </div>
        </div>
        <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
          ∑ P_k ≈ 1.000
        </div>
      </div>

      <div className="card-body">
        <p style={{ fontSize: '0.8125rem', color: '#475569', marginBottom: '20px' }}>
          Soft latent representation learned by OceanEmbed V2. The model partitions the spatiotemporal embedding space into K=4 continuous thermodynamic regime clusters to condition vertical profile decoding.
        </p>

        <div className="regime-list">
          {regime_probs.map((prob, idx) => {
            const percentage = (prob * 100).toFixed(1);
            return (
              <div key={`regime-${idx}`} className="regime-item">
                <div className="regime-meta">
                  <span className="regime-name">{regimeLabels[idx] || `Latent Regime ${idx + 1}`}</span>
                  <span className="regime-pct">{percentage}% ({prob.toFixed(4)})</span>
                </div>
                <div className="regime-track">
                  <div
                    className="regime-fill"
                    style={{ width: `${Math.max(1, Math.min(100, prob * 100))}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
