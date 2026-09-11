import React from 'react';
import type { ArgoStatusResponse, PredictionResponse } from '../../types';
import { AlertCircleIcon, DatabaseIcon } from '../common/Icons';

interface ArgoValidationProps {
  argoStatus: ArgoStatusResponse | null;
  prediction: PredictionResponse;
}

export const ArgoValidation: React.FC<ArgoValidationProps> = ({ argoStatus }) => {
  const isAvailable = Boolean(argoStatus?.available && argoStatus.profiles_available > 0);

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">
            <DatabaseIcon size={16} style={{ color: '#0284c7' }} />
            <span>In-Situ Validation (ARGO Profiling Floats)</span>
          </div>
          <div className="card-subtitle">
            Independent ground-truth comparison against autonomous in-situ profiling floats
          </div>
        </div>
        <div>
          <span
            style={{
              fontSize: '0.6875rem',
              fontWeight: 600,
              padding: '3px 8px',
              borderRadius: '4px',
              backgroundColor: isAvailable ? '#dcfce7' : '#f1f5f9',
              color: isAvailable ? '#15803d' : '#64748b',
              textTransform: 'uppercase',
            }}
          >
            {isAvailable ? 'In-Situ Available' : 'Status: Not Available'}
          </span>
        </div>
      </div>

      <div className="card-body">
        {!isAvailable ? (
          <div
            style={{
              backgroundColor: '#f8fafc',
              border: '1px solid #e2e8f0',
              borderRadius: '6px',
              padding: '24px',
              textAlign: 'center',
            }}
          >
            <div style={{ display: 'inline-flex', padding: '10px', backgroundColor: '#e2e8f0', borderRadius: '50%', marginBottom: '12px' }}>
              <AlertCircleIcon size={24} style={{ color: '#64748b' }} />
            </div>
            <div style={{ fontSize: '0.9375rem', fontWeight: 600, color: '#0f172a', marginBottom: '6px' }}>
              In-Situ ARGO Observation Unconfigured
            </div>
            <p style={{ fontSize: '0.8125rem', color: '#64748b', maxWidth: '520px', margin: '0 auto 16px' }}>
              No real ARGO NetCDF observations are currently configured for independent in-situ validation. In order to preserve strict scientific integrity, synthetic profiles are never substituted as empirical observations.
            </p>
            <div style={{ fontSize: '0.75rem', color: '#94a3b8', fontFamily: 'var(--font-mono)' }}>
              Data directory: data/raw/argo/ • Profiles indexed: {argoStatus?.profiles_available || 0}
            </div>
          </div>
        ) : (
          <div>
            <div style={{ color: '#15803d', fontWeight: 600, marginBottom: '8px' }}>
              Co-located ARGO Profiles Detected: {argoStatus?.profiles_available}
            </div>
            <p style={{ fontSize: '0.8125rem', color: '#475569' }}>
              Independent in-situ verification compares predicted vertical temperatures against CTD sensors on autonomous profiling floats.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
