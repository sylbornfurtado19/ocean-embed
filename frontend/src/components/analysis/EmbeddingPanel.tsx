import React, { useState } from 'react';
import type { PredictionResponse } from '../../types';
import { ChevronDownIcon, ChevronRightIcon, CpuIcon } from '../common/Icons';

interface EmbeddingPanelProps {
  prediction: PredictionResponse;
}

export const EmbeddingPanel: React.FC<EmbeddingPanelProps> = ({ prediction }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const { embedding } = prediction;

  if (!embedding || embedding.length === 0) {
    return (
      <div className="card">
        <div className="card-body" style={{ textAlign: 'center', color: '#64748b' }}>
          Embedding vector not available.
        </div>
      </div>
    );
  }

  // Calculate statistics
  const dim = embedding.length;
  const l2Norm = Math.sqrt(embedding.reduce((acc, v) => acc + v * v, 0)).toFixed(4);
  const mean = (embedding.reduce((acc, v) => acc + v, 0) / dim).toFixed(4);
  const variance = embedding.reduce((acc, v) => acc + Math.pow(v - parseFloat(mean), 2), 0) / dim;
  const std = Math.sqrt(variance).toFixed(4);

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">
            <CpuIcon size={16} style={{ color: '#0284c7' }} />
            <span>Ocean Embedding Representation</span>
          </div>
          <div className="card-subtitle">
            Continuous latent manifold extracted from CNN-ConvLSTM-CBAM spatial-temporal encoder
          </div>
        </div>
        <div style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: '#0284c7', fontWeight: 600 }}>
          {dim}-D Vector
        </div>
      </div>

      <div className="card-body">
        {/* Metric summary */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '20px' }}>
          <div className="metric-block" style={{ backgroundColor: '#f8fafc' }}>
            <div className="metric-label">Dimensions</div>
            <div className="metric-value" style={{ fontSize: '1.125rem' }}>{dim}</div>
            <div className="metric-sub">Latent manifold</div>
          </div>
          <div className="metric-block" style={{ backgroundColor: '#f8fafc' }}>
            <div className="metric-label">L2 Norm</div>
            <div className="metric-value" style={{ fontSize: '1.125rem' }}>{l2Norm}</div>
            <div className="metric-sub">Vector magnitude</div>
          </div>
          <div className="metric-block" style={{ backgroundColor: '#f8fafc' }}>
            <div className="metric-label">Mean Activation</div>
            <div className="metric-value" style={{ fontSize: '1.125rem' }}>{mean}</div>
            <div className="metric-sub">Centering check</div>
          </div>
          <div className="metric-block" style={{ backgroundColor: '#f8fafc' }}>
            <div className="metric-label">Standard Deviation</div>
            <div className="metric-value" style={{ fontSize: '1.125rem' }}>{std}</div>
            <div className="metric-sub">Embedding dispersion</div>
          </div>
        </div>

        {/* Expandable raw vector view */}
        <div style={{ border: '1px solid #e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            style={{
              width: '100%',
              padding: '10px 16px',
              backgroundColor: '#f1f5f9',
              border: 'none',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '0.8125rem',
              fontWeight: 600,
              color: '#334155',
            }}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              {isExpanded ? <ChevronDownIcon size={16} /> : <ChevronRightIcon size={16} />}
              <span>View full 512-D embedding vector values</span>
            </span>
            <span style={{ fontSize: '0.6875rem', color: '#64748b' }}>
              {isExpanded ? 'Click to collapse' : 'Click to inspect raw floats'}
            </span>
          </button>

          {isExpanded && (
            <div
              style={{
                padding: '16px',
                backgroundColor: '#0f172a',
                color: '#38bdf8',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.6875rem',
                maxHeight: '260px',
                overflowY: 'auto',
                lineHeight: 1.6,
                wordBreak: 'break-all',
              }}
            >
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                {JSON.stringify(embedding, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
