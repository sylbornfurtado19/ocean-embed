import React from 'react';
import type { PredictionResponse } from '../../types';

interface ClimatologyDecompositionProps {
  prediction: PredictionResponse;
}

export const ClimatologyDecomposition: React.FC<ClimatologyDecompositionProps> = ({ prediction }) => {
  const { depths, temperatures, climatology_prior, anomaly } = prediction;

  if (!climatology_prior || !anomaly) {
    return (
      <div className="card">
        <div className="card-body" style={{ textAlign: 'center', color: '#64748b' }}>
          Climatology decomposition data is not available for this prediction.
        </div>
      </div>
    );
  }

  // Chart setup
  const width = 600;
  const height = 400;
  const padding = { top: 30, right: 30, bottom: 50, left: 60 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const allVals = [...temperatures, ...climatology_prior];
  const minVal = Math.floor(Math.min(...allVals) - 1);
  const maxVal = Math.ceil(Math.max(...allVals) + 1);

  const scaleX = (val: number) => padding.left + ((val - minVal) / (maxVal - minVal)) * plotWidth;
  const scaleY = (depth: number) => padding.top + (depth / 1000) * plotHeight;

  const predPath = depths.map((d, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(temperatures[i])} ${scaleY(d)}`).join(' ');
  const climPath = depths.map((d, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(climatology_prior[i])} ${scaleY(d)}`).join(' ');

  const xTicks = [];
  for (let t = Math.ceil(minVal / 4) * 4; t <= maxVal; t += 4) {
    xTicks.push(t);
  }
  const yTicks = [0, 100, 200, 300, 500, 700, 1000];

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">
            <span>Climatology Residual Decomposition</span>
          </div>
          <div className="card-subtitle">
            Physics-informed prior baseline combined with deep residual anomaly prediction
          </div>
        </div>
        <div style={{ display: 'flex', gap: '16px', fontSize: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '12px', height: '2px', backgroundColor: '#64748b', display: 'inline-block', borderTop: '1px dashed #64748b' }} />
            <span style={{ color: '#64748b' }}>Climatology Prior T_clim(z)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '12px', height: '2.5px', backgroundColor: '#0284c7', display: 'inline-block' }} />
            <span style={{ color: '#0f172a', fontWeight: 600 }}>Final Profile T(z)</span>
          </div>
        </div>
      </div>

      <div className="card-body">
        {/* Formula Banner */}
        <div className="formula-banner">
          Final Temperature Profile T(z) = Climatology Prior T_clim(z) + Predicted Anomaly ΔT(z)
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '24px', alignItems: 'center' }}>
          {/* Visual SVG chart */}
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', maxWidth: '480px' }}>
              <g className="grid">
                {xTicks.map((t) => (
                  <line
                    key={`cx-${t}`}
                    x1={scaleX(t)}
                    y1={padding.top}
                    x2={scaleX(t)}
                    y2={padding.top + plotHeight}
                    stroke="#f1f5f9"
                    strokeWidth="1"
                  />
                ))}
                {yTicks.map((d) => (
                  <line
                    key={`cy-${d}`}
                    x1={padding.left}
                    y1={scaleY(d)}
                    x2={padding.left + plotWidth}
                    y2={scaleY(d)}
                    stroke="#f1f5f9"
                    strokeWidth="1"
                  />
                ))}
              </g>

              {/* Axes */}
              <line x1={padding.left} y1={padding.top + plotHeight} x2={padding.left + plotWidth} y2={padding.top + plotHeight} stroke="#94a3b8" />
              <line x1={padding.left} y1={padding.top} x2={padding.left} y2={padding.top + plotHeight} stroke="#94a3b8" />

              {xTicks.map((t) => (
                <text key={`cxt-${t}`} x={scaleX(t)} y={padding.top + plotHeight + 16} textAnchor="middle" fontSize="10" fill="#64748b" fontFamily="var(--font-mono)">
                  {t}
                </text>
              ))}
              {yTicks.map((d) => (
                <text key={`cyt-${d}`} x={padding.left - 8} y={scaleY(d) + 3} textAnchor="end" fontSize="10" fill="#64748b" fontFamily="var(--font-mono)">
                  {d}
                </text>
              ))}

              {/* Climatology line (dashed) */}
              <path d={climPath} fill="none" stroke="#64748b" strokeWidth="2" strokeDasharray="4,4" />

              {/* Final profile line (solid) */}
              <path d={predPath} fill="none" stroke="#0284c7" strokeWidth="2.5" />
            </svg>
          </div>

          {/* Table of Climatology & Anomalies at key depths */}
          <div style={{ maxHeight: '340px', overflowY: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Depth</th>
                  <th>Prior</th>
                  <th>Anomaly</th>
                  <th>Final</th>
                </tr>
              </thead>
              <tbody>
                {depths.map((d, i) => (
                  <tr key={`decomp-${d}`}>
                    <td style={{ fontWeight: 600 }}>{d} m</td>
                    <td style={{ color: '#64748b' }}>{climatology_prior[i].toFixed(2)}</td>
                    <td style={{ color: anomaly[i] >= 0 ? '#0284c7' : '#ef4444' }}>
                      {anomaly[i] >= 0 ? `+${anomaly[i].toFixed(2)}` : anomaly[i].toFixed(2)}
                    </td>
                    <td style={{ fontWeight: 600, color: '#0f172a' }}>{temperatures[i].toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
