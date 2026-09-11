import React, { useState } from 'react';
import type { PredictionResponse } from '../../types';

interface ProfileChartProps {
  prediction: PredictionResponse;
}

export const ProfileChart: React.FC<ProfileChartProps> = ({ prediction }) => {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const { depths, temperatures, sigma, lower_90, upper_90 } = prediction;

  // Chart dimensions
  const width = 600;
  const height = 480;
  const padding = { top: 40, right: 30, bottom: 50, left: 60 };

  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  // Domain scaling
  // X: Temperature (e.g. 0 to 35 °C)
  const allTemps = [...temperatures, ...lower_90, ...upper_90];
  const minTemp = Math.floor(Math.min(...allTemps) - 1);
  const maxTemp = Math.ceil(Math.max(...allTemps) + 1);

  // Y: Depth (0 to 1000 m, inverted)
  const maxDepth = 1000;

  const scaleX = (temp: number) => padding.left + ((temp - minTemp) / (maxTemp - minTemp)) * plotWidth;
  // Inverted depth: 0 at top (padding.top), 1000 at bottom (padding.top + plotHeight)
  const scaleY = (depth: number) => padding.top + (depth / maxDepth) * plotHeight;

  // Generate path for the 90% uncertainty polygon: lower bound down, upper bound back up
  const lowerPoints = depths.map((d, i) => `${scaleX(lower_90[i])},${scaleY(d)}`);
  const upperPointsReversed = depths
    .slice()
    .reverse()
    .map((d, i) => {
      const originalIdx = depths.length - 1 - i;
      return `${scaleX(upper_90[originalIdx])},${scaleY(d)}`;
    });
  const polygonPoints = [...lowerPoints, ...upperPointsReversed].join(' ');

  // Generate path for mean temperature reconstruction line
  const meanPath = depths.map((d, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(temperatures[i])} ${scaleY(d)}`).join(' ');

  // Grid tick values
  const xTicks = [];
  const tempStep = (maxTemp - minTemp) > 15 ? 5 : 2;
  for (let t = Math.ceil(minTemp / tempStep) * tempStep; t <= maxTemp; t += tempStep) {
    xTicks.push(t);
  }

  const yTicks = [0, 100, 200, 300, 500, 700, 1000];

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div className="card-header">
        <div>
          <div className="card-title">
            <span>Subsurface Temperature Reconstruction</span>
          </div>
          <div className="card-subtitle">
            Predicted continuous 15-depth vertical temperature profile (0–1000 m)
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '12px', height: '3px', backgroundColor: '#0284c7', display: 'inline-block' }} />
            <span style={{ color: '#0f172a', fontWeight: 600 }}>OceanEmbed V2</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '12px', height: '10px', backgroundColor: 'rgba(2, 132, 199, 0.2)', display: 'inline-block', border: '1px solid rgba(2, 132, 199, 0.4)' }} />
            <span style={{ color: '#64748b' }}>90% Predictive Interval (±1.645σ)</span>
          </div>
        </div>
      </div>

      <div className="card-body" style={{ position: 'relative', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          style={{ width: '100%', height: 'auto', maxWidth: '640px', overflow: 'visible' }}
        >
          {/* Background grid */}
          <g className="grid">
            {xTicks.map((t) => (
              <line
                key={`xtick-${t}`}
                x1={scaleX(t)}
                y1={padding.top}
                x2={scaleX(t)}
                y2={padding.top + plotHeight}
                stroke="#e2e8f0"
                strokeWidth="1"
                strokeDasharray="2,2"
              />
            ))}
            {yTicks.map((d) => (
              <line
                key={`ytick-${d}`}
                x1={padding.left}
                y1={scaleY(d)}
                x2={padding.left + plotWidth}
                y2={scaleY(d)}
                stroke="#e2e8f0"
                strokeWidth="1"
                strokeDasharray="2,2"
              />
            ))}
          </g>

          {/* Axes */}
          <line
            x1={padding.left}
            y1={padding.top + plotHeight}
            x2={padding.left + plotWidth}
            y2={padding.top + plotHeight}
            stroke="#94a3b8"
            strokeWidth="1.5"
          />
          <line
            x1={padding.left}
            y1={padding.top}
            x2={padding.left}
            y2={padding.top + plotHeight}
            stroke="#94a3b8"
            strokeWidth="1.5"
          />

          {/* X Tick labels */}
          {xTicks.map((t) => (
            <text
              key={`xlabel-${t}`}
              x={scaleX(t)}
              y={padding.top + plotHeight + 20}
              textAnchor="middle"
              fontSize="11"
              fill="#64748b"
              fontFamily="var(--font-mono)"
            >
              {t}
            </text>
          ))}
          <text
            x={padding.left + plotWidth / 2}
            y={height - 10}
            textAnchor="middle"
            fontSize="12"
            fontWeight="600"
            fill="#334155"
          >
            Temperature (°C)
          </text>

          {/* Y Tick labels */}
          {yTicks.map((d) => (
            <text
              key={`ylabel-${d}`}
              x={padding.left - 10}
              y={scaleY(d) + 4}
              textAnchor="end"
              fontSize="11"
              fill="#64748b"
              fontFamily="var(--font-mono)"
            >
              {d}
            </text>
          ))}
          <text
            x={-height / 2}
            y={18}
            transform="rotate(-90)"
            textAnchor="middle"
            fontSize="12"
            fontWeight="600"
            fill="#334155"
          >
            Depth (m) — Inverted
          </text>

          {/* 90% Gaussian Uncertainty Shaded Envelope */}
          <polygon
            points={polygonPoints}
            fill="rgba(2, 132, 199, 0.15)"
            stroke="rgba(2, 132, 199, 0.35)"
            strokeWidth="1"
          />

          {/* Mean Temperature Reconstruction Line */}
          <path
            d={meanPath}
            fill="none"
            stroke="#0284c7"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Depth points */}
          {depths.map((d, i) => {
            const isHovered = hoverIndex === i;
            return (
              <g
                key={`point-${d}`}
                onMouseEnter={() => setHoverIndex(i)}
                onMouseLeave={() => setHoverIndex(null)}
                style={{ cursor: 'pointer' }}
              >
                <circle
                  cx={scaleX(temperatures[i])}
                  cy={scaleY(d)}
                  r={isHovered ? 6 : 4}
                  fill={isHovered ? '#0369a1' : '#0284c7'}
                  stroke="#ffffff"
                  strokeWidth={isHovered ? 2 : 1.5}
                />
              </g>
            );
          })}
        </svg>

        {/* Hover Tooltip Card */}
        {hoverIndex !== null && (
          <div
            style={{
              position: 'absolute',
              top: '16px',
              right: '24px',
              backgroundColor: 'rgba(15, 23, 42, 0.95)',
              color: '#ffffff',
              padding: '8px 12px',
              borderRadius: '4px',
              fontSize: '0.75rem',
              boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
              pointerEvents: 'none',
              fontFamily: 'var(--font-mono)',
              lineHeight: 1.5,
              zIndex: 10,
            }}
          >
            <div style={{ fontWeight: 600, color: '#38bdf8' }}>Depth: {depths[hoverIndex]} m</div>
            <div>Predicted: {temperatures[hoverIndex].toFixed(2)} °C</div>
            <div style={{ color: '#94a3b8' }}>Uncertainty (σ): ±{sigma[hoverIndex].toFixed(2)} °C</div>
            <div style={{ color: '#cbd5e1', fontSize: '0.6875rem' }}>
              90% Interval: [{lower_90[hoverIndex].toFixed(2)}, {upper_90[hoverIndex].toFixed(2)}] °C
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
