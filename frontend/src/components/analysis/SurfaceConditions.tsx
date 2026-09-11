import React from 'react';
import type { PredictionResponse } from '../../types';

interface SurfaceConditionsProps {
  prediction: PredictionResponse;
}

const VARIABLE_METADATA: Record<string, { label: string; unit: string; desc: string }> = {
  sst: { label: 'Sea Surface Temperature (SST)', unit: '°C', desc: 'Thermal boundary forcing from thermal infrared radiometry' },
  sss: { label: 'Sea Surface Salinity (SSS)', unit: 'PSU', desc: 'Halocline forcing from microwave radiometry' },
  sla: { label: 'Sea Level Anomaly (SLA / SSHA)', unit: 'm', desc: 'Dynamic ocean topography indicating baroclinic pressure' },
  wind_u: { label: 'Zonal Surface Wind (Wind-U)', unit: 'm/s', desc: 'East-west atmospheric momentum flux component' },
  wind_v: { label: 'Meridional Surface Wind (Wind-V)', unit: 'm/s', desc: 'North-south atmospheric momentum flux component' },
  heat_flux: { label: 'Net Surface Heat Flux', unit: 'W/m²', desc: 'Combined latent, sensible, and radiative energy budget' },
};

export const SurfaceConditions: React.FC<SurfaceConditionsProps> = ({ prediction }) => {
  const { surface_inputs, data_mode } = prediction;

  if (!surface_inputs) {
    return (
      <div className="card">
        <div className="card-body" style={{ textAlign: 'center', color: '#64748b' }}>
          Surface condition telemetry is unavailable for this coordinate.
        </div>
      </div>
    );
  }

  const isSynthetic = data_mode.includes('synthetic');

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">
            <span>Surface Input Conditions</span>
          </div>
          <div className="card-subtitle">
            Spatiotemporal surface boundary variables ingested by OceanEmbed V2
          </div>
        </div>
        <div>
          <span
            style={{
              fontSize: '0.6875rem',
              fontWeight: 600,
              padding: '3px 8px',
              borderRadius: '4px',
              backgroundColor: isSynthetic ? '#fef3c7' : '#e0f2fe',
              color: isSynthetic ? '#b45309' : '#0369a1',
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
            }}
          >
            {isSynthetic ? 'Synthetic Demo Surface Conditions' : 'Real Satellite Stream'}
          </span>
        </div>
      </div>

      <div className="card-body" style={{ padding: 0 }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Variable</th>
              <th>Observed Value</th>
              <th>Physical Description</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(surface_inputs).map(([key, val]) => {
              const meta = VARIABLE_METADATA[key.toLowerCase()] || {
                label: key.toUpperCase(),
                unit: '',
                desc: 'Atmospheric/oceanic boundary variable',
              };
              return (
                <tr key={key}>
                  <td style={{ fontWeight: 600 }}>{meta.label}</td>
                  <td style={{ color: '#0284c7', fontWeight: 600 }}>
                    {typeof val === 'number' ? val.toFixed(2) : String(val)} {meta.unit}
                  </td>
                  <td style={{ color: '#64748b', fontSize: '0.75rem', fontFamily: 'var(--font-sans)' }}>
                    {meta.desc}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
