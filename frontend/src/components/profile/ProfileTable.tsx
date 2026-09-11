import React from 'react';
import type { PredictionResponse } from '../../types';
import { DownloadIcon } from '../common/Icons';

interface ProfileTableProps {
  prediction: PredictionResponse;
}

export const ProfileTable: React.FC<ProfileTableProps> = ({ prediction }) => {
  const { depths, temperatures, sigma, lower_90, upper_90, latitude, longitude, date } = prediction;

  const handleDownloadCsv = () => {
    const headers = ['Depth_m', 'Temperature_degC', 'Sigma_degC', 'Lower_90_degC', 'Upper_90_degC'];
    const rows = depths.map((d, i) => [
      d,
      temperatures[i].toFixed(4),
      sigma[i].toFixed(4),
      lower_90[i].toFixed(4),
      upper_90[i].toFixed(4),
    ]);

    const csvContent = [
      `# OceanEmbed V2 Profile Reconstruction`,
      `# Latitude: ${latitude.toFixed(4)} N, Longitude: ${longitude.toFixed(4)} E, Date: ${date}`,
      headers.join(','),
      ...rows.map((r) => r.join(',')),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `oceanembed_profile_${latitude.toFixed(2)}N_${longitude.toFixed(2)}E_${date}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">
            <span>Reconstructed Profile Numerical Values</span>
          </div>
          <div className="card-subtitle">
            15 standard vertical levels with predictive standard deviations and 90% intervals
          </div>
        </div>
        <button type="button" className="btn-secondary" onClick={handleDownloadCsv}>
          <DownloadIcon size={14} />
          <span>Download CSV</span>
        </button>
      </div>

      <div className="card-body" style={{ padding: 0 }}>
        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Depth (m)</th>
                <th>Temperature (°C)</th>
                <th>Uncertainty σ (°C)</th>
                <th>Lower 90% Bound</th>
                <th>Upper 90% Bound</th>
              </tr>
            </thead>
            <tbody>
              {depths.map((d, i) => (
                <tr key={d}>
                  <td style={{ fontWeight: 600 }}>{d}</td>
                  <td style={{ color: '#0284c7', fontWeight: 600 }}>{temperatures[i].toFixed(2)}</td>
                  <td style={{ color: '#64748b' }}>±{sigma[i].toFixed(2)}</td>
                  <td>{lower_90[i].toFixed(2)}</td>
                  <td>{upper_90[i].toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
