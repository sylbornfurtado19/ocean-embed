import React from 'react';
import { CalendarIcon, CompassIcon, RefreshCwIcon, ThermometerIcon } from '../common/Icons';

interface ConfigPanelProps {
  latitude: number;
  longitude: number;
  date: string;
  dataMode: 'synthetic' | 'real';
  loading: boolean;
  onLatitudeChange: (val: number) => void;
  onLongitudeChange: (val: number) => void;
  onDateChange: (val: string) => void;
  onDataModeChange: (val: 'synthetic' | 'real') => void;
  onReconstruct: () => void;
}

const PRESETS = [
  { name: 'Central Bay of Bengal', icon: '🌊', lat: 14.5, lon: 88.0, desc: 'Deep Ocean Basin' },
  { name: 'Cyclone Genesis Zone', icon: '🌀', lat: 11.0, lon: 89.5, desc: 'High TCHP Heat Potential' },
  { name: 'Northern Coastal Shelf', icon: '⚓', lat: 19.5, lon: 89.0, desc: 'Estuarine Freshwater Outflow' },
  { name: 'Andaman Sea Basin', icon: '🏝️', lat: 12.0, lon: 93.5, desc: 'Internal Waves & Stratification' },
  { name: 'Southern Equatorial', icon: '🧭', lat: 6.5, lon: 84.5, desc: 'Equatorial Jet Watermass' },
  { name: 'Sri Lanka Basin', icon: '🛥️', lat: 8.0, lon: 82.5, desc: 'Coastal Summer Upwelling' },
];

export const ConfigPanel: React.FC<ConfigPanelProps> = ({
  latitude,
  longitude,
  date,
  dataMode,
  loading,
  onLatitudeChange,
  onLongitudeChange,
  onDateChange,
  onDataModeChange,
  onReconstruct,
}) => {
  const isLatValid = latitude >= 5.0 && latitude <= 23.0;
  const isLonValid = longitude >= 80.0 && longitude <= 100.0;
  const isValid = isLatValid && isLonValid && Boolean(date);

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">
            <CompassIcon size={18} style={{ color: '#0ea5e9' }} />
            <span>Prediction Configuration</span>
          </div>
          <div className="card-subtitle">Set coordinates, temporal anchor, and data stream</div>
        </div>
      </div>

      <div className="card-body">
        {/* Preset Locations Visual Quick-Pickers */}
        <div className="form-group">
          <label className="form-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Visual Region Quick-Pickers</span>
            <span style={{ fontSize: '0.6875rem', color: '#64748b' }}>Select region preset</span>
          </label>
          <div className="preset-pills">
            {PRESETS.map((preset) => {
              const isActive = Math.abs(latitude - preset.lat) < 0.2 && Math.abs(longitude - preset.lon) < 0.2;
              return (
                <button
                  key={preset.name}
                  type="button"
                  className={`preset-pill ${isActive ? 'active' : ''}`}
                  onClick={() => {
                    onLatitudeChange(preset.lat);
                    onLongitudeChange(preset.lon);
                  }}
                >
                  <span className="preset-icon">{preset.icon}</span>
                  <div>
                    <span className="preset-name">{preset.name}</span>
                    <span className="preset-coords">{preset.lat.toFixed(1)}°N, {preset.lon.toFixed(1)}°E</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Coordinates Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
          <div className="form-group">
            <label className="form-label">
              Latitude (°N)
              {!isLatValid && <span style={{ color: '#ef4444', marginLeft: '6px' }}>[5°–23°N]</span>}
            </label>
            <input
              type="number"
              step="0.01"
              min="5.0"
              max="23.0"
              className="form-input"
              value={latitude}
              onChange={(e) => onLatitudeChange(parseFloat(e.target.value) || 0)}
            />
            <div className="form-help">Supported: 5.00°N to 23.00°N</div>
          </div>

          <div className="form-group">
            <label className="form-label">
              Longitude (°E)
              {!isLonValid && <span style={{ color: '#ef4444', marginLeft: '6px' }}>[80°–100°E]</span>}
            </label>
            <input
              type="number"
              step="0.01"
              min="80.0"
              max="100.0"
              className="form-input"
              value={longitude}
              onChange={(e) => onLongitudeChange(parseFloat(e.target.value) || 0)}
            />
            <div className="form-help">Supported: 80.00°E to 100.00°E</div>
          </div>
        </div>

        {/* Date and Data Source */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
          <div className="form-group">
            <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <CalendarIcon size={14} style={{ color: '#0ea5e9' }} />
              <span>Target Date</span>
            </label>
            <input
              type="date"
              className="form-input"
              value={date}
              min="2020-01-01"
              max="2026-12-31"
              onChange={(e) => onDateChange(e.target.value)}
            />
            <div className="form-help">Sequence central date</div>
          </div>

          <div className="form-group">
            <label className="form-label">Observation Stream</label>
            <select
              className="form-input"
              value={dataMode}
              onChange={(e) => onDataModeChange(e.target.value as 'synthetic' | 'real')}
            >
              <option value="synthetic">Synthetic Development Stream</option>
              <option value="real">Real Satellite (CMEMS)</option>
            </select>
            <div className="form-help">Input feature stream</div>
          </div>
        </div>

        {/* Target summary box */}
        <div
          style={{
            backgroundColor: 'rgba(7, 18, 36, 0.8)',
            border: '1px solid rgba(14, 165, 233, 0.25)',
            borderRadius: '8px',
            padding: '12px 16px',
            marginBottom: '20px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <div>
            <div style={{ fontSize: '0.6875rem', color: '#64748b', textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.04em' }}>
              Target Coordinates & Anchor Date
            </div>
            <div style={{ fontSize: '0.875rem', fontWeight: 700, color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>
              {latitude.toFixed(2)}°N, {longitude.toFixed(2)}°E • {date}
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <span
              style={{
                fontSize: '0.75rem',
                padding: '3px 10px',
                borderRadius: '4px',
                backgroundColor: dataMode === 'real' ? 'rgba(14, 165, 233, 0.2)' : 'rgba(245, 158, 11, 0.15)',
                border: dataMode === 'real' ? '1px solid rgba(14, 165, 233, 0.4)' : '1px solid rgba(245, 158, 11, 0.3)',
                color: dataMode === 'real' ? '#38bdf8' : '#fbbf24',
                fontWeight: 600,
                fontFamily: 'var(--font-mono)',
              }}
            >
              {dataMode === 'real' ? 'Real Stream' : 'Synthetic Data'}
            </span>
          </div>
        </div>

        {/* Action Button */}
        <button
          type="button"
          className="btn-primary"
          disabled={!isValid || loading}
          onClick={onReconstruct}
        >
          {loading ? (
            <>
              <RefreshCwIcon size={16} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
              <span>Reconstructing Profile...</span>
            </>
          ) : (
            <>
              <ThermometerIcon size={16} />
              <span>Reconstruct 3D Profile</span>
            </>
          )}
        </button>

        {loading && (
          <div
            style={{
              marginTop: '12px',
              fontSize: '0.75rem',
              color: '#94a3b8',
              textAlign: 'center',
              lineHeight: 1.4,
              fontFamily: 'var(--font-mono)',
            }}
          >
            Processing 31-day spatiotemporal tensor • Executing OceanEmbed V2 • Evaluating 90% Gaussian interval
          </div>
        )}
      </div>
    </div>
  );
};
