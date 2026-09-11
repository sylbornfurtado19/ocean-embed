import React, { useState } from 'react';
import type { ArgoStatusResponse, PredictionResponse } from '../types';
import { BayOfBengalMap } from '../components/map/BayOfBengalMap';
import { ConfigPanel } from '../components/prediction/ConfigPanel';
import { ProfileSummary } from '../components/profile/ProfileSummary';
import { ProfileChart } from '../components/profile/ProfileChart';
import { ProfileTable } from '../components/profile/ProfileTable';
import { ClimatologyDecomposition } from '../components/analysis/ClimatologyDecomposition';
import { RegimeContext } from '../components/analysis/RegimeContext';
import { SurfaceConditions } from '../components/analysis/SurfaceConditions';
import { EmbeddingPanel } from '../components/analysis/EmbeddingPanel';
import { ArgoValidation } from '../components/analysis/ArgoValidation';

interface ReconstructionPageProps {
  prediction: PredictionResponse | null;
  argoStatus: ArgoStatusResponse | null;
  loading: boolean;
  error: string | null;
  onReconstruct: (lat: number, lon: number, date: string, mode: 'synthetic' | 'real') => void;
}

type TabKey = 'profile' | 'climatology' | 'regime' | 'surface' | 'embedding' | 'argo';

export const ReconstructionPage: React.FC<ReconstructionPageProps> = ({
  prediction,
  argoStatus,
  loading,
  error,
  onReconstruct,
}) => {
  const [latitude, setLatitude] = useState(14.5);
  const [longitude, setLongitude] = useState(88.0);
  const [date, setDate] = useState('2023-06-15');
  const [dataMode, setDataMode] = useState<'synthetic' | 'real'>('synthetic');
  const [activeTab, setActiveTab] = useState<TabKey>('profile');

  const handleReconstruct = () => {
    onReconstruct(latitude, longitude, date, dataMode);
  };

  return (
    <div>


      {/* Error Message */}
      {error && (
        <div
          style={{
            backgroundColor: '#fef2f2',
            border: '1px solid #fecaca',
            borderLeft: '4px solid #ef4444',
            borderRadius: '4px',
            padding: '12px 16px',
            color: '#991b1b',
            fontSize: '0.8125rem',
            marginBottom: '24px',
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: '2px' }}>Operational Notice / Error</div>
          <div>{error}</div>
        </div>
      )}

      {/* Main Two-Column Workspace */}
      <div className="workspace-grid">
        {/* Left Column: Leaflet Map */}
        <div>
          <BayOfBengalMap
            latitude={latitude}
            longitude={longitude}
            onSelectCoordinates={(lat: number, lon: number) => {
              setLatitude(lat);
              setLongitude(lon);
            }}
          />
        </div>

        {/* Right Column: Prediction Configuration */}
        <div>
          <ConfigPanel
            latitude={latitude}
            longitude={longitude}
            date={date}
            dataMode={dataMode}
            loading={loading}
            onLatitudeChange={setLatitude}
            onLongitudeChange={setLongitude}
            onDateChange={setDate}
            onDataModeChange={setDataMode}
            onReconstruct={handleReconstruct}
          />
        </div>
      </div>

      {/* Reconstructed Profile Results & Scientific Analysis */}
      {prediction && (
        <div>
          {/* Key Metrics Row */}
          <ProfileSummary prediction={prediction} />

          {/* Centerpiece Profile Chart */}
          <div style={{ marginBottom: '24px' }}>
            <ProfileChart prediction={prediction} />
          </div>

          {/* Segmented Scientific Analysis Section */}
          <div className="card">
            <div className="tabs-header">
              <button
                type="button"
                className={`tab-btn ${activeTab === 'profile' ? 'active' : ''}`}
                onClick={() => setActiveTab('profile')}
              >
                Profile Table
              </button>
              <button
                type="button"
                className={`tab-btn ${activeTab === 'climatology' ? 'active' : ''}`}
                onClick={() => setActiveTab('climatology')}
              >
                Climatology Decomposition
              </button>
              <button
                type="button"
                className={`tab-btn ${activeTab === 'regime' ? 'active' : ''}`}
                onClick={() => setActiveTab('regime')}
              >
                Latent Regimes
              </button>
              <button
                type="button"
                className={`tab-btn ${activeTab === 'surface' ? 'active' : ''}`}
                onClick={() => setActiveTab('surface')}
              >
                Surface Conditions
              </button>
              <button
                type="button"
                className={`tab-btn ${activeTab === 'embedding' ? 'active' : ''}`}
                onClick={() => setActiveTab('embedding')}
              >
                512-D Embedding
              </button>
              <button
                type="button"
                className={`tab-btn ${activeTab === 'argo' ? 'active' : ''}`}
                onClick={() => setActiveTab('argo')}
              >
                In-Situ Validation
              </button>
            </div>

            <div style={{ padding: '24px' }}>
              {activeTab === 'profile' && <ProfileTable prediction={prediction} />}
              {activeTab === 'climatology' && <ClimatologyDecomposition prediction={prediction} />}
              {activeTab === 'regime' && <RegimeContext prediction={prediction} />}
              {activeTab === 'surface' && <SurfaceConditions prediction={prediction} />}
              {activeTab === 'embedding' && <EmbeddingPanel prediction={prediction} />}
              {activeTab === 'argo' && <ArgoValidation argoStatus={argoStatus} prediction={prediction} />}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
