import React, { useEffect, useState } from 'react';
import type { ArgoStatusResponse, HealthResponse, ModelStatusResponse, PredictionResponse, SatelliteStatusResponse } from './types';
import { apiClient } from './api/client';
import { TopNav, type NavPage } from './components/layout/TopNav';
import { StatusStrip } from './components/layout/StatusStrip';
import { Footer } from './components/layout/Footer';
import { ReconstructionPage } from './pages/ReconstructionPage';
import { OverviewPage } from './pages/OverviewPage';
import { ValidationPage } from './pages/ValidationPage';
import { SystemPage } from './pages/SystemPage';
import './styles/index.css';
import './styles/components.css';

export const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<NavPage>('reconstruction');

  // Subsystem states
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatusResponse | null>(null);
  const [argoStatus, setArgoStatus] = useState<ArgoStatusResponse | null>(null);
  const [satelliteStatus, setSatelliteStatus] = useState<SatelliteStatusResponse | null>(null);
  const [apiConnected, setApiConnected] = useState(false);

  // Prediction state
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initial fetch of system statuses and initial reconstruction
  useEffect(() => {
    let isMounted = true;

    const fetchInitialData = async () => {
      try {
        const [h, m, a, s] = await Promise.allSettled([
          apiClient.getHealth(),
          apiClient.getModelStatus(),
          apiClient.getArgoStatus(),
          apiClient.getSatelliteStatus(),
        ]);

        if (!isMounted) return;

        if (h.status === 'fulfilled') {
          setHealth(h.value);
          setApiConnected(true);
        } else {
          setApiConnected(false);
        }

        if (m.status === 'fulfilled') setModelStatus(m.value);
        if (a.status === 'fulfilled') setArgoStatus(a.value);
        if (s.status === 'fulfilled') setSatelliteStatus(s.value);

        // Pre-fetch default reconstruction for initial display
        try {
          setLoading(true);
          const defaultPrediction = await apiClient.predictProfile({
            latitude: 14.5,
            longitude: 88.0,
            date: '2023-06-15',
            data_mode: 'synthetic',
          });
          if (isMounted) {
            setPrediction(defaultPrediction);
            setError(null);
          }
        } catch (predErr: unknown) {
          if (isMounted) {
            setError(predErr instanceof Error ? predErr.message : 'Initial prediction failed.');
          }
        } finally {
          if (isMounted) setLoading(false);
        }
      } catch (err: unknown) {
        if (isMounted) {
          setApiConnected(false);
          setError('Could not connect to FastAPI backend at http://localhost:8000.');
        }
      }
    };

    fetchInitialData();

    return () => {
      isMounted = false;
    };
  }, []);

  const handleReconstruct = async (
    lat: number,
    lon: number,
    date: string,
    mode: 'synthetic' | 'real'
  ) => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.predictProfile({
        latitude: lat,
        longitude: lon,
        date,
        data_mode: mode,
      });
      setPrediction(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Subsurface reconstruction failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      {/* Top Navigation */}
      <TopNav
        currentPage={currentPage}
        onPageChange={setCurrentPage}
        apiConnected={apiConnected}
      />

      {/* System Status Strip */}
      <StatusStrip
        apiStatus={health ? health.status : apiConnected ? 'ok' : 'error'}
        modelAvailable={Boolean(modelStatus?.is_loaded ?? health?.model.available)}
        dataMode={health?.mode || 'synthetic_demo'}
        argoAvailable={Boolean(argoStatus?.available && argoStatus.profiles_available > 0)}
        satelliteAvailable={Boolean(satelliteStatus?.available)}
      />

      {/* Main Content Area */}
      <main className="main-content">
        {currentPage === 'reconstruction' && (
          <ReconstructionPage
            prediction={prediction}
            argoStatus={argoStatus}
            loading={loading}
            error={error}
            onReconstruct={handleReconstruct}
          />
        )}

        {currentPage === 'overview' && <OverviewPage />}

        {currentPage === 'validation' && <ValidationPage />}

        {currentPage === 'system' && (
          <SystemPage
            health={health}
            modelStatus={modelStatus}
            argoStatus={argoStatus}
            satelliteStatus={satelliteStatus}
          />
        )}
      </main>

      {/* Footer */}
      <Footer />
    </div>
  );
};

export default App;
