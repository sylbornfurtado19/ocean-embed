import React from 'react';
import { StatusBadge } from '../common/StatusBadge';

interface StatusStripProps {
  apiStatus: 'ok' | 'degraded' | 'error';
  modelAvailable: boolean;
  dataMode: string;
  argoAvailable: boolean;
  satelliteAvailable: boolean;
}

export const StatusStrip: React.FC<StatusStripProps> = ({
  apiStatus,
  modelAvailable,
  dataMode,
  argoAvailable,
  satelliteAvailable,
}) => {
  return (
    <div className="status-strip">
      <div className="status-strip-group">
        <StatusBadge
          label="DATA SOURCE"
          variant={dataMode === 'real' && satelliteAvailable ? 'available' : 'warning'}
          text={dataMode === 'real' && satelliteAvailable ? 'Real Satellite (CMEMS)' : 'Synthetic Development Data'}
        />
        <StatusBadge
          label="MODEL"
          variant={modelAvailable ? 'operational' : 'unavailable'}
          text={modelAvailable ? 'OceanEmbed V2' : 'Unavailable'}
        />
      </div>

      <div className="status-strip-group">
        <StatusBadge
          label="API"
          variant={apiStatus === 'ok' ? 'operational' : 'unavailable'}
          text={apiStatus === 'ok' ? 'Operational' : apiStatus === 'degraded' ? 'Degraded' : 'Offline'}
        />
        <StatusBadge
          label="ARGO"
          variant={argoAvailable ? 'available' : 'unavailable'}
          text={argoAvailable ? 'Available' : 'Unavailable'}
        />
      </div>
    </div>
  );
};
