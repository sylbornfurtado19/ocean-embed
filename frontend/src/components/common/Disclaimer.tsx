import React from 'react';
import { InfoIcon } from './Icons';

interface DisclaimerProps {
  title?: string;
  message?: string;
}

export const Disclaimer: React.FC<DisclaimerProps> = ({
  title = 'DEVELOPMENT DATA',
  message = 'This operational demonstration uses deterministic synthetic development data for model verification across the Bay of Bengal domain. Real satellite and in-situ ARGO observations require separate live telemetry configurations.',
}) => {
  return (
    <div className="disclaimer-card">
      <InfoIcon size={18} style={{ color: '#0284c7', flexShrink: 0, marginTop: '2px' }} />
      <div>
        <div className="disclaimer-title">{title}</div>
        <div>{message}</div>
      </div>
    </div>
  );
};
