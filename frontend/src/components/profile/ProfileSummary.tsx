import React from 'react';
import type { PredictionResponse } from '../../types';
import { MetricBlock } from '../common/MetricBlock';

interface ProfileSummaryProps {
  prediction: PredictionResponse;
}

export const ProfileSummary: React.FC<ProfileSummaryProps> = ({ prediction }) => {
  const surfaceTemp = prediction.temperatures[0];
  const meanSigma = (
    prediction.sigma.reduce((acc, curr) => acc + curr, 0) / prediction.sigma.length
  ).toFixed(2);
  const latency = prediction.inference_latency_ms.toFixed(1);

  return (
    <div className="metrics-row">
      <MetricBlock
        label="Surface Temperature"
        value={surfaceTemp.toFixed(2)}
        unit="°C"
        subtext="Depth: 0 m level"
      />
      <MetricBlock
        label="Mean Uncertainty"
        value={`±${meanSigma}`}
        unit="°C"
        subtext="Mean σ across 15 levels"
      />
      <MetricBlock
        label="Depth Range"
        value="0–1000"
        unit="m"
        subtext="15 standardized levels"
      />
      <MetricBlock
        label="Inference Latency"
        value={latency}
        unit="ms"
        subtext="Forward pass (CPU)"
      />
    </div>
  );
};
