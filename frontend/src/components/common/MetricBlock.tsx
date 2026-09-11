import React from 'react';

interface MetricBlockProps {
  label: string;
  value: string | number;
  unit?: string;
  subtext?: string;
}

export const MetricBlock: React.FC<MetricBlockProps> = ({ label, value, unit, subtext }) => {
  return (
    <div className="metric-block">
      <div className="metric-label">{label}</div>
      <div className="metric-value">
        {value}
        {unit && <span className="metric-unit">{unit}</span>}
      </div>
      {subtext && <div className="metric-sub">{subtext}</div>}
    </div>
  );
};
