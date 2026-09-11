import React from 'react';

export type StatusVariant = 'operational' | 'available' | 'unavailable' | 'warning';

interface StatusBadgeProps {
  label: string;
  variant: StatusVariant;
  text?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ label, variant, text }) => {
  return (
    <div className="status-item">
      <span className="status-label">{label}:</span>
      <span className="status-value">
        <span className={`status-dot ${variant}`} />
        {text || (variant === 'operational' ? 'Operational' : variant === 'available' ? 'Available' : 'Not Available')}
      </span>
    </div>
  );
};
