import React from 'react';
import { CompassIcon } from '../common/Icons';

export type NavPage = 'reconstruction' | 'insights' | 'overview' | 'validation' | 'system';

interface TopNavProps {
  currentPage: NavPage;
  onPageChange: (page: NavPage) => void;
  apiConnected: boolean;
}

export const TopNav: React.FC<TopNavProps> = ({ currentPage, onPageChange, apiConnected }) => {
  return (
    <header className="top-nav">
      <div className="nav-brand-group">
        <button
          type="button"
          className="nav-brand-title-btn"
          onClick={() => onPageChange('reconstruction')}
          title="Return to Reconstruction Workspace"
        >
          <CompassIcon size={22} style={{ color: '#38bdf8' }} />
          <span>OCEANEMBED</span>
        </button>
        <span className="nav-brand-tag">SIH 2026</span>
        <div className="nav-brand-divider" />
        <button
          type="button"
          className={`nav-brand-subtitle-btn ${currentPage === 'insights' ? 'active' : ''}`}
          onClick={() => onPageChange('insights')}
          title="Open Ocean Insights & Analytics"
        >
          Ocean Insights
        </button>
      </div>

      <nav className="nav-links">
        <button
          type="button"
          className={`nav-link ${currentPage === 'reconstruction' ? 'active' : ''}`}
          onClick={() => onPageChange('reconstruction')}
        >
          Reconstruction
        </button>
        <button
          type="button"
          className={`nav-link ${currentPage === 'insights' ? 'active' : ''}`}
          onClick={() => onPageChange('insights')}
        >
          Ocean Insights
        </button>
        <button
          type="button"
          className={`nav-link ${currentPage === 'overview' ? 'active' : ''}`}
          onClick={() => onPageChange('overview')}
        >
          Overview
        </button>
        <button
          type="button"
          className={`nav-link ${currentPage === 'validation' ? 'active' : ''}`}
          onClick={() => onPageChange('validation')}
        >
          Validation
        </button>
        <button
          type="button"
          className={`nav-link ${currentPage === 'system' ? 'active' : ''}`}
          onClick={() => onPageChange('system')}
        >
          System
        </button>
      </nav>

      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div className="status-item">
          <span className="status-label" style={{ color: '#94a3b8' }}>API:</span>
          <span className="status-value" style={{ fontSize: '0.75rem' }}>
            <span className={`status-dot ${apiConnected ? 'operational' : 'error'}`} />
            {apiConnected ? 'Operational' : 'Disconnected'}
          </span>
        </div>
      </div>
    </header>
  );
};
