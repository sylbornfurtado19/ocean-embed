import React from 'react';
import { CompassIcon } from '../common/Icons';

export type NavPage = 'reconstruction' | 'insights' | 'overview' | 'validation' | 'system';

interface TopNavProps {
  currentPage: NavPage;
  onPageChange: (page: NavPage) => void;
  apiConnected: boolean;
}

export const TopNav: React.FC<TopNavProps> = ({ currentPage, onPageChange }) => {
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
    </header>
  );
};
