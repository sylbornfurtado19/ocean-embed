import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer className="app-footer">
      <div className="footer-content">
        <div>
          <span style={{ fontWeight: 600, color: '#f8fafc' }}>OCEANEMBED</span>
          <span style={{ margin: '0 8px' }}>|</span>
          <span>Smart India Hackathon 2026</span>
          <span style={{ margin: '0 8px' }}>•</span>
          <span>Problem Statement 26066 (Disaster Management)</span>
          <span style={{ margin: '0 8px' }}>•</span>
          <span>Team Bug Dealers</span>
        </div>
        <div style={{ color: '#64748b' }}>
          Architecture: CNN + ConvLSTM + CBAM Spatial/Temporal Attention + GMM Latent Regimes (K=4)
        </div>
      </div>
    </footer>
  );
};
