import React from 'react';
import type { PredictionResponse } from '../types';
import { ActivityIcon, CompassIcon, LayersIcon } from '../components/common/Icons';

interface InsightsPageProps {
  prediction: PredictionResponse | null;
  onNavigateReconstruction: () => void;
}

export const InsightsPage: React.FC<InsightsPageProps> = ({
  prediction,
  onNavigateReconstruction,
}) => {
  // Compute oceanographic physical metrics if prediction is available
  let d26: number | null = null;
  let d20: number | null = null;
  let mld: number | null = null;
  let tchp: number | null = null; // kJ/cm^2
  let maxGradient: number = 0; // degC/m
  let maxGradientDepth: number = 0;

  if (prediction && prediction.depths && prediction.temperatures) {
    const depths = prediction.depths;
    const temps = prediction.temperatures;

    // Surface temperature
    const sst = temps[0];

    // MLD: depth where temp drops by 0.2 deg C from surface
    for (let i = 1; i < depths.length; i++) {
      if (sst - temps[i] >= 0.2) {
        const frac = (0.2 - (sst - temps[i - 1])) / (temps[i - 1] - temps[i] || 1e-6);
        mld = Number((depths[i - 1] + frac * (depths[i] - depths[i - 1])).toFixed(1));
        break;
      }
    }
    if (mld === null && depths.length > 0) mld = depths[depths.length - 1];

    // D26 isotherm depth
    if (sst >= 26.0) {
      for (let i = 1; i < depths.length; i++) {
        if (temps[i] <= 26.0) {
          const frac = (temps[i - 1] - 26.0) / (temps[i - 1] - temps[i] || 1e-6);
          d26 = Number((depths[i - 1] + frac * (depths[i] - depths[i - 1])).toFixed(1));
          break;
        }
      }
      if (d26 === null) d26 = depths[depths.length - 1];
    } else {
      d26 = 0;
    }

    // D20 isotherm depth
    for (let i = 1; i < depths.length; i++) {
      if (temps[i] <= 20.0) {
        const frac = (temps[i - 1] - 20.0) / (temps[i - 1] - temps[i] || 1e-6);
        d20 = Number((depths[i - 1] + frac * (depths[i] - depths[i - 1])).toFixed(1));
        break;
      }
    }
    if (d20 === null && depths.length > 0) d20 = depths[depths.length - 1];

    // TCHP estimation (Tropical Cyclone Heat Potential in kJ/cm^2)
    // TCHP = rho * Cp * integral_{0}^{D26} (T(z) - 26) dz
    // rho = 1025 kg/m^3, Cp = 3985 J/(kg*C) -> rho * Cp ~ 4.084 x 10^6 J/(m^3*C) = 4.084 kJ/(cm^2 * m * C) * 0.01 = 0.04084 kJ/(cm^2 * m)
    let heatSum = 0;
    for (let i = 1; i < depths.length; i++) {
      const z0 = depths[i - 1];
      const z1 = depths[i];
      const t0 = temps[i - 1];
      const t1 = temps[i];

      if (z1 <= (d26 ?? 0)) {
        const dz = z1 - z0;
        const avgDeltaT = Math.max(0, (t0 + t1) / 2 - 26.0);
        heatSum += avgDeltaT * dz;
      } else if (z0 < (d26 ?? 0)) {
        const dz = (d26 ?? 0) - z0;
        const avgDeltaT = Math.max(0, (t0 - 26.0) / 2);
        heatSum += avgDeltaT * dz;
        break;
      }
    }
    // Convert to kJ / cm^2 (rho * cp * 1e-4) -> 0.4084 kJ / (cm^2 * C * m)
    tchp = Number((heatSum * 0.4084).toFixed(1));

    // Maximum vertical gradient (thermocline steepness)
    for (let i = 1; i < depths.length; i++) {
      const dz = depths[i] - depths[i - 1];
      const grad = Math.abs(temps[i] - temps[i - 1]) / dz;
      if (grad > maxGradient) {
        maxGradient = grad;
        maxGradientDepth = (depths[i] + depths[i - 1]) / 2;
      }
    }
  }

  // TCHP Severity Level
  const getTCHPRisk = (val: number | null) => {
    if (val === null || val === 0) return { label: 'Negligible', color: '#64748b', bg: '#f1f5f9' };
    if (val < 50) return { label: 'Moderate', color: '#0284c7', bg: '#e0f2fe' };
    if (val < 80) return { label: 'High Potential', color: '#d97706', bg: '#fef3c7' };
    return { label: 'Extreme Intensification Risk', color: '#dc2626', bg: '#fee2e2' };
  };

  const tchpRisk = getTCHPRisk(tchp);

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', paddingBottom: '40px' }}>
      {/* Hero Header */}
      <div style={{ marginBottom: '28px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.05em', color: '#0284c7', textTransform: 'uppercase', backgroundColor: '#e0f2fe', padding: '3px 8px', borderRadius: '4px' }}>
              SIH 2026 Problem Statement 26066
            </span>
            <span style={{ fontSize: '0.75rem', color: '#64748b' }}>• Disaster Management Theme</span>
          </div>
          <h1 style={{ fontSize: '1.75rem', fontWeight: 700, color: '#0f172a', margin: '0 0 6px 0' }}>
            Oceanographic Insights & Disaster Intelligence
          </h1>
          <p style={{ fontSize: '0.9375rem', color: '#64748b', margin: 0, maxWidth: '750px', lineHeight: 1.5 }}>
            Subsurface thermal energy analytics for the Bay of Bengal, quantifying Tropical Cyclone Heat Potential (TCHP), thermocline stratification, and barrier layer dynamics directly from continuous satellite surface embeddings.
          </p>
        </div>

        <button
          type="button"
          onClick={onNavigateReconstruction}
          className="btn btn-primary"
          style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 18px' }}
        >
          <CompassIcon size={16} />
          <span>Launch 3D Reconstructor</span>
        </button>
      </div>

      {/* Live Profile Indicator Banner */}
      {prediction ? (
        <div
          style={{
            backgroundColor: '#f8fafc',
            border: '1px solid #cbd5e1',
            borderRadius: '8px',
            padding: '16px 20px',
            marginBottom: '28px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '16px',
          }}
        >
          <div>
            <div style={{ fontSize: '0.8125rem', color: '#64748b', fontWeight: 500 }}>
              Active Reconstruction Location & Observation Time
            </div>
            <div style={{ fontSize: '1.125rem', fontWeight: 700, color: '#0f172a', display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span>{prediction.latitude.toFixed(2)}°N, {prediction.longitude.toFixed(2)}°E</span>
              <span style={{ fontSize: '0.875rem', fontWeight: 500, color: '#0284c7', backgroundColor: '#e0f2fe', padding: '2px 8px', borderRadius: '4px' }}>
                Date: {prediction.date}
              </span>
              <span style={{ fontSize: '0.8125rem', fontWeight: 500, color: '#059669', backgroundColor: '#d1fae5', padding: '2px 8px', borderRadius: '4px' }}>
                {prediction.data_mode}
              </span>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.8125rem', color: '#475569' }}>Cyclone Intensification Potential:</span>
            <span
              style={{
                fontSize: '0.8125rem',
                fontWeight: 700,
                color: tchpRisk.color,
                backgroundColor: tchpRisk.bg,
                padding: '4px 10px',
                borderRadius: '6px',
                border: `1px solid ${tchpRisk.color}40`,
              }}
            >
              {tchpRisk.label} ({tchp ?? 0} kJ/cm²)
            </span>
          </div>
        </div>
      ) : (
        <div
          style={{
            backgroundColor: '#eff6ff',
            border: '1px solid #bfdbfe',
            borderRadius: '8px',
            padding: '16px 20px',
            marginBottom: '28px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <div style={{ fontSize: '0.875rem', color: '#1e40af' }}>
            Showing baseline oceanographic metrics. Run a prediction on the <strong>Reconstruction</strong> tab to compute real-time physical metrics for any coordinates.
          </div>
          <button
            type="button"
            onClick={onNavigateReconstruction}
            className="btn btn-primary"
            style={{ fontSize: '0.8125rem', padding: '6px 14px' }}
          >
            Compute Live Metrics
          </button>
        </div>
      )}

      {/* Physics KPI Metrics Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '32px' }}>
        <div className="card" style={{ padding: '16px' }}>
          <div style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 600, textTransform: 'uppercase', marginBottom: '4px' }}>
            Tropical Cyclone Heat (TCHP)
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#0f172a', fontFamily: 'var(--font-mono)' }}>
            {tchp !== null ? `${tchp}` : '62.4'} <span style={{ fontSize: '0.875rem', fontWeight: 500, color: '#64748b' }}>kJ/cm²</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#0284c7', marginTop: '4px' }}>
            Energy fuel for rapid intensification
          </div>
        </div>

        <div className="card" style={{ padding: '16px' }}>
          <div style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 600, textTransform: 'uppercase', marginBottom: '4px' }}>
            26°C Isotherm Depth (D26)
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#0f172a', fontFamily: 'var(--font-mono)' }}>
            {d26 !== null ? `${d26}` : '58.2'} <span style={{ fontSize: '0.875rem', fontWeight: 500, color: '#64748b' }}>m</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#059669', marginTop: '4px' }}>
            Thermal reservoir base boundary
          </div>
        </div>

        <div className="card" style={{ padding: '16px' }}>
          <div style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 600, textTransform: 'uppercase', marginBottom: '4px' }}>
            Mixed Layer Depth (MLD)
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#0f172a', fontFamily: 'var(--font-mono)' }}>
            {mld !== null ? `${mld}` : '24.5'} <span style={{ fontSize: '0.875rem', fontWeight: 500, color: '#64748b' }}>m</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#d97706', marginTop: '4px' }}>
            ΔT = 0.2°C surface mixed zone
          </div>
        </div>

        <div className="card" style={{ padding: '16px' }}>
          <div style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 600, textTransform: 'uppercase', marginBottom: '4px' }}>
            Main Thermocline (D20)
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#0f172a', fontFamily: 'var(--font-mono)' }}>
            {d20 !== null ? `${d20}` : '112.0'} <span style={{ fontSize: '0.875rem', fontWeight: 500, color: '#64748b' }}>m</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#7c3aed', marginTop: '4px' }}>
            Baroclinic ocean energy boundary
          </div>
        </div>

        <div className="card" style={{ padding: '16px' }}>
          <div style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 600, textTransform: 'uppercase', marginBottom: '4px' }}>
            Max Vertical Gradient
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#0f172a', fontFamily: 'var(--font-mono)' }}>
            {maxGradient > 0 ? maxGradient.toFixed(3) : '0.145'} <span style={{ fontSize: '0.875rem', fontWeight: 500, color: '#64748b' }}>°C/m</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#475569', marginTop: '4px' }}>
            Peak steepness at {maxGradientDepth > 0 ? `${maxGradientDepth.toFixed(0)} m` : '85 m'}
          </div>
        </div>
      </div>

      {/* 4 Thematic Insight Panels */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '32px' }}>
        {/* Section 1: Tropical Cyclones */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <ActivityIcon size={18} style={{ color: '#dc2626' }} />
              <span>1. Tropical Cyclone Intensification in Bay of Bengal</span>
            </div>
          </div>
          <div className="card-body" style={{ fontSize: '0.875rem', color: '#475569', lineHeight: 1.6 }}>
            <p style={{ marginBottom: '12px' }}>
              The Bay of Bengal is world-renowned for catastrophic cyclone disasters. While satellite radiometers measure Sea Surface Temperature (SST), SST alone is notoriously misleading: strong cyclones produce intense wind stress that churns up subsurface water.
            </p>
            <p style={{ marginBottom: '12px' }}>
              <strong>The Subsurface Mechanism:</strong> If the warm layer is thin (D26 &lt; 30 m), cold subsurface water is quickly brought to the surface, creating a "cold wake" that starves the cyclone of energy. Conversely, if D26 &gt; 60 m and TCHP &gt; 60 kJ/cm², cyclonic mixing simply circulates warm water back to the surface, causing <strong>explosive, rapid intensification</strong> into Category 4 or 5 Super Cyclonic Storms before landfall.
            </p>
            <div style={{ backgroundColor: '#fff1f2', border: '1px solid #fecdd3', borderRadius: '6px', padding: '12px', color: '#9f1239', fontSize: '0.8125rem' }}>
              <strong>Operational Advantage:</strong> OceanEmbed delivers continuous 0–1000 m thermal profiles, enabling meteorologists to calculate TCHP along forecasted cyclone tracks days before storm arrival.
            </div>
          </div>
        </div>

        {/* Section 2: Salinity Barrier Layer */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <LayersIcon size={18} style={{ color: '#0284c7' }} />
              <span>2. Freshwater River Runoff & Barrier Layer Dynamics</span>
            </div>
          </div>
          <div className="card-body" style={{ fontSize: '0.875rem', color: '#475569', lineHeight: 1.6 }}>
            <p style={{ marginBottom: '12px' }}>
              Unlike the Arabian Sea, the Northern Bay of Bengal receives colossal freshwater runoff from the Ganges-Brahmaputra-Meghna and Irrawaddy river networks, accompanied by intense summer monsoon precipitation.
            </p>
            <p style={{ marginBottom: '12px' }}>
              <strong>The Barrier Layer Phenomenon:</strong> Low-salinity water (&lt; 31 PSU) creates a strong halocline shallower than the isothermal layer depth. The distance between the Mixed Layer Depth (MLD) and the Isothermal Layer Depth (ILD) forms a <strong>Barrier Layer</strong>.
            </p>
            <div style={{ backgroundColor: '#f0f9ff', border: '1px solid #bae6fd', borderRadius: '6px', padding: '12px', color: '#0369a1', fontSize: '0.8125rem' }}>
              <strong>Heat Trapping Effect:</strong> This stable, buoyant freshwater lid strongly suppresses turbulent wind mixing, trapping incoming solar radiation beneath the surface and forming subsurface temperature inversions (T(z) &gt; T(0)).
            </div>
          </div>
        </div>

        {/* Section 3: Mesoscale Eddies */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <CompassIcon size={18} style={{ color: '#7c3aed' }} />
              <span>3. Mesoscale Eddies & Sea Level Anomaly (SLA)</span>
            </div>
          </div>
          <div className="card-body" style={{ fontSize: '0.875rem', color: '#475569', lineHeight: 1.6 }}>
            <p style={{ marginBottom: '12px' }}>
              The Bay of Bengal is populated with energetic cyclonic and anticyclonic mesoscale eddies spanning 100–300 km in diameter:
            </p>
            <ul style={{ paddingLeft: '20px', marginBottom: '12px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <li>
                <strong>Anticyclonic Eddies (Warm Core):</strong> Elevated Sea Level Anomaly (SLA &gt; 0). Pushes the thermocline downwards (D20 deepening up to 50 m), concentrating tremendous ocean heat content.
              </li>
              <li>
                <strong>Cyclonic Eddies (Cold Core):</strong> Depressed Sea Level Anomaly (SLA &lt; 0). Uplifts cold, nutrient-rich thermocline water toward the surface, promoting biological productivity while reducing TCHP.
              </li>
            </ul>
            <div style={{ backgroundColor: '#f5f3ff', border: '1px solid #ddd6fe', borderRadius: '6px', padding: '12px', color: '#5b21b6', fontSize: '0.8125rem' }}>
              <strong>Neural Spatial Feature Extractor:</strong> OceanEmbed's 32×32 spatial patch extracts spatial SLA gradients to accurately resolve eddy vertical pumping signatures without requiring physical float immersion.
            </div>
          </div>
        </div>

        {/* Section 4: Hydrographic Regimes */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <LayersIcon size={18} style={{ color: '#059669' }} />
              <span>4. Four Hydrographic Regimes in the Bay of Bengal</span>
            </div>
          </div>
          <div className="card-body" style={{ fontSize: '0.875rem', color: '#475569', lineHeight: 1.6 }}>
            <p style={{ marginBottom: '12px' }}>
              OceanEmbed's unsupervised latent regime classifier partitions the continuous 512-D ocean state into four distinct hydrographic regimes:
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div style={{ padding: '8px 12px', backgroundColor: '#f8fafc', borderLeft: '3px solid #0284c7', borderRadius: '4px' }}>
                <strong style={{ color: '#0f172a' }}>Regime 1: Northern Freshwater Pool</strong>
                <div style={{ fontSize: '0.75rem', color: '#64748b' }}>High stratification, shallow halocline, thick barrier layer (&gt;18°N).</div>
              </div>
              <div style={{ padding: '8px 12px', backgroundColor: '#f8fafc', borderLeft: '3px solid #059669', borderRadius: '4px' }}>
                <strong style={{ color: '#0f172a' }}>Regime 2: Central Basin Dynamic Eddy Field</strong>
                <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Moderate salinity, intense eddy baroclinic pumping (12°N–18°N).</div>
              </div>
              <div style={{ padding: '8px 12px', backgroundColor: '#f8fafc', borderLeft: '3px solid #d97706', borderRadius: '4px' }}>
                <strong style={{ color: '#0f172a' }}>Regime 3: Southern Bay & Sri Lanka Dome</strong>
                <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Equatorial water exchange, strong southwest monsoon drift (&lt;12°N).</div>
              </div>
              <div style={{ padding: '8px 12px', backgroundColor: '#f8fafc', borderLeft: '3px solid #7c3aed', borderRadius: '4px' }}>
                <strong style={{ color: '#0f172a' }}>Regime 4: Andaman Sea Semi-Enclosed Basin</strong>
                <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Deep sills, internal solitary waves, protected thermal reservoir.</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
