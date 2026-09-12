import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { LayersIcon, MapPinIcon } from '../common/Icons';

interface BayOfBengalMapProps {
  latitude: number;
  longitude: number;
  onSelectCoordinates: (lat: number, lon: number) => void;
}

// Bounding box for Bay of Bengal domain
const DOMAIN_BOUNDS = {
  latMin: 5.0,
  latMax: 23.0,
  lonMin: 80.0,
  lonMax: 100.0,
};

export type MapTileMode = 'dark' | 'satellite' | 'ocean';

const TILE_LAYERS: Record<MapTileMode, { url: string; referenceUrl?: string; attribution: string; name: string }> = {
  dark: {
    name: 'Dark Ocean',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    referenceUrl: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
  },
  satellite: {
    name: 'Satellite',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
  },
  ocean: {
    name: 'Ocean Topo',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
    referenceUrl: 'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Reference/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Tiles &copy; Esri &mdash; Sources: GEBCO, NOAA, CHS, OSU, UNH, CSUMB, National Geographic, DeLorme, NAVTEQ, and Esri',
  },
};

const createLayerForMode = (mode: MapTileMode): L.Layer => {
  const config = TILE_LAYERS[mode];
  const baseLayer = L.tileLayer(config.url, {
    attribution: config.attribution,
    maxZoom: 16,
  });

  if (config.referenceUrl) {
    const refLayer = L.tileLayer(config.referenceUrl, {
      maxZoom: 16,
    });
    return L.layerGroup([baseLayer, refLayer]);
  }

  return baseLayer;
};

// Glowing pulse marker icon with crosshairs using HTML divIcon
const createPulseIcon = () => {
  return L.divIcon({
    className: 'custom-pulse-marker',
    html: `
      <div class="ocean-pulse-marker">
        <div class="ocean-pulse-ring"></div>
        <div class="ocean-crosshair-h"></div>
        <div class="ocean-crosshair-v"></div>
        <div class="ocean-pulse-core"></div>
      </div>
    `,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
  });
};

export const BayOfBengalMap: React.FC<BayOfBengalMapProps> = ({
  latitude,
  longitude,
  onSelectCoordinates,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markerRef = useRef<L.Marker | null>(null);
  const tileLayerRef = useRef<L.Layer | null>(null);

  const [tileMode, setTileMode] = useState<MapTileMode>('dark');

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    // Initialize Leaflet Map centered on Bay of Bengal
    const map = L.map(mapContainerRef.current, {
      center: [14.5, 90.0],
      zoom: 5,
      minZoom: 4,
      maxZoom: 10,
      zoomControl: false,
      scrollWheelZoom: false,
    });

    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // Initial tile layer (Dark Ocean)
    const initialTileLayer = createLayerForMode(tileMode);
    initialTileLayer.addTo(map);
    tileLayerRef.current = initialTileLayer;

    // Draw Supported Domain Bounding Box with Cyan Neon Glow
    const domainPolygon = L.polygon(
      [
        [DOMAIN_BOUNDS.latMin, DOMAIN_BOUNDS.lonMin],
        [DOMAIN_BOUNDS.latMax, DOMAIN_BOUNDS.lonMin],
        [DOMAIN_BOUNDS.latMax, DOMAIN_BOUNDS.lonMax],
        [DOMAIN_BOUNDS.latMin, DOMAIN_BOUNDS.lonMax],
      ],
      {
        color: '#0ea5e9',
        weight: 2,
        dashArray: '6, 6',
        fillColor: '#0ea5e9',
        fillOpacity: 0.08,
      }
    ).addTo(map);

    domainPolygon.bindTooltip('Supported Ocean Domain (5°N–23°N, 80°E–100°E)', {
      permanent: false,
      direction: 'top',
      className: 'domain-tooltip',
    });

    // Add glowing pulse coordinate marker
    const marker = L.marker([latitude, longitude], {
      icon: createPulseIcon(),
      draggable: true,
    }).addTo(map);

    marker.on('dragend', () => {
      const pos = marker.getLatLng();
      const clampedLat = Math.max(DOMAIN_BOUNDS.latMin, Math.min(DOMAIN_BOUNDS.latMax, pos.lat));
      const clampedLon = Math.max(DOMAIN_BOUNDS.lonMin, Math.min(DOMAIN_BOUNDS.lonMax, pos.lng));
      marker.setLatLng([clampedLat, clampedLon]);
      onSelectCoordinates(Number(clampedLat.toFixed(4)), Number(clampedLon.toFixed(4)));
    });

    // Map click handler to update target coordinates
    map.on('click', (e: L.LeafletMouseEvent) => {
      const { lat, lng } = e.latlng;
      if (
        lat >= DOMAIN_BOUNDS.latMin &&
        lat <= DOMAIN_BOUNDS.latMax &&
        lng >= DOMAIN_BOUNDS.lonMin &&
        lng <= DOMAIN_BOUNDS.lonMax
      ) {
        marker.setLatLng([lat, lng]);
        onSelectCoordinates(Number(lat.toFixed(4)), Number(lng.toFixed(4)));
      }
    });

    mapInstanceRef.current = map;
    markerRef.current = marker;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Handle Tile Mode Changes
  useEffect(() => {
    if (!mapInstanceRef.current) return;
    const map = mapInstanceRef.current;

    if (tileLayerRef.current) {
      map.removeLayer(tileLayerRef.current);
    }

    const newLayer = createLayerForMode(tileMode);
    newLayer.addTo(map);
    tileLayerRef.current = newLayer;
  }, [tileMode]);

  // Update marker position when coordinates change from outside
  useEffect(() => {
    if (markerRef.current) {
      markerRef.current.setLatLng([latitude, longitude]);
    }
  }, [latitude, longitude]);

  return (
    <div
      className="card"
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        position: 'relative',
        overflow: 'hidden',
        isolation: 'isolate',
      }}
    >
      <div className="card-header">
        <div>
          <div className="card-title">
            <MapPinIcon size={18} style={{ color: '#0ea5e9' }} />
            <span>Bay of Bengal Spatial Domain</span>
          </div>
          <div className="card-subtitle">
            5.0°N–23.0°N | 80.0°E–100.0°E
          </div>
        </div>

        {/* Custom Layer Switcher */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <LayersIcon size={14} style={{ color: '#64748b' }} />
          <div
            style={{
              display: 'flex',
              background: 'rgba(7, 18, 36, 0.9)',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '6px',
              padding: '2px',
            }}
          >
            {(['dark', 'satellite', 'ocean'] as MapTileMode[]).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setTileMode(mode)}
                style={{
                  fontSize: '0.6875rem',
                  fontWeight: tileMode === mode ? 700 : 500,
                  padding: '3px 8px',
                  borderRadius: '4px',
                  background: tileMode === mode ? 'rgba(14, 165, 233, 0.25)' : 'transparent',
                  color: tileMode === mode ? '#38bdf8' : '#94a3b8',
                  border: tileMode === mode ? '1px solid rgba(14, 165, 233, 0.4)' : '1px solid transparent',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                {TILE_LAYERS[mode].name}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div style={{ flex: 1, minHeight: '400px', position: 'relative', overflow: 'hidden', borderRadius: '0 0 var(--radius-md) var(--radius-md)' }}>
        <div ref={mapContainerRef} style={{ width: '100%', height: '100%', minHeight: '400px' }} />

        {/* Dynamic Glassmorphic Floating Coordinate Badge Overlay */}
        <div
          style={{
            position: 'absolute',
            bottom: '16px',
            left: '16px',
            background: 'rgba(5, 12, 26, 0.85)',
            backdropFilter: 'blur(12px)',
            WebkitBackdropFilter: 'blur(12px)',
            border: '1px solid rgba(14, 165, 233, 0.3)',
            borderRadius: '8px',
            padding: '10px 14px',
            boxShadow: '0 8px 20px rgba(0, 0, 0, 0.5)',
            zIndex: 400,
            display: 'flex',
            alignItems: 'center',
            gap: '14px',
          }}
        >
          <div>
            <div style={{ fontSize: '0.625rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700 }}>
              Active Coordinate Target
            </div>
            <div style={{ fontSize: '0.9375rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>
              {latitude.toFixed(2)}°N, {longitude.toFixed(2)}°E
            </div>
          </div>

          <div style={{ width: '1px', height: '24px', background: 'rgba(255,255,255,0.1)' }} />

          <div>
            <div style={{ fontSize: '0.625rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700 }}>
              Depth Profiling Range
            </div>
            <div style={{ fontSize: '0.8125rem', fontWeight: 600, fontFamily: 'var(--font-mono)', color: '#f1f5f9' }}>
              0 – 1000 m (15 Levels)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
