import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { MapPinIcon } from '../common/Icons';

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

// Clean scientific target marker icon using SVG
const createTargetIcon = () => {
  return L.divIcon({
    className: 'custom-map-marker',
    html: `
      <div style="
        position: relative;
        width: 24px;
        height: 24px;
        transform: translate(-12px, -12px);
      ">
        <div style="
          position: absolute;
          width: 24px;
          height: 24px;
          border-radius: 50%;
          background: rgba(2, 132, 199, 0.25);
          border: 1.5px solid #0284c7;
        "></div>
        <div style="
          position: absolute;
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #0284c7;
          top: 8px;
          left: 8px;
          box-shadow: 0 0 4px rgba(0,0,0,0.5);
        "></div>
        <div style="
          position: absolute;
          width: 28px;
          height: 1px;
          background: #0284c7;
          top: 11.5px;
          left: -2px;
        "></div>
        <div style="
          position: absolute;
          width: 1px;
          height: 28px;
          background: #0284c7;
          top: -2px;
          left: 11.5px;
        "></div>
      </div>
    `,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
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

  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    // Initialize Leaflet Map centered on Bay of Bengal
    const map = L.map(mapContainerRef.current, {
      center: [14.5, 90.0],
      zoom: 5,
      minZoom: 4,
      maxZoom: 10,
      zoomControl: false,
    });

    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // CartoDB Positron Basemap - Clean, subdued, research-grade
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; OpenStreetMap',
      subdomains: 'abcd',
      maxZoom: 19,
    }).addTo(map);

    // Draw Supported Domain Bounding Box
    const domainPolygon = L.polygon(
      [
        [DOMAIN_BOUNDS.latMin, DOMAIN_BOUNDS.lonMin],
        [DOMAIN_BOUNDS.latMax, DOMAIN_BOUNDS.lonMin],
        [DOMAIN_BOUNDS.latMax, DOMAIN_BOUNDS.lonMax],
        [DOMAIN_BOUNDS.latMin, DOMAIN_BOUNDS.lonMax],
      ],
      {
        color: '#0284c7',
        weight: 1.5,
        dashArray: '4, 4',
        fillColor: '#0284c7',
        fillOpacity: 0.05,
      }
    ).addTo(map);

    domainPolygon.bindTooltip('Supported Domain (5°N–23°N, 80°E–100°E)', {
      permanent: false,
      direction: 'top',
      className: 'domain-tooltip',
    });

    // Add selected coordinate marker
    const marker = L.marker([latitude, longitude], {
      icon: createTargetIcon(),
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

  // Update marker position when coordinates change from outside
  useEffect(() => {
    if (markerRef.current) {
      markerRef.current.setLatLng([latitude, longitude]);
    }
  }, [latitude, longitude]);

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div className="card-header">
        <div>
          <div className="card-title">
            <MapPinIcon size={16} style={{ color: '#0284c7' }} />
            <span>Bay of Bengal Spatial Domain</span>
          </div>
          <div className="card-subtitle">
            Domain: 5.0°N–23.0°N | 80.0°E–100.0°E
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '0.6875rem', color: '#64748b', fontWeight: 600, textTransform: 'uppercase' }}>
            Selected Position
          </div>
          <div style={{ fontSize: '0.8125rem', fontWeight: 600, fontFamily: 'var(--font-mono)', color: '#0f172a' }}>
            {latitude.toFixed(2)}°N, {longitude.toFixed(2)}°E
          </div>
        </div>
      </div>
      <div style={{ flex: 1, minHeight: '380px', position: 'relative' }}>
        <div ref={mapContainerRef} style={{ width: '100%', height: '100%', minHeight: '380px' }} />
        <div
          style={{
            position: 'absolute',
            bottom: '12px',
            left: '12px',
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            border: '1px solid #cbd5e1',
            borderRadius: '4px',
            padding: '4px 8px',
            fontSize: '0.6875rem',
            color: '#475569',
            pointerEvents: 'none',
            zIndex: 400,
          }}
        >
          Click within domain bounding box to update target coordinate
        </div>
      </div>
    </div>
  );
};
