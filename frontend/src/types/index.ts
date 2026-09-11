/**
 * OceanEmbed API TypeScript Definitions
 * Directly matching FastAPI Pydantic Schemas
 */

export interface PredictionRequest {
  latitude: number;
  longitude: number;
  date: string;
  data_mode: 'synthetic' | 'real';
}

export interface PredictionResponse {
  latitude: number;
  longitude: number;
  date: string;
  depths: number[];
  temperatures: number[];
  sigma: number[];
  lower_90: number[];
  upper_90: number[];
  regime_probs: number[];
  embedding: number[];
  climatology_prior: number[] | null;
  anomaly: number[] | null;
  surface_inputs: Record<string, number> | null;
  inference_latency_ms: number;
  data_mode: string;
}

export interface ModelInfo {
  available: boolean;
  version: string;
}

export interface DataInfo {
  satellite: boolean;
  argo: boolean;
}

export interface HealthResponse {
  status: 'ok' | 'degraded';
  model: ModelInfo;
  data: DataInfo;
  mode: 'synthetic_demo' | 'real';
}

export interface DomainInfo {
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
}

export interface ModelStatusResponse {
  version: string;
  checkpoint_path: string;
  checkpoint_exists: boolean;
  checkpoint_size_mb: number;
  is_loaded: boolean;
  embedding_dim: number;
  num_depths: number;
  target_depths: number[];
  num_regimes: number;
  domain: DomainInfo;
  surface_channels: string[];
  load_error: string | null;
}

export interface ArgoStatusResponse {
  available: boolean;
  files_found: number;
  profiles_available: number;
  directory: string;
  message: string;
  geographic_coverage: Record<string, unknown> | null;
  depth_coverage: Record<string, unknown> | null;
  temporal_coverage: Record<string, unknown> | null;
}

export interface SatelliteStatusResponse {
  available: boolean;
  mode: string;
  provider: string | null;
  variables: string[];
  coverage: Record<string, unknown>;
  message: string;
}

export interface BenchmarkResult {
  name: string;
  architecture: string;
  rmse: number;
  mae: number;
  bias: number;
  samples: string;
}
