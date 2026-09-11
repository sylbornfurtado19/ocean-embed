import type {
  ArgoStatusResponse,
  HealthResponse,
  ModelStatusResponse,
  PredictionRequest,
  PredictionResponse,
  SatelliteStatusResponse,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...(options?.headers || {}),
        },
      });

      if (!response.ok) {
        let errorDetail = `HTTP ${response.status}: ${response.statusText}`;
        try {
          const errorJson = await response.json();
          if (errorJson.detail) {
            errorDetail = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
          }
        } catch {
          // Response body was not JSON
        }
        throw new Error(errorDetail);
      }

      return (await response.json()) as T;
    } catch (err: unknown) {
      if (err instanceof Error) {
        throw err;
      }
      throw new Error('An unknown network error occurred while communicating with the OceanEmbed API.');
    }
  }

  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }

  async getModelStatus(): Promise<ModelStatusResponse> {
    return this.request<ModelStatusResponse>('/model/status');
  }

  async predictProfile(payload: PredictionRequest): Promise<PredictionResponse> {
    return this.request<PredictionResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async getArgoStatus(): Promise<ArgoStatusResponse> {
    return this.request<ArgoStatusResponse>('/data/argo/status');
  }

  async getSatelliteStatus(): Promise<SatelliteStatusResponse> {
    return this.request<SatelliteStatusResponse>('/data/satellite/status');
  }
}

export const apiClient = new ApiClient(API_BASE_URL);
