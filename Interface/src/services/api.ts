export interface BackendHealth {
  status: string;
  service: string;
  version: string;
}

export interface SystemInfo {
  python_version: string;
  python_executable: string;
  platform: string;
  torch_version: string | null;
  cuda: {
    available: boolean;
    name: string | null;
    memory_total_bytes: number | null;
    memory_allocated_bytes: number | null;
    memory_reserved_bytes: number | null;
    memory_free_bytes: number | null;
    version: string | null;
  };
  system_ram: {
    total_bytes: number;
    available_bytes: number;
    used_bytes: number;
    percent_used: number;
  };
}

export interface Checkpoint {
  id: string;
  name: string;
  path: string;
  format: string;
  precision: string;
  size_bytes: number;
  parameters: number | null;
  lineage: string;
  multimodal: boolean;
  available: boolean;
}

export interface CalibrationFile {
  id: string;
  name: string;
  path: string;
  size_bytes: number;
}

export interface JobLog {
  timestamp: string;
  level: string;
  message: string;
}

export interface Job {
  id: string;
  operation: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  progress: number | null;
  request: Record<string, unknown>;
  error: string | null;
  result: Record<string, unknown> | null;
  logs: JobLog[];
}

export interface ModelRecord {
  id: string;
  name: string;
  family: string;
  backbone: string;
  architecture_importable: boolean;
  registered_checkpoints: Checkpoint[];
  capabilities: Record<string, unknown>;
}

export interface BenchmarkReport {
  status: 'measured' | 'not_measured';
  items: Array<{ path: string; origin: string; imported_at: string | null; source_label: string | null; metadata: Record<string, unknown>; results: Record<string, unknown> }>;
}

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers }
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload?.detail;
    throw new Error(Array.isArray(detail) ? detail.map((item: { msg?: string }) => item.msg).join('; ') : detail || `Backend request failed (${response.status})`);
  }
  return payload as T;
}

export const apiService = {
  baseUrl: apiBaseUrl,
  health: () => request<BackendHealth>('/api/health'),
  system: () => request<SystemInfo>('/api/system/info'),
  models: () => request<ModelRecord[]>('/api/models'),
  checkpoints: () => request<Checkpoint[]>('/api/checkpoints'),
  calibrationFiles: () => request<CalibrationFile[]>('/api/calibration-data'),
  jobs: () => request<Job[]>('/api/jobs'),
  job: (id: string) => request<Job>(`/api/jobs/${encodeURIComponent(id)}`),
  submitJob: (payload: Record<string, unknown>) => request<{ job_id: string; status: string; message: string }>('/api/jobs', { method: 'POST', body: JSON.stringify(payload) }),
  cancelJob: (id: string) => request<Job>(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' }),
  benchmarks: () => request<BenchmarkReport>('/api/benchmarks'),
  importResults: (manifest: Record<string, unknown>, sourceLabel?: string) => request<{ import_id: string; path: string; origin: string; imported_at: string; measured: boolean }>('/api/results/import', { method: 'POST', body: JSON.stringify({ manifest, source_label: sourceLabel }) }),
  capabilities: () => request<{ operations: Record<string, { status: string; reason?: string; [key: string]: unknown }> }>('/api/pipeline/capabilities')
};
