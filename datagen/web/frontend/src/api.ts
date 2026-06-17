import type { BuilderPayload, ConfigSummary, JobSnapshot, TemplateContext } from "./types";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    }
  });
  const data = await response.json();
  if (!response.ok) {
    const message = Array.isArray(data.errors) ? data.errors.join("\n") : response.statusText;
    throw new Error(message);
  }
  return data as T;
}

export const api = {
  configs: () => request<{ configs: ConfigSummary[] }>("/api/configs"),
  config: (name: string) => request<{ name: string; payload: BuilderPayload }>(`/api/configs/${encodeURIComponent(name)}`),
  previewConfig: (payload: BuilderPayload) =>
    request<{ ok: boolean; errors: string[]; toml: string }>("/api/configs/preview", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  createConfig: (name: string, payload: BuilderPayload, overwrite = false) =>
    request<{ ok: boolean; errors: string[]; name: string; toml: string }>("/api/configs", {
      method: "POST",
      body: JSON.stringify({ name, payload, overwrite })
    }),
  updateConfig: (name: string, payload: BuilderPayload) =>
    request<{ ok: boolean; errors: string[]; name: string; toml: string }>(`/api/configs/${encodeURIComponent(name)}`, {
      method: "PUT",
      body: JSON.stringify(payload)
    }),
  settings: () => request<{ models: Record<string, { backend: string; params: Record<string, unknown> }>; settings_path: string }>("/api/settings/models"),
  saveModel: (payload: Record<string, unknown>) =>
    request<{ ok: boolean; models: Record<string, unknown> }>("/api/settings/models", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  deleteModel: (name: string) =>
    request<{ ok: boolean; models: Record<string, unknown> }>(`/api/settings/models/${encodeURIComponent(name)}`, {
      method: "DELETE"
    }),
  sourcePreview: (source: { path: string; subset?: string; split: string }) =>
    request<{
      ok: boolean;
      errors: string[];
      columns: string[];
      rows: Record<string, unknown>[];
      table_rows: Record<string, { display: string; full: string }>[];
      column_metadata: Array<{ name: string; type: string; feature: string }>;
      dataset_metadata: Record<string, unknown>;
    }>("/api/sources/preview", {
      method: "POST",
      body: JSON.stringify({ ...source, limit: 5 })
    }),
  templateContext: (payload: BuilderPayload) =>
    request<TemplateContext>("/api/templates/context", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  renderPrompt: (payload: BuilderPayload, source: string, row: Record<string, unknown>) =>
    request<{ ok: boolean; errors: string[]; prompt: string; normalized_row: Record<string, unknown> }>("/api/templates/render-prompt", {
      method: "POST",
      body: JSON.stringify({ payload, source, row })
    }),
  trial: (payload: BuilderPayload, limit = 1) =>
    request<JobSnapshot>("/api/jobs/trial", {
      method: "POST",
      body: JSON.stringify({ payload, limit })
    }),
  generate: (configName: string) =>
    request<JobSnapshot>("/api/jobs/generate", {
      method: "POST",
      body: JSON.stringify({ config_name: configName })
    }),
  cancelJob: (jobId: string) =>
    request<JobSnapshot>(`/api/jobs/${jobId}/cancel`, {
      method: "POST",
      body: JSON.stringify({})
    })
};
