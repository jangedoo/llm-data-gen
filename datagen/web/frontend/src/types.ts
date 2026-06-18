export type ModelPayload = {
  kind?: "settings" | "custom";
  name: string;
  backend?: string;
  params?: Record<string, unknown> | string;
  overrides?: {
    params?: Record<string, unknown>;
  };
};

export type SavedModel = {
  backend: string;
  params: Record<string, unknown>;
};

export type SourcePayload = {
  name: string;
  path: string;
  subset?: string;
  split: string;
  columns?: string[];
  preview_rows?: Record<string, unknown>[];
};

export type SourceDatasetPayload = {
  name: string;
  model?: string;
  max_records: number | string;
  max_failures: number | string;
  shuffle: boolean;
  system_prompt?: string;
  prompt_template?: string;
};

export type BuilderPayload = {
  dataset_name: string;
  description: string;
  authors: string[];
  generation_output_dir: string;
  generation_logging_steps: number | string;
  sources: SourcePayload[];
  models: ModelPayload[];
  default_model: string;
  default_system_prompt: string;
  default_prompt_template: string;
  output_template: string;
  structured_output_schema: string;
  source_datasets: SourceDatasetPayload[];
  aliases: Array<{ source: string; column_map: Record<string, string> }>;
  curator: {
    upload_to_hf: boolean;
    upload_repo_id?: string;
    train_test_split: boolean;
    update_card: boolean;
    language: string[];
    license: string;
    task_categories?: string[];
    task_ids?: string[];
    citation_bibtex?: string;
  };
};

export type ConfigSummary = {
  name: string;
  dataset_name: string;
  description: string;
  generation_output_dir: string;
  sources: string[];
  models: string[];
  valid: boolean;
  errors: string[];
};

export type TemplateContext = {
  sources: Array<{
    name: string;
    columns: string[];
    aliases: Record<string, string>;
    available_input_fields: string[];
    missing_input_fields: string[];
    unused_aliases: string[];
    errors: string[];
  }>;
  output: {
    input_fields: string[];
    uses_llm_output: boolean;
    errors: string[];
  };
  globals: string[];
};

export type JobSnapshot = {
  id: string;
  kind: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  config_path: string;
  logs: string[];
  output_files: string[];
  output_dir?: string;
  run_state?: {
    total_processed?: number;
    total_valid?: number;
    total_invalid?: number;
    datasets?: Record<string, unknown>;
  } | null;
  summary?: {
    totals?: {
      rows?: number;
      valid?: number;
      invalid?: number;
      total_tokens?: number;
      latency_ms_mean?: number;
    };
    per_dataset?: Record<string, unknown>;
  } | null;
};
