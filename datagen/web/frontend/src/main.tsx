import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import * as Tabs from "@radix-ui/react-tabs";
import {
  Activity,
  ChevronDown,
  ChevronRight,
  Database,
  FileCheck,
  Loader2,
  Pencil,
  Play,
  Plus,
  RotateCcw,
  Save,
  SlidersHorizontal,
  Square,
  Trash2
} from "lucide-react";
import { api } from "./api";
import type { BuilderPayload, ConfigSummary, JobSnapshot, ModelPayload, SavedModel, SourcePayload, TemplateContext } from "./types";
import "./styles.css";

const emptyPayload: BuilderPayload = {
  dataset_name: "Dataset generated with templated generator",
  description: "Dataset generated using the templated generator.",
  authors: ["Your Name <you@example.com>"],
  generation_output_dir: "../raw_data/new_dataset",
  generation_logging_steps: 100,
  sources: [{ name: "source", path: "", split: "train", columns: [] }],
  models: [{ kind: "custom", name: "dummy", backend: "dummy", params: { response: "ok" } }],
  default_model: "dummy",
  default_system_prompt: "You are a helpful assistant.",
  default_prompt_template: "Generate a response for: {{ input.text }}",
  output_template: '{\n  "input": "{{ input.text }}",\n  "output": "{{ llm_output }}"\n}',
  structured_output_schema: "",
  source_datasets: [{ name: "source", max_records: 100, max_failures: 0.5, shuffle: false }],
  aliases: [{ source: "source", column_map: { text: "body" } }],
  curator: {
    upload_to_hf: false,
    train_test_split: true,
    update_card: true,
    language: ["en"],
    license: "mit",
    task_categories: [],
    task_ids: [],
    citation_bibtex: ""
  }
};

type Page = "configs" | "models";

type ModelFormState = {
  name: string;
  backend: "openai" | "dummy";
  openai_model: string;
  openai_provider_preset: "openai" | "openrouter" | "ollama" | "custom";
  openai_api_base: string;
  openai_key_source: "omitted" | "env" | "literal";
  openai_api_key_env: string;
  openai_api_key_literal: string;
  openai_temperature: string;
  openai_max_tokens: string;
  openai_top_p: string;
  openai_frequency_penalty: string;
  openai_presence_penalty: string;
  dummy_response: string;
};

const emptyModelForm: ModelFormState = {
  name: "",
  backend: "openai",
  openai_model: "gpt-4.1-mini",
  openai_provider_preset: "openai",
  openai_api_base: "",
  openai_key_source: "omitted",
  openai_api_key_env: "",
  openai_api_key_literal: "",
  openai_temperature: "0.3",
  openai_max_tokens: "1000",
  openai_top_p: "1",
  openai_frequency_penalty: "0",
  openai_presence_penalty: "0",
  dummy_response: "ok"
};

const numericParamFields = new Set(["temperature", "top_p", "frequency_penalty", "presence_penalty"]);

function paramsOf(model: ModelPayload): Record<string, unknown> {
  if (!model.params || typeof model.params === "string") return {};
  return model.params;
}

function effectiveSettingsParams(model: ModelPayload, base: SavedModel): Record<string, unknown> {
  return { ...base.params, ...(model.overrides?.params || {}) };
}

function formatValue(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function parseParamValue(field: string, value: string): unknown {
  if (field === "max_tokens") return value === "" ? "" : Number.parseInt(value, 10);
  if (numericParamFields.has(field)) return value === "" ? "" : Number.parseFloat(value);
  if (field === "api_key_env") return value ? { env: value } : undefined;
  return value;
}

function modelSummary(model: SavedModel): string {
  if (model.backend === "dummy") return `dummy · ${formatValue(model.params.response || "response")}`;
  const parts = [formatValue(model.params.model || "model")];
  if (model.params.api_base) parts.push(formatValue(model.params.api_base));
  if (typeof model.params.api_key === "object" && model.params.api_key && "env" in model.params.api_key) {
    parts.push(`env: ${String((model.params.api_key as { env?: unknown }).env)}`);
  } else if (model.params.api_key) {
    parts.push("literal key");
  } else {
    parts.push("key omitted");
  }
  return parts.join(" · ");
}

function formFromSavedModel(name: string, model: SavedModel): ModelFormState {
  const params = model.params || {};
  const apiKey = params.api_key;
  let keySource: ModelFormState["openai_key_source"] = "omitted";
  let keyEnv = "";
  let keyLiteral = "";
  if (typeof apiKey === "object" && apiKey && "env" in apiKey) {
    keySource = "env";
    keyEnv = String((apiKey as { env?: unknown }).env || "");
  } else if (typeof apiKey === "string") {
    keySource = "literal";
    keyLiteral = apiKey;
  }
  return {
    ...emptyModelForm,
    name,
    backend: model.backend === "dummy" ? "dummy" : "openai",
    openai_model: formatValue(params.model || emptyModelForm.openai_model),
    openai_provider_preset: inferProviderPreset(params),
    openai_api_base: formatValue(params.api_base),
    openai_key_source: keySource,
    openai_api_key_env: keyEnv,
    openai_api_key_literal: keyLiteral,
    openai_temperature: formatValue(params.temperature ?? emptyModelForm.openai_temperature),
    openai_max_tokens: formatValue(params.max_tokens ?? emptyModelForm.openai_max_tokens),
    openai_top_p: formatValue(params.top_p ?? emptyModelForm.openai_top_p),
    openai_frequency_penalty: formatValue(params.frequency_penalty ?? emptyModelForm.openai_frequency_penalty),
    openai_presence_penalty: formatValue(params.presence_penalty ?? emptyModelForm.openai_presence_penalty),
    dummy_response: formatValue(params.response || emptyModelForm.dummy_response)
  };
}

function inferProviderPreset(params: Record<string, unknown>): ModelFormState["openai_provider_preset"] {
  if (params.api_base === "https://openrouter.ai/api/v1") return "openrouter";
  if (params.api_base === "http://localhost:11434/v1") return "ollama";
  if (!params.api_base) return "openai";
  return "custom";
}

function modelFormPayload(form: ModelFormState): Record<string, unknown> {
  const payload: Record<string, unknown> = { ...form };
  if (form.openai_provider_preset !== "custom") {
    delete payload.openai_api_base;
  }
  return payload;
}

function clonePayload(payload: BuilderPayload): BuilderPayload {
  return JSON.parse(JSON.stringify(payload));
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function IconButton({ children, onClick, title }: { children: React.ReactNode; onClick: () => void; title: string }) {
  return (
    <button className="icon-button" type="button" onClick={onClick} title={title} aria-label={title}>
      {children}
    </button>
  );
}

function useTemplateContext(payload: BuilderPayload) {
  const [context, setContext] = useState<TemplateContext | null>(null);
  useEffect(() => {
    const timer = window.setTimeout(() => {
      api.templateContext(payload).then(setContext).catch(() => setContext(null));
    }, 250);
    return () => window.clearTimeout(timer);
  }, [payload]);
  return context;
}

function App() {
  const [configs, setConfigs] = useState<ConfigSummary[]>([]);
  const [settingsModels, setSettingsModels] = useState<Record<string, SavedModel>>({});
  const [settingsPath, setSettingsPath] = useState("");
  const [payload, setPayload] = useState<BuilderPayload>(() => clonePayload(emptyPayload));
  const [activeConfig, setActiveConfig] = useState<string>("");
  const [activePage, setActivePage] = useState<Page>("configs");
  const [message, setMessage] = useState("");
  const [toml, setToml] = useState("");
  const [job, setJob] = useState<JobSnapshot | null>(null);
  const [busy, setBusy] = useState(false);
  const context = useTemplateContext(payload);

  async function refresh() {
    const [configData, settingsData] = await Promise.all([api.configs(), api.settings()]);
    setConfigs(configData.configs);
    setSettingsModels(settingsData.models);
    setSettingsPath(settingsData.settings_path);
  }

  useEffect(() => {
    refresh().catch((error) => setMessage(error.message));
  }, []);

  useEffect(() => {
    if (!job || ["succeeded", "failed", "cancelled"].includes(job.status)) return;
    const events = new EventSource(`/api/jobs/${job.id}/events`);
    events.addEventListener("job", (event) => setJob(JSON.parse((event as MessageEvent).data)));
    events.addEventListener("done", (event) => {
      setJob(JSON.parse((event as MessageEvent).data));
      events.close();
    });
    events.onerror = () => events.close();
    return () => events.close();
  }, [job?.id]);

  const modelNames = useMemo(() => {
    const selected = payload.models.map((model) => model.name).filter(Boolean);
    return Array.from(new Set(selected));
  }, [payload.models]);

  function updatePayload(updater: (draft: BuilderPayload) => void) {
    setPayload((current) => {
      const next = clonePayload(current);
      updater(next);
      return next;
    });
  }

  async function loadConfig(name: string) {
    setBusy(true);
    try {
      const data = await api.config(name);
      setPayload(data.payload);
      setActiveConfig(name);
      setActivePage("configs");
      setMessage(`Loaded ${name}`);
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function previewToml() {
    try {
      const result = await api.previewConfig(payload);
      setToml(result.toml);
      setMessage(result.ok ? "Config is valid" : result.errors.join("\n"));
    } catch (error) {
      setMessage((error as Error).message);
    }
  }

  async function saveConfig() {
    const name = activeConfig || `${payload.dataset_name.toLowerCase().replace(/[^a-z0-9]+/g, "_") || "new_dataset"}.toml`;
    try {
      const result = activeConfig ? await api.updateConfig(activeConfig, payload) : await api.createConfig(name, payload, false);
      setActiveConfig(result.name);
      setToml(result.toml);
      setMessage(`Saved ${result.name}`);
      await refresh();
    } catch (error) {
      setMessage((error as Error).message);
    }
  }

  async function startTrial() {
    try {
      const next = await api.trial(payload, 1);
      setJob(next);
      setMessage("Trial run started");
    } catch (error) {
      setMessage((error as Error).message);
    }
  }

  async function startFullRun() {
    if (!activeConfig) {
      setMessage("Save this config before starting a full run.");
      return;
    }
    try {
      const next = await api.generate(activeConfig);
      setJob(next);
      setMessage("Full run started");
    } catch (error) {
      setMessage((error as Error).message);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <Database size={24} />
          <div>
            <strong>Dataset Studio</strong>
            <span>Local generator UI</span>
          </div>
        </div>
        <button className="nav-action" onClick={() => { setPayload(clonePayload(emptyPayload)); setActiveConfig(""); setToml(""); setActivePage("configs"); }}>
          <Plus size={18} /> New config
        </button>
        <div className="config-list">
          <span className="nav-label">Workspace</span>
          <button className={activePage === "configs" ? "config active" : "config"} onClick={() => setActivePage("configs")}>
            <span>Configs</span>
            <small>Builder and runs</small>
          </button>
          <button className={activePage === "models" ? "config active" : "config"} onClick={() => setActivePage("models")}>
            <span>Models</span>
            <small>Reusable model library</small>
          </button>
        </div>
        <div className="config-list">
          <span className="nav-label">Configs</span>
          {configs.map((config) => (
            <button key={config.name} className={activeConfig === config.name ? "config active" : "config"} onClick={() => loadConfig(config.name)}>
              <span>{config.dataset_name || config.name}</span>
              <small>{config.valid ? "valid" : "needs attention"}</small>
            </button>
          ))}
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <h1>{activePage === "models" ? "Models" : activeConfig || "New templated config"}</h1>
            <p>{activePage === "models" ? "Reusable model definitions for dataset configs." : payload.description}</p>
          </div>
          {activePage === "configs" ? <div className="actions">
            <button onClick={previewToml}><FileCheck size={18} /> Validate</button>
            <button onClick={saveConfig}><Save size={18} /> Save</button>
            <button onClick={startTrial}><Play size={18} /> Trial</button>
            <button className="primary" onClick={startFullRun}><Activity size={18} /> Full run</button>
          </div> : null}
        </header>

        {message && <div className="status-line">{message}</div>}
        {busy && <div className="status-line"><Loader2 className="spin" size={16} /> Loading...</div>}

        {activePage === "models" ? (
          <ModelLibraryPage models={settingsModels} settingsPath={settingsPath} refresh={refresh} setMessage={setMessage} />
        ) : <div className="content-grid">
          <section className="builder-panel">
            <Tabs.Root defaultValue="metadata" className="tabs">
              <Tabs.List className="tab-list">
                <Tabs.Trigger value="metadata">Metadata</Tabs.Trigger>
                <Tabs.Trigger value="sources">Sources & aliases</Tabs.Trigger>
                <Tabs.Trigger value="models">Models</Tabs.Trigger>
                <Tabs.Trigger value="templates">Templates</Tabs.Trigger>
                <Tabs.Trigger value="rules">Rules</Tabs.Trigger>
                <Tabs.Trigger value="curator">Curator</Tabs.Trigger>
              </Tabs.List>
              <Tabs.Content value="metadata">
                <MetadataPanel payload={payload} updatePayload={updatePayload} />
              </Tabs.Content>
              <Tabs.Content value="sources">
                <SourcesPanel payload={payload} updatePayload={updatePayload} />
              </Tabs.Content>
              <Tabs.Content value="models">
                <ModelsPanel payload={payload} settingsModels={settingsModels} updatePayload={updatePayload} />
              </Tabs.Content>
              <Tabs.Content value="templates">
                <TemplatesPanel payload={payload} updatePayload={updatePayload} context={context} />
              </Tabs.Content>
              <Tabs.Content value="rules">
                <RulesPanel payload={payload} modelNames={modelNames} updatePayload={updatePayload} />
              </Tabs.Content>
              <Tabs.Content value="curator">
                <CuratorPanel payload={payload} updatePayload={updatePayload} />
              </Tabs.Content>
            </Tabs.Root>
          </section>

          <aside className="inspector">
            <TemplateInspector context={context} />
            <RunPanel job={job} cancel={async () => {
              if (!job) return;
              setJob(await api.cancelJob(job.id));
            }} />
            {toml && <pre className="toml-preview">{toml}</pre>}
          </aside>
        </div>}
      </main>
    </div>
  );
}

function MetadataPanel({ payload, updatePayload }: { payload: BuilderPayload; updatePayload: (fn: (draft: BuilderPayload) => void) => void }) {
  return (
    <div className="form-grid">
      <Field label="Dataset name"><input value={payload.dataset_name} onChange={(e) => updatePayload((d) => { d.dataset_name = e.target.value; })} /></Field>
      <Field label="Output directory"><input value={payload.generation_output_dir} onChange={(e) => updatePayload((d) => { d.generation_output_dir = e.target.value; })} /></Field>
      <Field label="Authors"><textarea value={payload.authors.join("\n")} onChange={(e) => updatePayload((d) => { d.authors = e.target.value.split(/\n+/).filter(Boolean); })} /></Field>
      <Field label="Description"><textarea value={payload.description} onChange={(e) => updatePayload((d) => { d.description = e.target.value; })} /></Field>
      <Field label="Logging steps"><input type="number" min={1} value={payload.generation_logging_steps} onChange={(e) => updatePayload((d) => { d.generation_logging_steps = e.target.value; })} /></Field>
    </div>
  );
}

function SourcesPanel({ payload, updatePayload }: { payload: BuilderPayload; updatePayload: (fn: (draft: BuilderPayload) => void) => void }) {
  async function previewSource(source: SourcePayload, index: number) {
    const preview = await api.sourcePreview(source);
    updatePayload((draft) => {
      draft.sources[index].columns = preview.columns;
      draft.sources[index].preview_rows = preview.rows;
    });
  }
  return (
    <div className="stack">
      {payload.sources.map((source, index) => (
        <article className="item-panel" key={index}>
          <div className="item-head">
            <strong>{source.name || "Source"}</strong>
            <IconButton title="Remove source" onClick={() => updatePayload((d) => { d.sources.splice(index, 1); })}><Trash2 size={16} /></IconButton>
          </div>
          <div className="form-grid compact">
            <Field label="Name"><input value={source.name} onChange={(e) => updatePayload((d) => { d.sources[index].name = e.target.value; })} /></Field>
            <Field label="HF path"><input value={source.path} onChange={(e) => updatePayload((d) => { d.sources[index].path = e.target.value; })} /></Field>
            <Field label="Subset"><input value={source.subset || ""} onChange={(e) => updatePayload((d) => { d.sources[index].subset = e.target.value; })} /></Field>
            <Field label="Split"><input value={source.split} onChange={(e) => updatePayload((d) => { d.sources[index].split = e.target.value; })} /></Field>
          </div>
          <div className="inline-actions">
            <button onClick={() => previewSource(source, index)}>Preview source</button>
            <button onClick={() => updatePayload((d) => {
              const target = d.aliases.find((alias) => alias.source === source.name) || { source: source.name, column_map: {} };
              target.column_map[`field_${Object.keys(target.column_map).length + 1}`] = source.columns?.[0] || "";
              if (!d.aliases.includes(target)) d.aliases.push(target);
            })}>Add alias</button>
          </div>
          <AliasEditor payload={payload} source={source} updatePayload={updatePayload} />
          {source.preview_rows?.length ? <pre className="row-preview">{JSON.stringify(source.preview_rows[0], null, 2)}</pre> : null}
        </article>
      ))}
      <button onClick={() => updatePayload((d) => { d.sources.push({ name: `source_${d.sources.length + 1}`, path: "", split: "train", columns: [] }); })}><Plus size={16} /> Add source</button>
    </div>
  );
}

function AliasEditor({ payload, source, updatePayload }: { payload: BuilderPayload; source: SourcePayload; updatePayload: (fn: (draft: BuilderPayload) => void) => void }) {
  const alias = payload.aliases.find((item) => item.source === source.name);
  const entries = Object.entries(alias?.column_map || {});
  if (!entries.length) return null;
  return (
    <div className="alias-grid">
      {entries.map(([name, column]) => (
        <React.Fragment key={name}>
          <input value={name} onChange={(e) => updatePayload((d) => {
            const item = d.aliases.find((row) => row.source === source.name);
            if (!item) return;
            const value = item.column_map[name];
            delete item.column_map[name];
            item.column_map[e.target.value] = value;
          })} />
          <input list={`${source.name}-columns`} value={column} onChange={(e) => updatePayload((d) => {
            const item = d.aliases.find((row) => row.source === source.name);
            if (item) item.column_map[name] = e.target.value;
          })} />
          <code>{`input.${name}`}</code>
          <IconButton title="Remove alias" onClick={() => updatePayload((d) => {
            const item = d.aliases.find((row) => row.source === source.name);
            if (item) delete item.column_map[name];
          })}><Trash2 size={16} /></IconButton>
        </React.Fragment>
      ))}
      <datalist id={`${source.name}-columns`}>
        {(source.columns || []).map((column) => <option key={column} value={column} />)}
      </datalist>
    </div>
  );
}

function ModelsPanel({ payload, settingsModels, updatePayload }: { payload: BuilderPayload; settingsModels: Record<string, SavedModel>; updatePayload: (fn: (draft: BuilderPayload) => void) => void }) {
  const [expandedSaved, setExpandedSaved] = useState<Record<string, boolean>>({});
  const [customOpen, setCustomOpen] = useState(false);
  const customModels = payload.models.map((model, index) => ({ model, index })).filter(({ model }) => model.kind !== "settings");

  function selectSaved(name: string, checked: boolean) {
    updatePayload((d) => {
      d.models = d.models.filter((item) => !(item.kind === "settings" && item.name === name));
      if (checked) {
        d.models.push({ kind: "settings", name });
        if (!d.default_model || !d.models.some((model) => model.name === d.default_model)) d.default_model = name;
      }
    });
    if (checked) setExpandedSaved((current) => ({ ...current, [name]: true }));
  }

  return (
    <div className="stack">
      <div className="model-picker">
        {Object.entries(settingsModels).map(([name, model]) => {
          const selected = payload.models.find((item) => item.kind === "settings" && item.name === name);
          const checked = Boolean(selected);
          const isExpanded = Boolean(expandedSaved[name]);
          return (
            <article className={checked ? "model-row selected" : "model-row"} key={name}>
              <label className="check-row">
                <input type="checkbox" checked={checked} onChange={(e) => selectSaved(name, e.target.checked)} />
                <strong>{name}</strong>
              </label>
              <small>{model.backend} · {modelSummary(model)}</small>
              {checked && selected ? (
                <>
                  <button className="subtle-button" type="button" onClick={() => setExpandedSaved((current) => ({ ...current, [name]: !isExpanded }))}>
                    {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                    {isExpanded ? "Hide effective fields" : "Edit overrides"}
                  </button>
                  {isExpanded ? (
                    <ModelParamEditor
                      backend={model.backend}
                      params={effectiveSettingsParams(selected, model)}
                      baseParams={model.params}
                      onChange={(field, value) => updatePayload((d) => {
                        const item = d.models.find((row) => row.kind === "settings" && row.name === name);
                        if (!item) return;
                        item.overrides = item.overrides || {};
                        item.overrides.params = item.overrides.params || {};
                        item.overrides.params[field] = value;
                      })}
                      onReset={(field) => updatePayload((d) => {
                        const item = d.models.find((row) => row.kind === "settings" && row.name === name);
                        if (!item?.overrides?.params) return;
                        delete item.overrides.params[field];
                        if (!Object.keys(item.overrides.params).length) delete item.overrides.params;
                        if (item.overrides && !Object.keys(item.overrides).length) delete item.overrides;
                      })}
                    />
                  ) : null}
                </>
              ) : null}
            </article>
          );
        })}
      </div>
      {!Object.keys(settingsModels).length ? <p>No saved models yet. Add reusable models from the Models page or create a custom model below.</p> : null}
      <details className="advanced-section" open={customOpen} onToggle={(event) => setCustomOpen(event.currentTarget.open)}>
        <summary><SlidersHorizontal size={16} /> Advanced custom models</summary>
        <div className="stack">
          {customModels.map(({ model, index }) => (
        <article className="item-panel" key={`${model.name}-${index}`}>
          <div className="item-head">
            <strong>{model.name || "Custom model"}</strong>
            <IconButton title="Remove custom model" onClick={() => updatePayload((d) => { d.models.splice(index, 1); })}><Trash2 size={16} /></IconButton>
          </div>
          <div className="form-grid compact">
            <Field label="Model name"><input value={model.name} onChange={(e) => updatePayload((d) => { d.models[index].name = e.target.value; })} /></Field>
            <Field label="Backend">
              <select value={model.backend || "openai"} onChange={(e) => updatePayload((d) => {
                d.models[index].backend = e.target.value;
                d.models[index].params = e.target.value === "dummy" ? { response: "ok" } : { model: "gpt-4.1-mini", temperature: 0.3, max_tokens: 1000 };
              })}>
                <option value="openai">openai</option>
                <option value="dummy">dummy</option>
              </select>
            </Field>
          </div>
          <ModelParamEditor
            backend={model.backend || "openai"}
            params={paramsOf(model)}
            onChange={(field, value) => updatePayload((d) => {
              const params = paramsOf(d.models[index]);
              if (value === undefined) delete params[field];
              else params[field] = value;
              d.models[index].params = params;
            })}
          />
        </article>
          ))}
          <button onClick={() => updatePayload((d) => { d.models.push({ kind: "custom", name: `model_${d.models.length + 1}`, backend: "openai", params: { model: "gpt-4.1-mini", temperature: 0.3, max_tokens: 1000 } }); })}><Plus size={16} /> Add custom model</button>
        </div>
      </details>
      <Field label="Default model">
        <select value={payload.default_model} onChange={(e) => updatePayload((d) => { d.default_model = e.target.value; })}>
          {payload.models.map((model) => <option key={model.name} value={model.name}>{model.name}</option>)}
        </select>
      </Field>
    </div>
  );
}

function ModelParamEditor({
  backend,
  params,
  baseParams,
  onChange,
  onReset
}: {
  backend: string;
  params: Record<string, unknown>;
  baseParams?: Record<string, unknown>;
  onChange: (field: string, value: unknown) => void;
  onReset?: (field: string) => void;
}) {
  if (backend === "dummy") {
    return (
      <ParamField field="response" label="Dummy response" value={params.response} baseValue={baseParams?.response} onChange={onChange} onReset={onReset} />
    );
  }

  const apiKey = params.api_key;
  const keyEnv = typeof apiKey === "object" && apiKey && "env" in apiKey ? String((apiKey as { env?: unknown }).env || "") : "";
  const keyLiteral = typeof apiKey === "string" ? apiKey : "";
  const keyMode = keyEnv ? "env" : keyLiteral ? "literal" : "omitted";
  return (
    <div className="stack tight">
      <div className="form-grid compact">
        <ParamField field="model" label="Model" value={params.model} baseValue={baseParams?.model} onChange={onChange} onReset={onReset} />
        <ParamField field="api_base" label="API base" value={params.api_base} baseValue={baseParams?.api_base} onChange={onChange} onReset={onReset} />
        <Field label="API key source">
          <select value={keyMode} onChange={(e) => {
            if (e.target.value === "omitted") onChange("api_key", undefined);
            if (e.target.value === "env") onChange("api_key", { env: keyEnv || "OPENAI_API_KEY" });
            if (e.target.value === "literal") onChange("api_key", keyLiteral || "");
          }}>
            <option value="omitted">omitted</option>
            <option value="env">env</option>
            <option value="literal">literal</option>
          </select>
        </Field>
        {keyMode === "env" ? (
          <ParamField field="api_key_env" label="API key env" value={keyEnv} baseValue={typeof baseParams?.api_key === "object" && baseParams.api_key && "env" in baseParams.api_key ? String((baseParams.api_key as { env?: unknown }).env || "") : undefined} onChange={(field, value) => onChange("api_key", parseParamValue(field, String(value || "")))} onReset={onReset ? () => onReset("api_key") : undefined} />
        ) : null}
        {keyMode === "literal" ? (
          <ParamField field="api_key" label="Literal API key" value={keyLiteral} baseValue={typeof baseParams?.api_key === "string" ? baseParams.api_key : undefined} onChange={onChange} onReset={onReset} />
        ) : null}
      </div>
      <div className="form-grid compact">
        <ParamField field="temperature" label="Temperature" type="number" step="0.1" value={params.temperature} baseValue={baseParams?.temperature} onChange={onChange} onReset={onReset} />
        <ParamField field="max_tokens" label="Max tokens" type="number" value={params.max_tokens} baseValue={baseParams?.max_tokens} onChange={onChange} onReset={onReset} />
        <ParamField field="top_p" label="Top P" type="number" step="0.05" value={params.top_p} baseValue={baseParams?.top_p} onChange={onChange} onReset={onReset} />
        <ParamField field="frequency_penalty" label="Frequency penalty" type="number" step="0.1" value={params.frequency_penalty} baseValue={baseParams?.frequency_penalty} onChange={onChange} onReset={onReset} />
        <ParamField field="presence_penalty" label="Presence penalty" type="number" step="0.1" value={params.presence_penalty} baseValue={baseParams?.presence_penalty} onChange={onChange} onReset={onReset} />
      </div>
    </div>
  );
}

function ParamField({
  field,
  label,
  value,
  baseValue,
  type = "text",
  step,
  onChange,
  onReset
}: {
  field: string;
  label: string;
  value: unknown;
  baseValue?: unknown;
  type?: string;
  step?: string;
  onChange: (field: string, value: unknown) => void;
  onReset?: (field: string) => void;
}) {
  const isOverride = baseValue !== undefined && JSON.stringify(value) !== JSON.stringify(baseValue);
  return (
    <label className={isOverride ? "field override-field" : "field"}>
      <span>
        {label}
        {isOverride ? <em>override</em> : null}
      </span>
      <div className="field-with-reset">
        <input type={type} step={step} value={formatValue(value)} onChange={(e) => onChange(field, parseParamValue(field, e.target.value))} />
        {isOverride && onReset ? <IconButton title={`Reset ${label}`} onClick={() => onReset(field)}><RotateCcw size={15} /></IconButton> : null}
      </div>
    </label>
  );
}

function TemplatesPanel({ payload, updatePayload, context }: { payload: BuilderPayload; updatePayload: (fn: (draft: BuilderPayload) => void) => void; context: TemplateContext | null }) {
  const firstPreview = payload.sources.find((source) => source.preview_rows?.length);
  const [rendered, setRendered] = useState("");
  async function renderPrompt() {
    if (!firstPreview?.preview_rows?.[0]) return;
    const result = await api.renderPrompt(payload, firstPreview.name, firstPreview.preview_rows[0]);
    setRendered(result.prompt);
  }
  return (
    <div className="stack">
      <Field label="Default system prompt"><textarea value={payload.default_system_prompt} onChange={(e) => updatePayload((d) => { d.default_system_prompt = e.target.value; })} /></Field>
      <Field label="Default prompt template"><textarea className="code tall" value={payload.default_prompt_template} onChange={(e) => updatePayload((d) => { d.default_prompt_template = e.target.value; })} /></Field>
      <Field label="Output template"><textarea className="code tall" value={payload.output_template} onChange={(e) => updatePayload((d) => { d.output_template = e.target.value; })} /></Field>
      <Field label="Structured output schema"><textarea className="code tall" value={payload.structured_output_schema} onChange={(e) => updatePayload((d) => { d.structured_output_schema = e.target.value; })} /></Field>
      <div className="inline-actions">
        <button disabled={!firstPreview} onClick={renderPrompt}>Render prompt with preview row</button>
      </div>
      {rendered && <pre className="row-preview">{rendered}</pre>}
      {context?.output.errors.map((error) => <div className="error-line" key={error}>{error}</div>)}
    </div>
  );
}

function RulesPanel({ payload, modelNames, updatePayload }: { payload: BuilderPayload; modelNames: string[]; updatePayload: (fn: (draft: BuilderPayload) => void) => void }) {
  return (
    <div className="stack">
      {payload.source_datasets.map((rule, index) => (
        <article className="item-panel" key={index}>
          <div className="form-grid compact">
            <Field label="Source">
              <select value={rule.name} onChange={(e) => updatePayload((d) => { d.source_datasets[index].name = e.target.value; })}>
                {payload.sources.map((source) => <option key={source.name}>{source.name}</option>)}
              </select>
            </Field>
            <Field label="Model override">
              <select value={rule.model || ""} onChange={(e) => updatePayload((d) => { d.source_datasets[index].model = e.target.value; })}>
                <option value="">Use default</option>
                {modelNames.map((name) => <option key={name}>{name}</option>)}
              </select>
            </Field>
            <Field label="Max records"><input type="number" min={1} value={rule.max_records} onChange={(e) => updatePayload((d) => { d.source_datasets[index].max_records = e.target.value; })} /></Field>
            <Field label="Max failures"><input type="number" min={0} step="0.05" value={rule.max_failures} onChange={(e) => updatePayload((d) => { d.source_datasets[index].max_failures = e.target.value; })} /></Field>
          </div>
          <label className="check-row"><input type="checkbox" checked={rule.shuffle} onChange={(e) => updatePayload((d) => { d.source_datasets[index].shuffle = e.target.checked; })} /> Shuffle before selecting records</label>
          <Field label="Prompt override"><textarea className="code" value={rule.prompt_template || ""} onChange={(e) => updatePayload((d) => { d.source_datasets[index].prompt_template = e.target.value; })} /></Field>
        </article>
      ))}
      <button onClick={() => updatePayload((d) => { d.source_datasets.push({ name: d.sources[0]?.name || "", max_records: 100, max_failures: 0.5, shuffle: false }); })}><Plus size={16} /> Add rule</button>
    </div>
  );
}

function CuratorPanel({ payload, updatePayload }: { payload: BuilderPayload; updatePayload: (fn: (draft: BuilderPayload) => void) => void }) {
  return (
    <div className="form-grid">
      <Field label="Upload repo id"><input value={payload.curator.upload_repo_id || ""} onChange={(e) => updatePayload((d) => { d.curator.upload_repo_id = e.target.value; })} /></Field>
      <Field label="License"><input value={payload.curator.license} onChange={(e) => updatePayload((d) => { d.curator.license = e.target.value; })} /></Field>
      <Field label="Languages"><textarea value={payload.curator.language.join("\n")} onChange={(e) => updatePayload((d) => { d.curator.language = e.target.value.split(/\n+/).filter(Boolean); })} /></Field>
      <label className="check-row"><input type="checkbox" checked={payload.curator.upload_to_hf} onChange={(e) => updatePayload((d) => { d.curator.upload_to_hf = e.target.checked; })} /> Upload after generation</label>
      <label className="check-row"><input type="checkbox" checked={payload.curator.train_test_split} onChange={(e) => updatePayload((d) => { d.curator.train_test_split = e.target.checked; })} /> Train/test split</label>
      <label className="check-row"><input type="checkbox" checked={payload.curator.update_card} onChange={(e) => updatePayload((d) => { d.curator.update_card = e.target.checked; })} /> Update dataset card</label>
    </div>
  );
}

function ModelLibraryPage({
  models,
  settingsPath,
  refresh,
  setMessage
}: {
  models: Record<string, SavedModel>;
  settingsPath: string;
  refresh: () => Promise<void>;
  setMessage: (message: string) => void;
}) {
  const [form, setForm] = useState<ModelFormState>(emptyModelForm);
  const [deleteName, setDeleteName] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const modelEntries = Object.entries(models);
  const editingExisting = Boolean(form.name && models[form.name]);

  async function save() {
    try {
      setError("");
      setSaving(true);
      await api.saveModel(modelFormPayload(form));
      await refresh();
      setMessage(`Saved model ${form.name}`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function deleteModel() {
    if (!deleteName) return;
    try {
      setDeleting(true);
      await api.deleteModel(deleteName);
      setMessage(`Deleted model ${deleteName}`);
      if (form.name === deleteName) setForm(emptyModelForm);
      setDeleteName("");
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setDeleting(false);
    }
  }

  function updateForm(next: Partial<ModelFormState>) {
    setForm((current) => ({ ...current, ...next }));
  }

  function resetForm() {
    setError("");
    setForm(emptyModelForm);
  }

  return (
    <div className="model-page">
      <section className="builder-panel model-library">
        <div className="section-head">
          <div>
            <h2>Saved Models</h2>
            <p className="path-text">{settingsPath || "Settings file will be created when you save a model."}</p>
          </div>
          <button className="secondary-action" onClick={resetForm}><Plus size={16} /> New</button>
        </div>
        <div className="model-list">
          {modelEntries.map(([name, model]) => {
            const selected = form.name === name;
            return (
              <article className={selected ? "model-row selected" : "model-row"} key={name}>
                <button className="model-select" type="button" onClick={() => setForm(formFromSavedModel(name, model))} aria-pressed={selected}>
                  <span>
                    <strong>{name}</strong>
                    <small>{model.backend} · {modelSummary(model)}</small>
                  </span>
                </button>
                <div className="model-row-actions">
                  <button className="text-button" type="button" onClick={() => setForm(formFromSavedModel(name, model))}>
                    <Pencil size={15} /> Edit
                  </button>
                  <IconButton title={`Delete ${name}`} onClick={() => setDeleteName(name)}><Trash2 size={16} /></IconButton>
                </div>
              </article>
            );
          })}
          {!modelEntries.length ? (
            <div className="empty-state">
              <strong>No reusable models yet</strong>
              <p>Create a saved model once, then include it in any dataset config.</p>
              <button className="primary" type="button" onClick={resetForm}><Plus size={16} /> Create model</button>
            </div>
          ) : null}
        </div>
      </section>

      <section className="builder-panel model-form-panel">
        <div className="section-head">
          <div>
            <h2>{editingExisting ? `Edit ${form.name}` : "Create Model"}</h2>
            <p>Define the reusable base model. Dataset configs can select it and override individual parameters later.</p>
          </div>
        </div>
        <div className="model-form">
          <section className="form-section" aria-labelledby="model-identity-title">
            <div className="form-section-heading">
              <h3 id="model-identity-title">Model identity</h3>
              <p>Name the reusable model and choose its runtime backend.</p>
            </div>
            <div className="form-grid model-basic-grid">
              <Field label="Name"><input placeholder="local-gemma" value={form.name} onChange={(e) => updateForm({ name: e.target.value })} /></Field>
              <Field label="Backend">
                <select value={form.backend} onChange={(e) => updateForm({ backend: e.target.value as ModelFormState["backend"] })}>
                  <option value="openai">openai</option>
                  <option value="dummy">dummy</option>
                </select>
              </Field>
            </div>
          </section>

          {form.backend === "openai" ? (
            <>
              <section className="form-section" aria-labelledby="provider-title">
                <div className="form-section-heading">
                  <h3 id="provider-title">Provider</h3>
                  <p>Pick a preset for common OpenAI-compatible endpoints, or use a custom base URL.</p>
                </div>
                <div className="form-grid responsive-field-grid">
                  <Field label="Provider preset">
                    <select value={form.openai_provider_preset} onChange={(e) => {
                      const preset = e.target.value as ModelFormState["openai_provider_preset"];
                      if (preset === "openrouter") updateForm({ openai_provider_preset: preset, openai_api_base: "", openai_key_source: "env", openai_api_key_env: "OPENROUTER_API_KEY" });
                      else if (preset === "ollama") updateForm({ openai_provider_preset: preset, openai_api_base: "", openai_key_source: "literal", openai_api_key_literal: "abc" });
                      else updateForm({ openai_provider_preset: preset, openai_api_base: "", openai_key_source: "omitted" });
                    }}>
                      <option value="openai">OpenAI</option>
                      <option value="openrouter">OpenRouter</option>
                      <option value="ollama">Ollama/local</option>
                      <option value="custom">Custom</option>
                    </select>
                  </Field>
                  <Field label="Model"><input placeholder="gpt-4.1-mini" value={form.openai_model} onChange={(e) => updateForm({ openai_model: e.target.value })} /></Field>
                  {form.openai_provider_preset === "custom" ? (
                    <Field label="API base"><input placeholder="https://example.com/v1" value={form.openai_api_base} onChange={(e) => updateForm({ openai_api_base: e.target.value })} /></Field>
                  ) : null}
                </div>
              </section>

              <section className="form-section" aria-labelledby="credentials-title">
                <div className="form-section-heading">
                  <h3 id="credentials-title">Credentials</h3>
                  <p>Store no key, reference an environment variable, or save a literal local value.</p>
                </div>
                <div className="form-grid responsive-field-grid">
                  <Field label="API key source">
                    <select value={form.openai_key_source} onChange={(e) => updateForm({ openai_key_source: e.target.value as ModelFormState["openai_key_source"] })}>
                      <option value="omitted">omitted</option>
                      <option value="env">env</option>
                      <option value="literal">literal</option>
                    </select>
                  </Field>
                  {form.openai_key_source === "env" ? (
                    <Field label="API key env"><input placeholder="OPENAI_API_KEY" value={form.openai_api_key_env} onChange={(e) => updateForm({ openai_api_key_env: e.target.value })} /></Field>
                  ) : null}
                  {form.openai_key_source === "literal" ? (
                    <Field label="Literal API key"><input placeholder="sk-..." value={form.openai_api_key_literal} onChange={(e) => updateForm({ openai_api_key_literal: e.target.value })} /></Field>
                  ) : null}
                </div>
              </section>

              <section className="form-section" aria-labelledby="sampling-title">
                <div className="form-section-heading">
                  <h3 id="sampling-title">Sampling</h3>
                  <p>Defaults used by configs unless a config overrides a field.</p>
                </div>
                <div className="form-grid sampling-grid">
                  <Field label="Temperature"><input type="number" step="0.1" value={form.openai_temperature} onChange={(e) => updateForm({ openai_temperature: e.target.value })} /></Field>
                  <Field label="Max tokens"><input type="number" min={1} value={form.openai_max_tokens} onChange={(e) => updateForm({ openai_max_tokens: e.target.value })} /></Field>
                  <Field label="Top P"><input type="number" step="0.05" value={form.openai_top_p} onChange={(e) => updateForm({ openai_top_p: e.target.value })} /></Field>
                  <Field label="Frequency penalty"><input type="number" step="0.1" value={form.openai_frequency_penalty} onChange={(e) => updateForm({ openai_frequency_penalty: e.target.value })} /></Field>
                  <Field label="Presence penalty"><input type="number" step="0.1" value={form.openai_presence_penalty} onChange={(e) => updateForm({ openai_presence_penalty: e.target.value })} /></Field>
                </div>
              </section>
            </>
          ) : (
            <section className="form-section" aria-labelledby="dummy-title">
              <div className="form-section-heading">
                <h3 id="dummy-title">Dummy response</h3>
                <p>Return this value for local smoke tests and UI validation.</p>
              </div>
              <Field label="Response"><input placeholder="ok" value={form.dummy_response} onChange={(e) => updateForm({ dummy_response: e.target.value })} /></Field>
            </section>
          )}

          {error && <div className="form-alert" role="alert">{error}</div>}

          <div className="form-footer">
            <button className="secondary-action" type="button" onClick={resetForm} disabled={saving}>Reset form</button>
            <button className="primary" onClick={save} disabled={saving}>
              {saving ? <Loader2 className="spin" size={16} /> : <Save size={16} />}
              {saving ? "Saving..." : "Save reusable model"}
            </button>
          </div>
        </div>
      </section>

      {deleteName ? (
        <div className="modal-backdrop">
          <div className="modal">
            <h2>Delete {deleteName}?</h2>
            <p>Existing config TOML files keep their concrete model settings, but this reusable model will disappear from the library.</p>
            <div className="inline-actions">
              <button onClick={() => setDeleteName("")} disabled={deleting}>Cancel</button>
              <button className="danger" onClick={deleteModel} disabled={deleting}>
                {deleting ? <Loader2 className="spin" size={16} /> : <Trash2 size={16} />}
                {deleting ? "Deleting..." : "Delete"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
      </div>
  );
}

function TemplateInspector({ context }: { context: TemplateContext | null }) {
  return (
    <section className="side-card">
      <h2>Template fields</h2>
      {!context ? <p>Preview a source or edit aliases to populate context.</p> : (
        <div className="stack tight">
          {context.sources.map((source) => (
            <div key={source.name}>
              <strong>{source.name}</strong>
              <div className="chips">
                {source.available_input_fields.map((field) => <code key={field}>{`input.${field}`}</code>)}
              </div>
              {source.missing_input_fields.map((field) => <div className="error-line" key={field}>{`Missing input.${field}`}</div>)}
              {source.unused_aliases.map((field) => <div className="hint-line" key={field}>{`Unused alias input.${field}`}</div>)}
              {source.errors.map((error) => <div className="error-line" key={error}>{error}</div>)}
            </div>
          ))}
          <div>
            <strong>Output globals</strong>
            <div className="chips"><code>llm_output</code>{context.output.input_fields.map((field) => <code key={field}>{`input.${field}`}</code>)}</div>
          </div>
        </div>
      )}
    </section>
  );
}

function RunPanel({ job, cancel }: { job: JobSnapshot | null; cancel: () => Promise<void> }) {
  const totals = job?.summary?.totals || {};
  return (
    <section className="side-card">
      <div className="item-head">
        <h2>Run</h2>
        {job?.status === "running" && <button className="danger" onClick={cancel}><Square size={16} /> Cancel</button>}
      </div>
      {!job ? <p>No active run.</p> : (
        <div className="stack tight">
          <div className={`run-state ${job.status}`}>{job.kind} · {job.status}</div>
          <div className="metric-grid">
            <span>Rows <strong>{job.run_state?.total_processed ?? totals.rows ?? 0}</strong></span>
            <span>Valid <strong>{job.run_state?.total_valid ?? totals.valid ?? 0}</strong></span>
            <span>Invalid <strong>{job.run_state?.total_invalid ?? totals.invalid ?? 0}</strong></span>
            <span>Tokens <strong>{totals.total_tokens ?? 0}</strong></span>
          </div>
          <pre className="log-view">{(job.logs || []).slice(-80).join("\n")}</pre>
          {job.output_files?.length ? <div className="chips">{job.output_files.map((file) => <code key={file}>{file}</code>)}</div> : null}
        </div>
      )}
    </section>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
