import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import * as Tabs from "@radix-ui/react-tabs";
import {
  Activity,
  Database,
  FileCheck,
  Loader2,
  Play,
  Plus,
  Save,
  Settings,
  Square,
  Trash2
} from "lucide-react";
import { api } from "./api";
import type { BuilderPayload, ConfigSummary, JobSnapshot, SourcePayload, TemplateContext } from "./types";
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
  const [settingsModels, setSettingsModels] = useState<Record<string, { backend: string; params: Record<string, unknown> }>>({});
  const [payload, setPayload] = useState<BuilderPayload>(() => clonePayload(emptyPayload));
  const [activeConfig, setActiveConfig] = useState<string>("");
  const [message, setMessage] = useState("");
  const [toml, setToml] = useState("");
  const [job, setJob] = useState<JobSnapshot | null>(null);
  const [busy, setBusy] = useState(false);
  const context = useTemplateContext(payload);

  async function refresh() {
    const [configData, settingsData] = await Promise.all([api.configs(), api.settings()]);
    setConfigs(configData.configs);
    setSettingsModels(settingsData.models);
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
        <button className="nav-action" onClick={() => { setPayload(clonePayload(emptyPayload)); setActiveConfig(""); setToml(""); }}>
          <Plus size={18} /> New config
        </button>
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
            <h1>{activeConfig || "New templated config"}</h1>
            <p>{payload.description}</p>
          </div>
          <div className="actions">
            <button onClick={previewToml}><FileCheck size={18} /> Validate</button>
            <button onClick={saveConfig}><Save size={18} /> Save</button>
            <button onClick={startTrial}><Play size={18} /> Trial</button>
            <button className="primary" onClick={startFullRun}><Activity size={18} /> Full run</button>
          </div>
        </header>

        {message && <div className="status-line">{message}</div>}
        {busy && <div className="status-line"><Loader2 className="spin" size={16} /> Loading...</div>}

        <div className="content-grid">
          <section className="builder-panel">
            <Tabs.Root defaultValue="metadata" className="tabs">
              <Tabs.List className="tab-list">
                <Tabs.Trigger value="metadata">Metadata</Tabs.Trigger>
                <Tabs.Trigger value="sources">Sources & aliases</Tabs.Trigger>
                <Tabs.Trigger value="models">Models</Tabs.Trigger>
                <Tabs.Trigger value="templates">Templates</Tabs.Trigger>
                <Tabs.Trigger value="rules">Rules</Tabs.Trigger>
                <Tabs.Trigger value="curator">Curator</Tabs.Trigger>
                <Tabs.Trigger value="settings"><Settings size={16} /> Settings</Tabs.Trigger>
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
              <Tabs.Content value="settings">
                <SettingsPanel models={settingsModels} refresh={refresh} />
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
        </div>
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

function ModelsPanel({ payload, settingsModels, updatePayload }: { payload: BuilderPayload; settingsModels: Record<string, { backend: string }>; updatePayload: (fn: (draft: BuilderPayload) => void) => void }) {
  return (
    <div className="stack">
      <div className="settings-model-grid">
        {Object.entries(settingsModels).map(([name, model]) => {
          const checked = payload.models.some((item) => item.kind === "settings" && item.name === name);
          return (
            <label className="check-card" key={name}>
              <input type="checkbox" checked={checked} onChange={(e) => updatePayload((d) => {
                d.models = d.models.filter((item) => !(item.kind === "settings" && item.name === name));
                if (e.target.checked) d.models.push({ kind: "settings", name });
              })} />
              <span>{name}</span>
              <small>{model.backend}</small>
            </label>
          );
        })}
      </div>
      {payload.models.filter((model) => model.kind !== "settings").map((model, index) => (
        <article className="item-panel" key={index}>
          <div className="form-grid compact">
            <Field label="Model name"><input value={model.name} onChange={(e) => updatePayload((d) => { d.models[index].name = e.target.value; })} /></Field>
            <Field label="Backend">
              <select value={model.backend || "openai"} onChange={(e) => updatePayload((d) => { d.models[index].backend = e.target.value; })}>
                <option value="openai">openai</option>
                <option value="dummy">dummy</option>
              </select>
            </Field>
          </div>
          <Field label="Params JSON"><textarea className="code" value={typeof model.params === "string" ? model.params : JSON.stringify(model.params || {}, null, 2)} onChange={(e) => updatePayload((d) => { d.models[index].params = e.target.value; })} /></Field>
        </article>
      ))}
      <Field label="Default model">
        <select value={payload.default_model} onChange={(e) => updatePayload((d) => { d.default_model = e.target.value; })}>
          {payload.models.map((model) => <option key={model.name} value={model.name}>{model.name}</option>)}
        </select>
      </Field>
      <button onClick={() => updatePayload((d) => { d.models.push({ kind: "custom", name: `model_${d.models.length + 1}`, backend: "openai", params: { model: "gpt-4.1-mini", temperature: 0.3, max_tokens: 1000 } }); })}><Plus size={16} /> Add custom model</button>
    </div>
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

function SettingsPanel({ models, refresh }: { models: Record<string, { backend: string; params: Record<string, unknown> }>; refresh: () => Promise<void> }) {
  const [form, setForm] = useState({ name: "", backend: "openai", openai_model: "gpt-4.1-mini", openai_provider_preset: "openai", dummy_response: "ok" });
  const [error, setError] = useState("");
  async function save() {
    try {
      setError("");
      await api.saveModel(form);
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    }
  }
  return (
    <div className="stack">
      <div className="settings-model-grid">
        {Object.entries(models).map(([name, model]) => (
          <div className="check-card" key={name}>
            <span>{name}</span>
            <small>{model.backend}</small>
            <IconButton title="Delete model" onClick={async () => { await api.deleteModel(name); await refresh(); }}><Trash2 size={16} /></IconButton>
          </div>
        ))}
      </div>
      <div className="form-grid compact">
        <Field label="Name"><input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
        <Field label="Backend"><select value={form.backend} onChange={(e) => setForm({ ...form, backend: e.target.value })}><option value="openai">openai</option><option value="dummy">dummy</option></select></Field>
        {form.backend === "openai" ? (
          <>
            <Field label="Model"><input value={form.openai_model} onChange={(e) => setForm({ ...form, openai_model: e.target.value })} /></Field>
            <Field label="Provider"><select value={form.openai_provider_preset} onChange={(e) => setForm({ ...form, openai_provider_preset: e.target.value })}><option value="openai">OpenAI</option><option value="openrouter">OpenRouter</option><option value="ollama">Ollama/local</option><option value="custom">Custom</option></select></Field>
          </>
        ) : (
          <Field label="Dummy response"><input value={form.dummy_response} onChange={(e) => setForm({ ...form, dummy_response: e.target.value })} /></Field>
        )}
      </div>
      {error && <div className="error-line">{error}</div>}
      <button onClick={save}><Save size={16} /> Save reusable model</button>
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
