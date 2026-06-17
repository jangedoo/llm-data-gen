import json
import time
from pathlib import Path

import pytest
import datasets

from datagen.core.gen_config import GenerationPipelineConfig, OpenAIModelConfig
from datagen.llm.dummy import DummyLLM
from datagen.llm.openai import OpenAILLM
from datagen.web import builder
from datagen.web import config_utils
from datagen.web.app import create_app
from datagen.web.jobs import JobManager
from datagen.web.settings import SettingsStore


def write_config(
    tmp_path: Path, name: str = "sample.toml", generator: str = "templated"
):
    output_dir = tmp_path / "generated"
    config_path = tmp_path / name
    config_path.write_text(
        f"""
dataset_name = "Sample"
description = "Sample config"
authors = ["Tester <tester@example.com>"]
generation_output_dir = "{output_dir.name}"
generation_logging_steps = 1

[sources.source]
path = "dummy/source"
split = "train"

[models.dummy]
backend = "dummy"
[models.dummy.params]
response = "{{\\"answer\\": \\"ok\\"}}"

[generator]
generator = "{generator}"

[generator.params]
default_model = "dummy"
default_system_prompt = "System"
default_prompt_template = "Question: {{{{ input.text }}}}"
source_datasets = [{{ name = "source", max_records = 1 }}]
output_template = '{{"text": "{{{{ input.text }}}}", "answer": "{{{{ llm_output }}}}"}}'

[[generator.params.aliases]]
source = "source"
column_map = {{ text = "body" }}

[curator.params]
upload_to_hf = false
""".strip()
    )
    return config_path, output_dir


def test_templated_config_validates_without_creating_output_dir(tmp_path):
    config_path, output_dir = write_config(tmp_path)

    config = GenerationPipelineConfig.from_path(config_path, create_output_dir=False)

    assert config.generator_config.default_model == "dummy"
    assert not output_dir.exists()


def test_non_templated_config_is_rejected_clearly(tmp_path):
    config_path, _ = write_config(tmp_path, generator="triplets")

    with pytest.raises(ValueError, match="Only 'templated' generator is supported"):
        GenerationPipelineConfig.from_path(config_path, create_output_dir=False)


def test_prompt_rendering_uses_aliases_and_reports_missing_fields(tmp_path):
    config_path, _ = write_config(tmp_path)

    result = config_utils.render_prompt_from_row(
        config_path, source_name="source", raw_row=json.dumps({"body": "hello"})
    )
    missing = config_utils.render_prompt_from_row(
        config_path, source_name="source", raw_row=json.dumps({"other": "hello"})
    )

    assert result["ok"] is True
    assert result["prompt"] == "Question: hello"
    assert missing["ok"] is False
    assert "field 'text' not found" in missing["errors"][0]


def test_prompt_rendering_supports_jinja_filters(tmp_path):
    config_path, _ = write_config(tmp_path)
    config_text = config_path.read_text()
    config_path.write_text(
        config_text.replace(
            'default_prompt_template = "Question: {{ input.text }}"',
            'default_prompt_template = "Question: {{ input.text | truncate(20) }}"',
        )
    )

    result = config_utils.render_prompt_from_row(
        config_path,
        source_name="source",
        raw_row=json.dumps(
            {
                "body": (
                    "This is a longer sentence that should be truncated by "
                    "Jinja."
                )
            }
        ),
    )

    assert result["ok"] is True
    assert result["prompt"].startswith("Question: This is")
    assert result["prompt"].endswith("...")


def test_output_template_supports_jinja_filters_and_tojson(tmp_path):
    config_path, _ = write_config(tmp_path)
    config_text = config_path.read_text()
    config_path.write_text(
        config_text.replace(
            'output_template = \'{"text": "{{ input.text }}", "answer": "{{ llm_output }}"}\'',
            (
                'output_template = \'{'
                '"title": {{ input.title | tojson }}, '
                '"english": {{ input.text | truncate(20) | tojson }}, '
                '"answer": {{ llm_output | tojson }}'
                '}\''
            ),
        )
    )

    from datagen.generators.templated import TemplatedGenerator

    config = GenerationPipelineConfig.from_path(config_path, create_output_dir=False)
    generator = TemplatedGenerator(config=config.generator_config)
    ds_config = config.generator_config.source_datasets_config[0]

    result = generator._format_output(
        {
            "title": "Example title",
            "text": "This is a longer sentence that should be truncated by Jinja.",
        },
        "translated",
        ds_config,
    )

    assert result["title"] == "Example title"
    assert result["answer"] == "translated"
    assert result["english"].startswith("This is")
    assert result["english"].endswith("...")


def test_dummy_llm_import_and_openai_dict_response_format():
    dummy = DummyLLM(response="ok")
    assert dummy.generate([]) == "ok"

    class Message:
        content = "plain"

    class Choice:
        message = Message()

    class Response:
        usage = None
        choices = [Choice()]

    class Completions:
        def create(self, **kwargs):
            return Response()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    llm = OpenAILLM(client=Client(), model="test")
    assert llm.generate([], response_format={"type": "text"}) == "plain"


def test_openai_model_config_resolves_env_params(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

    config = OpenAIModelConfig.from_config(
        {
            "backend": "openai",
            "params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": {"env": "OPENROUTER_API_KEY"},
                "api_base": {"env": "OPENROUTER_API_BASE"},
            },
        }
    )

    assert config.api_key == "openrouter-key"
    assert config.api_base == "https://openrouter.ai/api/v1"


def test_openai_model_config_keeps_literal_env_like_strings():
    config = OpenAIModelConfig.from_config(
        {
            "backend": "openai",
            "params": {
                "model": "gpt-4.1-mini",
                "api_key": "$OPENROUTER_API_KEY",
                "api_base": "${OPENROUTER_API_BASE}",
            },
        }
    )

    assert config.api_key == "$OPENROUTER_API_KEY"
    assert config.api_base == "${OPENROUTER_API_BASE}"


def test_openai_model_config_rejects_missing_env_param(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="`api_key` references missing environment variable `OPENROUTER_API_KEY`",
    ):
        OpenAIModelConfig.from_config(
            {
                "backend": "openai",
                "params": {
                    "model": "openai/gpt-4.1-mini",
                    "api_key": {"env": "OPENROUTER_API_KEY"},
                },
            }
        )


def test_openai_model_config_rejects_invalid_env_param_shapes():
    base_config = {
        "backend": "openai",
        "params": {"model": "gpt-4.1-mini"},
    }

    config = {
        **base_config,
        "params": {**base_config["params"], "api_key": {"env": "KEY", "default": ""}},
    }
    with pytest.raises(ValueError, match="must only contain an `env` key"):
        OpenAIModelConfig.from_config(config)

    config = {**base_config, "params": {**base_config["params"], "api_key": {"env": ""}}}
    with pytest.raises(ValueError, match="must be a non-empty string"):
        OpenAIModelConfig.from_config(config)

    config = {**base_config, "params": {**base_config["params"], "api_key": 123}}
    with pytest.raises(ValueError, match="must be a string or an env reference"):
        OpenAIModelConfig.from_config(config)


def test_dashboard_editor_validation_and_prompt_route(monkeypatch, tmp_path):
    config_dir = tmp_path / "gen_configs"
    config_dir.mkdir()
    config_path, _ = write_config(config_dir)
    monkeypatch.setattr(config_utils, "CONFIG_DIR", config_dir)

    client = create_app(project_root=tmp_path)
    from fastapi.testclient import TestClient

    test_client = TestClient(client)

    dashboard = test_client.get("/")
    editor = test_client.get(f"/configs/{config_path.name}")
    validation = test_client.post(f"/configs/{config_path.name}/validate")
    prompt = test_client.post(
        f"/configs/{config_path.name}/render-prompt",
        data={"source": "source", "row_json": json.dumps({"body": "hello"})},
        headers={"accept": "application/json"},
    )

    assert dashboard.status_code == 200
    assert "Dataset Studio" in dashboard.text
    assert editor.status_code == 200
    assert "Guided editor for sample.toml" in editor.text
    assert "Save Form Fields" not in editor.text
    assert "Save TOML" not in editor.text
    assert validation.status_code == 200
    assert validation.json()["ok"] is True
    assert prompt.status_code == 200
    assert prompt.json()["prompt"] == "Question: hello"


def test_settings_page_saves_openai_model_with_literal_base_and_key(tmp_path):
    settings_path = tmp_path / ".datagen" / "settings.toml"
    app = create_app(project_root=tmp_path, settings_path=settings_path)
    from fastapi.testclient import TestClient

    test_client = TestClient(app)

    page = test_client.get("/settings")
    saved = test_client.post(
        "/settings/models",
        data={
            "name": "local-gemma",
            "backend": "openai",
            "openai_model": "gemma3:12b",
            "openai_provider_preset": "custom",
            "openai_api_base": "http://localhost:11434/v1",
            "openai_key_source": "literal",
            "openai_api_key_literal": "abc",
            "openai_temperature": "0.2",
            "openai_max_tokens": "512",
            "openai_top_p": "0.9",
            "openai_frequency_penalty": "0.1",
            "openai_presence_penalty": "0.2",
        },
    )

    settings = SettingsStore(settings_path).load()

    assert page.status_code == 200
    assert str(settings_path) in page.text
    assert "Provider preset" in page.text
    assert "API key source" in page.text
    assert "Edit" in page.text or "No reusable models configured yet" in page.text
    assert saved.status_code == 200
    assert settings.models["local-gemma"]["backend"] == "openai"
    assert settings.models["local-gemma"]["params"]["model"] == "gemma3:12b"
    assert settings.models["local-gemma"]["params"]["api_base"] == "http://localhost:11434/v1"
    assert settings.models["local-gemma"]["params"]["api_key"] == "abc"
    assert settings.models["local-gemma"]["params"]["temperature"] == 0.2
    assert settings.models["local-gemma"]["params"]["max_tokens"] == 512
    assert "Edit" in saved.text
    assert "local-gemma" in saved.text
    assert "literal" in saved.text


def test_settings_page_saves_openrouter_preset_env_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    settings_path = tmp_path / ".datagen" / "settings.toml"
    app = create_app(project_root=tmp_path, settings_path=settings_path)
    from fastapi.testclient import TestClient

    test_client = TestClient(app)
    saved = test_client.post(
        "/settings/models",
        data={
            "name": "router",
            "backend": "openai",
            "openai_model": "openai/gpt-4.1-mini",
            "openai_provider_preset": "openrouter",
        },
    )

    settings = SettingsStore(settings_path).load()

    assert saved.status_code == 200
    assert settings.models["router"]["params"]["api_base"] == "https://openrouter.ai/api/v1"
    assert settings.models["router"]["params"]["api_key"] == {"env": "OPENROUTER_API_KEY"}
    assert "env: OPENROUTER_API_KEY" in saved.text


def test_settings_page_saves_dummy_backend(tmp_path):
    settings_path = tmp_path / ".datagen" / "settings.toml"
    app = create_app(project_root=tmp_path, settings_path=settings_path)
    from fastapi.testclient import TestClient

    test_client = TestClient(app)
    saved = test_client.post(
        "/settings/models",
        data={
            "name": "dummy-model",
            "backend": "dummy",
            "dummy_response": "ok",
        },
    )

    settings = SettingsStore(settings_path).load()

    assert saved.status_code == 200
    assert settings.models["dummy-model"] == {
        "backend": "dummy",
        "params": {"response": "ok"},
    }
    assert "dummy-model" in saved.text
    assert "ok" in saved.text


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"name": "bad", "backend": "other", "openai_model": "gpt-4.1-mini"},
            "Backend must be openai or dummy",
        ),
        (
            {"name": "bad", "backend": "openai", "openai_provider_preset": "openai"},
            "OpenAI model is required",
        ),
        (
            {
                "name": "bad",
                "backend": "openai",
                "openai_model": "gpt-4.1-mini",
                "openai_provider_preset": "openai",
                "openai_temperature": "hot",
            },
            "Temperature must be a number",
        ),
        (
            {
                "name": "bad",
                "backend": "openai",
                "openai_model": "gpt-4.1-mini",
                "openai_provider_preset": "custom",
                "openai_key_source": "env",
            },
            "API key environment variable name is required",
        ),
        (
            {"name": "bad", "backend": "dummy", "dummy_response": ""},
            "Dummy response is required",
        ),
        (
            {
                "name": "bad",
                "backend": "openai",
                "params_json": json.dumps({"model": "gpt-4.1-mini"}),
            },
            "Params JSON is no longer accepted",
        ),
    ],
)
def test_settings_model_form_validation_errors(payload, message, tmp_path):
    app = create_app(project_root=tmp_path, settings_path=tmp_path / "settings.toml")
    from fastapi.testclient import TestClient

    test_client = TestClient(app)
    response = test_client.post("/settings/models", data=payload)

    assert response.status_code == 400
    assert message in response.text


def test_settings_model_form_rejects_missing_env_reference(monkeypatch, tmp_path):
    monkeypatch.delenv("MISSING_API_KEY", raising=False)
    app = create_app(project_root=tmp_path, settings_path=tmp_path / "settings.toml")
    from fastapi.testclient import TestClient

    test_client = TestClient(app)
    response = test_client.post(
        "/settings/models",
        data={
            "name": "missing-env",
            "backend": "openai",
            "openai_model": "gpt-4.1-mini",
            "openai_provider_preset": "custom",
            "openai_key_source": "env",
            "openai_api_key_env": "MISSING_API_KEY",
        },
    )

    assert response.status_code == 400
    assert "`api_key` references missing environment variable `MISSING_API_KEY`" in response.text


def test_guided_builder_creates_config_with_settings_model(monkeypatch, tmp_path):
    config_dir = tmp_path / "gen_configs"
    config_dir.mkdir()
    settings_path = tmp_path / ".datagen" / "settings.toml"
    SettingsStore(settings_path).upsert_model(
        name="configured-model",
        fields={"backend": "dummy", "dummy_response": "ok"},
    )
    monkeypatch.setattr(config_utils, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(builder, "CONFIG_DIR", config_dir)

    app = create_app(project_root=tmp_path, settings_path=settings_path)
    from fastapi.testclient import TestClient

    test_client = TestClient(app)
    payload = {
        "dataset_name": "Built",
        "description": "Built in UI",
        "authors": ["Tester"],
        "generation_output_dir": "../raw_data/built",
        "generation_logging_steps": 10,
        "sources": [{"name": "source", "path": "dummy/source", "split": "train"}],
        "models": [{"kind": "settings", "name": "configured-model"}],
        "default_model": "configured-model",
        "default_system_prompt": "System",
        "default_prompt_template": "Question: {{ input.text }}",
        "output_template": '{"text": "{{ input.text }}", "answer": "{{ llm_output }}"}',
        "source_datasets": [
            {
                "name": "source",
                "max_records": 1,
                "shuffle": False,
                "max_failures": 0.5,
            }
        ],
        "aliases": [{"source": "source", "column_map": {"text": "body"}}],
        "curator": {
            "upload_to_hf": False,
            "train_test_split": True,
            "update_card": True,
            "language": ["en"],
            "license": "mit",
        },
    }

    page = test_client.get("/configs/new")
    preview = test_client.post(
        "/configs/new/preview", data={"config_json": json.dumps(payload)}
    )
    created = test_client.post(
        "/configs/new",
        data={"name": "built.toml", "config_json": json.dumps(payload)},
    )

    assert page.status_code == 200
    assert "configured-model" in page.text
    assert "Preview Dataset" in page.text
    assert 'data-tab="preview"' in page.text
    assert "Template alias" in page.text
    assert preview.status_code == 200
    assert "Config is valid" in preview.text
    assert created.status_code == 200
    assert (config_dir / "built.toml").exists()
    assert config_utils.validate_config_path(config_dir / "built.toml").ok is True

    edit_page = test_client.get("/configs/built.toml")
    payload["description"] = "Updated in builder"
    updated = test_client.post(
        "/configs/built.toml",
        data={"config_json": json.dumps(payload)},
    )

    assert edit_page.status_code == 200
    assert "Saving back to built.toml" in edit_page.text
    assert updated.status_code == 200
    assert "Saved" in updated.text
    assert "Updated in builder" in (config_dir / "built.toml").read_text()


def test_builder_source_preview_returns_columns(monkeypatch, tmp_path):
    def fake_load_dataset(path, subset=None, split="train"):
        assert path == "dummy/source"
        assert subset is None
        assert split == "train"
        return datasets.Dataset.from_list([{"body": "hello", "label": 1}])

    monkeypatch.setattr(config_utils.datasets, "load_dataset", fake_load_dataset)
    app = create_app(project_root=tmp_path, settings_path=tmp_path / "settings.toml")
    from fastapi.testclient import TestClient

    test_client = TestClient(app)
    response = test_client.post(
        "/sources/preview",
        data={
            "path": "dummy/source",
            "subset": "",
            "split": "train",
            "source_dom_id": "source-1",
        },
    )

    assert response.status_code == 200
    assert "body, label" in response.text
    assert "<table" in response.text
    assert "Dataset Meta" in response.text
    assert "preview-tab-panel columns-panel" in response.text
    assert "num_rows" in response.text
    assert "hello" in response.text
    assert "Value(" in response.text


def test_job_manager_can_create_report_and_cancel(tmp_path):
    config_path, _ = write_config(tmp_path)
    manager = JobManager(project_root=tmp_path)
    job = manager.start_generate(
        config_path,
        command_builder=lambda _: [
            "python",
            "-c",
            "import time; print('started', flush=True); time.sleep(30)",
        ],
    )

    deadline = time.time() + 5
    while manager.get(job.id).status == "queued" and time.time() < deadline:
        time.sleep(0.05)

    cancelled = manager.cancel(job.id)

    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert manager.get(job.id).as_dict()["id"] == job.id
