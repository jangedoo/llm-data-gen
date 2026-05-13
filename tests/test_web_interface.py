import json
import time
from pathlib import Path

import pytest
import datasets

from datagen.core.gen_config import GenerationPipelineConfig
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


def test_settings_page_saves_reusable_models(tmp_path):
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
            "params_json": json.dumps(
                {
                    "model": "gemma3:12b",
                    "api_base": "http://localhost:11434/v1",
                    "api_key": "abc",
                }
            ),
        },
    )

    settings = SettingsStore(settings_path).load()

    assert page.status_code == 200
    assert str(settings_path) in page.text
    assert saved.status_code == 200
    assert settings.models["local-gemma"]["backend"] == "openai"
    assert settings.models["local-gemma"]["params"]["model"] == "gemma3:12b"


def test_guided_builder_creates_config_with_settings_model(monkeypatch, tmp_path):
    config_dir = tmp_path / "gen_configs"
    config_dir.mkdir()
    settings_path = tmp_path / ".datagen" / "settings.toml"
    SettingsStore(settings_path).upsert_model(
        name="configured-model",
        backend="dummy",
        params_json=json.dumps({"response": "ok"}),
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
