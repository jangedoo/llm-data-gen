import asyncio
import copy
import json
import tempfile
import time
import tomllib
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from datagen.web.builder import (
    build_config_dict,
    config_to_toml,
    parse_builder_payload,
    payload_from_config_dict,
    validate_builder_config,
    write_builder_config,
)
from datagen.web.config_utils import (
    PROJECT_ROOT,
    ValidationResult,
    get_config_summary,
    inspect_jsonl_file,
    list_config_files,
    list_outputs_for_config,
    load_config_dict,
    preview_hf_source,
    preview_source_dataset,
    render_prompt_from_row,
    resolve_config_path,
    resolve_output_dir,
    validate_config_path,
)
from datagen.web.jobs import JobManager
from datagen.web.settings import SettingsStore
from datagen.web.template_inspector import build_template_context


PACKAGE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = PACKAGE_DIR / "frontend" / "dist"
templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))


def _config_or_404(name: str) -> Path:
    try:
        return resolve_config_path(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _render(
    request: Request, template_name: str, context: dict, status_code: int = 200
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        template_name,
        context,
        status_code=status_code,
    )


def _spa_index() -> Path | None:
    index = FRONTEND_DIST / "index.html"
    return index if index.exists() else None


def _json_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "errors": [message]}, status_code=status_code)


async def _request_payload(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise ValueError(f"Request body must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    return payload


def _config_payload(name: str, settings: Any) -> dict[str, Any]:
    path = _config_or_404(name)
    return payload_from_config_dict(load_config_dict(path), settings)


def _raw_output_dir_from_config(path: Path) -> Path | None:
    try:
        with path.open("rb") as f:
            config = tomllib.load(f)
    except Exception:
        return None
    output = config.get("generation_output_dir")
    if not output:
        return None
    return path.parent.joinpath(Path(output)).resolve()


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _job_snapshot(job: Any) -> dict[str, Any]:
    data = job.as_dict()
    output_dir = _raw_output_dir_from_config(job.config_path)
    if output_dir is not None:
        data["output_dir"] = str(output_dir)
        data["run_state"] = _read_json_file(output_dir / "run_state.json")
        data["summary"] = _read_json_file(output_dir / "summary.json")
    return data


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _trial_config(payload: dict[str, Any], settings: Any, limit: int) -> Path:
    config = build_config_dict(payload, settings)
    config = copy.deepcopy(config)
    run_id = uuid.uuid4().hex[:12]
    trial_root = Path(tempfile.gettempdir()) / "datagen-web-trials" / run_id
    trial_root.mkdir(parents=True, exist_ok=True)
    config["generation_output_dir"] = "output"
    config["generation_logging_steps"] = 1
    config.setdefault("curator", {}).setdefault("params", {})["upload_to_hf"] = False
    params = config["generator"]["params"]
    source_datasets = params.get("source_datasets", [])
    for source_dataset in source_datasets:
        current = int(source_dataset.get("max_records") or limit)
        source_dataset["max_records"] = min(max(limit, 1), current)
    config_path = trial_root / "trial.toml"
    config_path.write_text(config_to_toml(config), encoding="utf-8")
    return config_path


def create_app(
    project_root: Path | None = None, settings_path: Path | None = None
) -> FastAPI:
    project_root = project_root or PROJECT_ROOT
    app = FastAPI(title="Dataset Studio")
    app.state.jobs = JobManager(project_root=project_root)
    app.state.settings = SettingsStore(path=settings_path)
    if (FRONTEND_DIST / "assets").exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(FRONTEND_DIST / "assets")),
            name="frontend-assets",
        )

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        index = _spa_index()
        if index is not None:
            return FileResponse(index)
        summaries = [get_config_summary(path) for path in list_config_files()]
        return _render(
            request,
            "dashboard.html",
            {
                "configs": summaries,
                "jobs": app.state.jobs.recent(),
            },
        )

    @app.get("/app/{path:path}")
    def spa_path(path: str):
        index = _spa_index()
        if index is None:
            raise HTTPException(status_code=404, detail="Frontend build not found")
        return FileResponse(index)

    @app.get("/api/configs")
    def api_configs():
        return {"configs": [get_config_summary(path) for path in list_config_files()]}

    @app.get("/api/configs/{name}")
    def api_config(name: str):
        return {
            "name": name,
            "payload": _config_payload(name, app.state.settings.load()),
        }

    @app.post("/api/configs/preview")
    async def api_preview_config(request: Request):
        try:
            payload = await _request_payload(request)
            config = build_config_dict(payload, app.state.settings.load())
            validation = validate_builder_config(config)
            return {
                "ok": validation.ok,
                "errors": validation.errors,
                "toml": config_to_toml(config),
            }
        except ValueError as exc:
            return _json_error(str(exc))

    @app.post("/api/configs")
    async def api_create_config(request: Request):
        try:
            body = await _request_payload(request)
            name = str(body.get("name") or "")
            payload = body.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            config_path, validation, toml_text = write_builder_config(
                name=name,
                payload_json=json.dumps(payload),
                settings=app.state.settings.load(),
                overwrite=bool(body.get("overwrite")),
            )
            return JSONResponse(
                {
                    "ok": validation.ok,
                    "errors": validation.errors,
                    "name": config_path.name,
                    "toml": toml_text,
                },
                status_code=200 if validation.ok else 400,
            )
        except ValueError as exc:
            return _json_error(str(exc))

    @app.put("/api/configs/{name}")
    async def api_update_config(name: str, request: Request):
        _config_or_404(name)
        try:
            payload = await _request_payload(request)
            config_path, validation, toml_text = write_builder_config(
                name=name,
                payload_json=json.dumps(payload),
                settings=app.state.settings.load(),
                overwrite=True,
            )
            return JSONResponse(
                {
                    "ok": validation.ok,
                    "errors": validation.errors,
                    "name": config_path.name,
                    "toml": toml_text,
                },
                status_code=200 if validation.ok else 400,
            )
        except ValueError as exc:
            return _json_error(str(exc))

    @app.post("/api/configs/{name}/validate")
    def api_validate_config(name: str):
        path = _config_or_404(name)
        result = validate_config_path(path)
        return JSONResponse(result.as_dict(), status_code=200 if result.ok else 400)

    @app.get("/api/settings/models")
    def api_settings_models():
        settings = app.state.settings.load()
        return {"models": settings.models, "settings_path": str(app.state.settings.path)}

    @app.post("/api/settings/models")
    async def api_save_model(request: Request):
        try:
            body = await _request_payload(request)
            name = str(body.pop("name", ""))
            settings = app.state.settings.upsert_model(name=name, fields=body)
            return {"ok": True, "models": settings.models}
        except ValueError as exc:
            return _json_error(str(exc))

    @app.delete("/api/settings/models/{name}")
    def api_delete_model(name: str):
        settings = app.state.settings.delete_model(name)
        return {"ok": True, "models": settings.models}

    @app.post("/api/sources/preview")
    async def api_source_preview(request: Request):
        try:
            payload = await _request_payload(request)
            preview = preview_hf_source(
                path=str(payload.get("path") or ""),
                subset=str(payload.get("subset") or "") or None,
                split=str(payload.get("split") or "train"),
                limit=int(payload.get("limit") or 5),
            )
            return JSONResponse(preview, status_code=200 if preview["ok"] else 400)
        except ValueError as exc:
            return _json_error(str(exc))

    @app.post("/api/templates/context")
    async def api_template_context(request: Request):
        try:
            payload = await _request_payload(request)
            return build_template_context(payload)
        except ValueError as exc:
            return _json_error(str(exc))

    @app.post("/api/templates/render-prompt")
    async def api_render_prompt(request: Request):
        try:
            body = await _request_payload(request)
            payload = body.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            config = build_config_dict(payload, app.state.settings.load())
            validation = validate_builder_config(config)
            if not validation.ok:
                return JSONResponse(
                    {"ok": False, "errors": validation.errors}, status_code=400
                )
            temp_dir = Path(tempfile.gettempdir()) / "datagen-web-render"
            temp_dir.mkdir(parents=True, exist_ok=True)
            path = temp_dir / f"{uuid.uuid4().hex}.toml"
            path.write_text(config_to_toml(config), encoding="utf-8")
            result = render_prompt_from_row(
                path,
                source_name=str(body.get("source") or ""),
                raw_row=json.dumps(body.get("row") or {}),
            )
            return JSONResponse(result, status_code=200 if result["ok"] else 400)
        except ValueError as exc:
            return _json_error(str(exc))

    @app.post("/api/jobs/trial")
    async def api_trial_job(request: Request):
        try:
            body = await _request_payload(request)
            payload = body.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            limit = int(body.get("limit") or 1)
            config_path = _trial_config(payload, app.state.settings.load(), limit)
            validation = validate_config_path(config_path)
            if not validation.ok:
                return JSONResponse(validation.as_dict(), status_code=400)
            job = app.state.jobs.start_generate(config_path)
            job.kind = "trial"
            return JSONResponse(_job_snapshot(job))
        except ValueError as exc:
            return _json_error(str(exc))

    @app.post("/api/jobs/generate")
    async def api_generate_job(request: Request):
        try:
            body = await _request_payload(request)
            config_name = str(body.get("config_name") or "")
            path = _config_or_404(config_name)
            validation = validate_config_path(path)
            if not validation.ok:
                return JSONResponse(validation.as_dict(), status_code=400)
            job = app.state.jobs.start_generate(path)
            return JSONResponse(_job_snapshot(job))
        except ValueError as exc:
            return _json_error(str(exc))

    @app.post("/api/jobs/upload")
    async def api_upload_job(request: Request):
        try:
            body = await _request_payload(request)
            config_name = str(body.get("config_name") or "")
            path = _config_or_404(config_name)
            validation = validate_config_path(path)
            if not validation.ok:
                return JSONResponse(validation.as_dict(), status_code=400)
            job = app.state.jobs.start_upload(path)
            return JSONResponse(_job_snapshot(job))
        except ValueError as exc:
            return _json_error(str(exc))

    @app.get("/api/jobs/{job_id}")
    def api_job_status(job_id: str):
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JSONResponse(_job_snapshot(job))

    @app.post("/api/jobs/{job_id}/cancel")
    def api_cancel_job(job_id: str):
        job = app.state.jobs.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JSONResponse(_job_snapshot(job))

    @app.get("/api/jobs/{job_id}/events")
    async def api_job_events(job_id: str):
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        async def stream():
            last_payload = ""
            while True:
                current = app.state.jobs.get(job_id)
                if current is None:
                    yield _sse("error", {"errors": ["Job not found"]})
                    return
                payload = _job_snapshot(current)
                encoded = json.dumps(payload, default=str, sort_keys=True)
                if encoded != last_payload:
                    last_payload = encoded
                    yield _sse("job", payload)
                if payload["status"] in {"succeeded", "failed", "cancelled"}:
                    yield _sse("done", payload)
                    return
                await asyncio.sleep(1)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/outputs/{config_name}")
    def api_outputs(config_name: str):
        path = _config_or_404(config_name)
        try:
            return list_outputs_for_config(path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/outputs/{config_name}/file")
    def api_output_file(config_name: str, path: str):
        config_path = _config_or_404(config_name)
        output_dir = resolve_output_dir(config_path).resolve()
        jsonl_path = Path(path).resolve()
        if output_dir not in jsonl_path.parents or jsonl_path.suffix != ".jsonl":
            raise HTTPException(status_code=400, detail="Output path is not allowed")
        if not jsonl_path.exists():
            raise HTTPException(status_code=404, detail="Output file not found")
        return JSONResponse(inspect_jsonl_file(jsonl_path, limit=50))

    @app.get("/configs", response_class=HTMLResponse)
    def configs(request: Request):
        summaries = [get_config_summary(path) for path in list_config_files()]
        return _render(request, "partials/config_list.html", {"configs": summaries})

    @app.get("/settings", response_class=HTMLResponse)
    def settings(request: Request):
        return _render(
            request,
            "settings.html",
            {
                "settings": app.state.settings.load(),
                "settings_path": app.state.settings.path,
            },
        )

    @app.get("/configs/new", response_class=HTMLResponse)
    def new_config(request: Request):
        return _render(
            request,
            "config_builder.html",
            {
                "settings": app.state.settings.load(),
                "initial_payload": None,
                "existing_name": None,
                "preview_url": "/configs/new/preview",
                "save_url": "/configs/new",
            },
        )

    @app.post("/configs/new/preview", response_class=HTMLResponse)
    def preview_new_config(request: Request, config_json: str = Form(...)):
        try:
            payload = parse_builder_payload(config_json)
            config = build_config_dict(payload, app.state.settings.load())
            validation = validate_builder_config(config)
            toml_text = config_to_toml(config)
        except ValueError as exc:
            validation = ValidationResult(ok=False, errors=[str(exc)])
            toml_text = ""
        return _render(
            request,
            "partials/builder_result.html",
            {"validation": validation, "toml_text": toml_text},
            status_code=200 if validation.ok else 400,
        )

    @app.post("/configs/new", response_class=HTMLResponse)
    def create_new_config(
        request: Request,
        name: str = Form(...),
        config_json: str = Form(...),
        overwrite: bool = Form(default=False),
    ):
        try:
            config_path, validation, toml_text = write_builder_config(
                name=name,
                payload_json=config_json,
                settings=app.state.settings.load(),
                overwrite=overwrite,
            )
        except ValueError as exc:
            return _render(
                request,
                "partials/builder_result.html",
                {
                    "validation": ValidationResult(ok=False, errors=[str(exc)]),
                    "toml_text": "",
                },
                status_code=400,
            )
        return _render(
            request,
            "partials/builder_result.html",
            {
                "validation": validation,
                "toml_text": toml_text,
                "created_name": config_path.name if validation.ok else None,
            },
            status_code=200 if validation.ok else 400,
        )

    def _render_config_builder(request: Request, name: str) -> HTMLResponse:
        path = _config_or_404(name)
        settings = app.state.settings.load()
        initial_payload = payload_from_config_dict(load_config_dict(path), settings)
        return _render(
            request,
            "config_builder.html",
            {
                "settings": settings,
                "initial_payload": initial_payload,
                "existing_name": name,
                "preview_url": "/configs/new/preview",
                "save_url": f"/configs/{name}",
            },
        )

    def _save_config_builder(
        request: Request,
        name: str,
        config_json: str,
    ) -> HTMLResponse:
        _config_or_404(name)
        try:
            config_path, validation, toml_text = write_builder_config(
                name=name,
                payload_json=config_json,
                settings=app.state.settings.load(),
                overwrite=True,
            )
        except ValueError as exc:
            return _render(
                request,
                "partials/builder_result.html",
                {
                    "validation": ValidationResult(ok=False, errors=[str(exc)]),
                    "toml_text": "",
                },
                status_code=400,
            )
        return _render(
            request,
            "partials/builder_result.html",
            {
                "validation": validation,
                "toml_text": toml_text,
                "saved_name": config_path.name if validation.ok else None,
            },
            status_code=200 if validation.ok else 400,
        )

    @app.get("/configs/{name}/builder", response_class=HTMLResponse)
    def edit_config_builder(request: Request, name: str):
        return _render_config_builder(request, name)

    @app.post("/configs/{name}/builder", response_class=HTMLResponse)
    def save_config_builder(
        request: Request,
        name: str,
        config_json: str = Form(...),
    ):
        return _save_config_builder(request, name, config_json)

    @app.post("/settings/models", response_class=HTMLResponse)
    async def save_model_setting(
        request: Request,
        name: str = Form(...),
    ):
        try:
            form = await request.form()
            if "params_json" in form:
                raise ValueError("Params JSON is no longer accepted; use the structured model fields")
            settings = app.state.settings.upsert_model(name=name, fields=form)
        except ValueError as exc:
            return _render(
                request,
                "partials/settings_models.html",
                {
                    "settings": app.state.settings.load(),
                    "settings_path": app.state.settings.path,
                    "error": str(exc),
                },
                status_code=400,
            )
        return _render(
            request,
            "partials/settings_models.html",
            {
                "settings": settings,
                "settings_path": app.state.settings.path,
                "saved": True,
            },
        )

    @app.post("/settings/models/delete", response_class=HTMLResponse)
    def delete_model_setting(request: Request, name: str = Form(...)):
        settings = app.state.settings.delete_model(name)
        return _render(
            request,
            "partials/settings_models.html",
            {
                "settings": settings,
                "settings_path": app.state.settings.path,
                "saved": True,
            },
        )

    @app.post("/sources/preview", response_class=HTMLResponse)
    def source_preview(
        request: Request,
        path: str = Form(...),
        subset: str = Form(default=""),
        split: str = Form(default="train"),
        source_dom_id: str = Form(default=""),
    ):
        preview = preview_hf_source(path=path, subset=subset, split=split)
        return _render(
            request,
            "partials/source_preview.html",
            {"preview": preview, "source_dom_id": source_dom_id},
            status_code=200 if preview["ok"] else 400,
        )

    @app.get("/configs/{name}", response_class=HTMLResponse)
    def edit_config(request: Request, name: str):
        return _render_config_builder(request, name)

    @app.post("/configs/{name}", response_class=HTMLResponse)
    def save_config(
        request: Request,
        name: str,
        config_json: str = Form(...),
    ):
        return _save_config_builder(request, name, config_json)

    @app.post("/configs/{name}/validate")
    def validate_config(name: str):
        path = _config_or_404(name)
        result = validate_config_path(path)
        return JSONResponse(result.as_dict(), status_code=200 if result.ok else 400)

    @app.get("/configs/{name}/preview", response_class=HTMLResponse)
    def dataset_preview(request: Request, name: str, source: str | None = None):
        path = _config_or_404(name)
        preview = preview_source_dataset(path, source_name=source)
        return _render(request, "partials/preview.html", {"preview": preview})

    @app.post("/configs/{name}/render-prompt")
    def render_prompt(
        request: Request,
        name: str,
        source: str | None = Form(default=None),
        row_json: str = Form(default="{}"),
    ):
        path = _config_or_404(name)
        result = render_prompt_from_row(path, source_name=source, raw_row=row_json)
        wants_json = "application/json" in request.headers.get("accept", "")
        if wants_json:
            return JSONResponse(result, status_code=200 if result["ok"] else 400)
        return _render(
            request,
            "partials/prompt.html",
            {"result": result},
            status_code=200 if result["ok"] else 400,
        )

    @app.post("/jobs/generate")
    def generate_job(config_name: str = Form(...)):
        path = _config_or_404(config_name)
        validation = validate_config_path(path)
        if not validation.ok:
            return JSONResponse(validation.as_dict(), status_code=400)
        job = app.state.jobs.start_generate(path)
        return JSONResponse(job.as_dict())

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JSONResponse(job.as_dict())

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = app.state.jobs.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JSONResponse(job.as_dict())

    @app.get("/outputs/{config_name}", response_class=HTMLResponse)
    def outputs(request: Request, config_name: str):
        path = _config_or_404(config_name)
        try:
            output_data = list_outputs_for_config(path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _render(
            request,
            "outputs.html",
            {"config_name": config_name, "outputs": output_data},
        )

    @app.get("/outputs/{config_name}/file")
    def output_file(config_name: str, path: str):
        config_path = _config_or_404(config_name)
        output_dir = resolve_output_dir(config_path).resolve()
        jsonl_path = Path(path).resolve()
        if output_dir not in jsonl_path.parents or jsonl_path.suffix != ".jsonl":
            raise HTTPException(status_code=400, detail="Output path is not allowed")
        if not jsonl_path.exists():
            raise HTTPException(status_code=404, detail="Output file not found")
        return JSONResponse(inspect_jsonl_file(jsonl_path, limit=50))

    @app.post("/jobs/upload")
    def upload_job(config_name: str = Form(...)):
        path = _config_or_404(config_name)
        validation = validate_config_path(path)
        if not validation.ok:
            return JSONResponse(validation.as_dict(), status_code=400)
        job = app.state.jobs.start_upload(path)
        return JSONResponse(job.as_dict())

    return app
