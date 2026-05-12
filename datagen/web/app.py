from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
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


PACKAGE_DIR = Path(__file__).resolve().parent
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


def create_app(
    project_root: Path | None = None, settings_path: Path | None = None
) -> FastAPI:
    project_root = project_root or PROJECT_ROOT
    app = FastAPI(title="Dataset Studio")
    app.state.jobs = JobManager(project_root=project_root)
    app.state.settings = SettingsStore(path=settings_path)
    app.mount(
        "/static",
        StaticFiles(directory=str(PACKAGE_DIR / "static")),
        name="static",
    )

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        summaries = [get_config_summary(path) for path in list_config_files()]
        return _render(
            request,
            "dashboard.html",
            {
                "configs": summaries,
                "jobs": app.state.jobs.recent(),
            },
        )

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
    def save_model_setting(
        request: Request,
        name: str = Form(...),
        backend: str = Form(...),
        params_json: str = Form(default="{}"),
    ):
        try:
            settings = app.state.settings.upsert_model(
                name=name,
                backend=backend,
                params_json=params_json,
            )
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
