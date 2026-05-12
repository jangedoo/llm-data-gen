import datetime as dt
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


CommandBuilder = Callable[[Path], list[str]]


@dataclass
class Job:
    id: str
    kind: str
    config_path: Path
    command: list[str]
    status: str = "queued"
    started_at: dt.datetime | None = None
    ended_at: dt.datetime | None = None
    return_code: int | None = None
    logs: list[str] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)
    process: subprocess.Popen | None = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "config_path": str(self.config_path),
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "return_code": self.return_code,
            "logs": self.logs[-500:],
            "output_files": self.output_files,
        }


class JobManager:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def _default_generate_command(self, config_path: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "datagen",
            "generate",
            "--config_file",
            str(config_path),
        ]

    def _default_upload_command(self, config_path: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "datagen",
            "push-to-hub",
            "--config_file",
            str(config_path),
        ]

    def start_generate(
        self, config_path: Path, command_builder: CommandBuilder | None = None
    ) -> Job:
        return self._start(
            kind="generate",
            config_path=config_path,
            command=(command_builder or self._default_generate_command)(config_path),
        )

    def start_upload(
        self, config_path: Path, command_builder: CommandBuilder | None = None
    ) -> Job:
        return self._start(
            kind="upload",
            config_path=config_path,
            command=(command_builder or self._default_upload_command)(config_path),
        )

    def _start(self, kind: str, config_path: Path, command: list[str]) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            config_path=config_path,
            command=command,
        )
        with self._lock:
            self.jobs[job.id] = job
        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        thread.start()
        return job

    def _run(self, job: Job) -> None:
        if job.status == "cancelled":
            job.ended_at = dt.datetime.now(dt.UTC)
            return
        job.started_at = dt.datetime.now(dt.UTC)
        job.status = "running"
        before = self._snapshot_jsonl_files()
        try:
            process = subprocess.Popen(
                job.command,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            job.process = process
            assert process.stdout is not None
            for line in process.stdout:
                job.logs.append(line.rstrip())
            job.return_code = process.wait()
            if job.status == "cancelled":
                return
            job.status = "succeeded" if job.return_code == 0 else "failed"
        except Exception as exc:
            job.status = "failed"
            job.logs.append(str(exc))
            job.return_code = -1
        finally:
            job.ended_at = dt.datetime.now(dt.UTC)
            job.output_files = sorted(
                str(p) for p in self._snapshot_jsonl_files() - before
            )
            job.process = None

    def _snapshot_jsonl_files(self) -> set[Path]:
        return set(self.project_root.glob("**/*.jsonl"))

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is None:
            return None
        was_running = job.status == "running"
        if job.status in {"queued", "running"}:
            job.status = "cancelled"
        if job.process is not None and was_running:
            job.process.terminate()
            try:
                job.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                job.process.kill()
        return job

    def recent(self) -> list[Job]:
        return sorted(
            self.jobs.values(),
            key=lambda job: job.started_at or dt.datetime.min.replace(tzinfo=dt.UTC),
            reverse=True,
        )
