import base64
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

MAX_CODE_BYTES = 200_000
MAX_OUTPUT_BYTES = 1_000_000
RESULT_MARKER = "\n__RESULT__"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
HARNESS = Path(__file__).with_name("harness.py")

app = FastAPI(title="Codegen sandbox")


class RunRequest(BaseModel):
    code: str = Field(max_length=MAX_CODE_BYTES)
    entrypoint: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
    args: list = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=dict)
    timeout_s: int = Field(default=60, ge=1, le=300)


class RunResponse(BaseModel):
    ok: bool
    result: object | None = None
    error: str | None = None


def _write_files(workdir: Path, files: dict[str, str]) -> None:
    for name, b64 in files.items():
        if not SAFE_NAME.match(name):
            raise ValueError(f"Unsafe file name: {name!r}")
        (workdir / name).write_bytes(base64.b64decode(b64))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        workdir = Path(tmp)
        try:
            _write_files(workdir, req.files)
            (workdir / "generated.py").write_text(req.code, encoding="utf-8")
            (workdir / "request.json").write_text(
                json.dumps({"entrypoint": req.entrypoint, "args": req.args}), encoding="utf-8"
            )
        except (ValueError, OSError) as exc:
            return RunResponse(ok=False, error=str(exc))

        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(HARNESS)],
                cwd=workdir,
                capture_output=True,
                timeout=req.timeout_s,
                env={
                    key: os.environ[key] for key in ("PATH", "SYSTEMROOT") if key in os.environ
                } | {"HOME": tmp, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return RunResponse(ok=False, error=f"Timed out after {req.timeout_s}s")

    stdout = proc.stdout[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
    if proc.returncode != 0 or RESULT_MARKER not in stdout:
        stderr_tail = proc.stderr[-2000:].decode("utf-8", errors="replace")
        return RunResponse(ok=False, error=stderr_tail or "No result produced")
    try:
        result = json.loads(stdout.rsplit(RESULT_MARKER, 1)[1])
    except json.JSONDecodeError as exc:
        return RunResponse(ok=False, error=f"Result was not JSON: {exc}")
    return RunResponse(ok=True, result=result)
