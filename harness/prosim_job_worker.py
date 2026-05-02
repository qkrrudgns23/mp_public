from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict


_ROOT = Path(__file__).resolve().parents[1]
_RESULT_STORAGE_DIR = (_ROOT / "data" / "Result_storage").resolve()


def _load_json_object(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"expected JSON object: {path}")
    return obj


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _is_safe_result_path(path: Path) -> bool:
    try:
        resolved = path.resolve()
        root = _RESULT_STORAGE_DIR.resolve()
    except OSError:
        return False
    return resolved.parent == root or root in resolved.parents


def _claim_job(path: Path) -> Path | None:
    running_path = path.with_suffix(path.suffix + ".running")
    try:
        path.replace(running_path)
        return running_path
    except FileNotFoundError:
        return None
    except OSError:
        return None


def _harness_run_popen_extra() -> Dict[str, Any]:
    if sys.platform != "win32":
        return {}
    flags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    flags |= int(getattr(subprocess, "HIGH_PRIORITY_CLASS", 0x00000080))
    return {"creationflags": flags}


def _harness_run_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(env.get("PROSIM_HASH_SEED") or "1")
    return env


def _run_job(job_path: Path) -> int:
    job = _load_json_object(job_path)
    job_id = str(job.get("jobId") or job_path.stem)
    stem = str(job.get("stem") or "")
    input_path = Path(str(job.get("inputPath") or "")).resolve()
    output_path = Path(str(job.get("outputPath") or "")).resolve()
    progress_path = Path(str(job.get("progressPath") or "")).resolve()
    metrics_path = Path(str(job.get("metricsPath") or "")).resolve()
    status_path = Path(str(job.get("statusPath") or "")).resolve()
    log_path = Path(str(job.get("logPath") or "")).resolve()
    progress_step = float(job.get("progressStepPercent") or 5.0)

    for p in (input_path, output_path, progress_path, metrics_path, status_path, log_path):
        if not _is_safe_result_path(p):
            raise ValueError(f"unsafe job path: {p}")

    _write_json_atomic(
        status_path,
        {
            "ok": True,
            "state": "running",
            "jobId": job_id,
            "stem": stem,
            "startedAt": time.time(),
        },
    )

    cmd = [
        sys.executable,
        "-m",
        "harness.run",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--no-validate",
        "--compact-output",
        "--stem",
        stem,
        "--progress",
        str(progress_path),
        "--progress-step-percent",
        str(progress_step),
        "--metrics-file",
        str(metrics_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log_fp:
        log_fp.write(f"\n--- terminal ProSim job {job_id} start {time.time():.3f} ---\n")
        log_fp.write(" ".join(cmd) + "\n")
        log_fp.flush()
        rc = subprocess.call(
            cmd,
            cwd=str(_ROOT),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            env=_harness_run_env(),
            **_harness_run_popen_extra(),
        )

    if rc == 0:
        _write_json_atomic(
            status_path,
            {
                "ok": True,
                "state": "completed",
                "jobId": job_id,
                "stem": stem,
                "completedAt": time.time(),
            },
        )
    else:
        _write_json_atomic(
            status_path,
            {
                "ok": False,
                "state": "failed",
                "jobId": job_id,
                "stem": stem,
                "returnCode": int(rc),
                "completedAt": time.time(),
            },
        )
    return int(rc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="process one pending job and exit")
    parser.add_argument("--poll-sec", type=float, default=0.25, help="job polling interval")
    args = parser.parse_args(argv)

    _RESULT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    print("ProSim terminal job worker started.", flush=True)
    print(f"Watching: {_RESULT_STORAGE_DIR}", flush=True)

    while True:
        jobs = sorted(_RESULT_STORAGE_DIR.glob(".*_prosim_job.json"))
        if not jobs:
            if args.once:
                return 0
            time.sleep(max(0.05, float(args.poll_sec)))
            continue

        for job_file in jobs:
            claimed = _claim_job(job_file)
            if claimed is None:
                continue
            rc = 1
            try:
                rc = _run_job(claimed)
            except Exception as exc:
                try:
                    job = _load_json_object(claimed)
                    status_raw = str(job.get("statusPath") or "")
                    status_path = Path(status_raw).resolve() if status_raw else claimed.with_suffix(".status.json")
                    _write_json_atomic(
                        status_path,
                        {
                            "ok": False,
                            "state": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "completedAt": time.time(),
                        },
                    )
                except Exception:
                    pass
                print(f"ProSim job failed: {exc}", file=sys.stderr, flush=True)
            finally:
                try:
                    claimed.unlink()
                except OSError:
                    pass
            if args.once:
                return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
