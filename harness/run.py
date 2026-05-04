from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from harness.validate import validate_sim_result


_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"expected json object at top-level: {path}")
    return obj


def _process_priority_label() -> str:
    if os.name != "nt":
        try:
            return str(os.getpriority(os.PRIO_PROCESS, 0))
        except Exception:
            return ""
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetPriorityClass.argtypes = [wintypes.HANDLE]
        kernel32.GetPriorityClass.restype = wintypes.DWORD
        handle = kernel32.GetCurrentProcess()
        cls = kernel32.GetPriorityClass(handle)
        return str(int(cls))
    except Exception:
        return ""


def run_simulation_job(
    *,
    input_path: Path,
    output_path: Path,
    progress_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
    stem: str = "",
    dt: float = 5.0,
    no_validate: bool = False,
    compact_output: bool = False,
    progress_step_percent: float = 5.0,
    metrics_json: bool = False,
) -> int:
    """Run ``run_simulation``, write result/metrics/progress; same behavior as CLI ``main``."""
    in_path = Path(input_path)
    out_path = Path(output_path)
    if not in_path.exists():
        print(f"run: input not found: {in_path}", file=sys.stderr)
        return 2

    t_load0 = time.perf_counter()
    try:
        input_text = in_path.read_text(encoding="utf-8")
        input_sha256 = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
        obj = json.loads(input_text)
        if not isinstance(obj, dict):
            raise TypeError(f"expected json object at top-level: {in_path}")
        sim_input = obj
    except Exception as e:
        print(f"run: failed to load input json: {e}", file=sys.stderr)
        return 2
    t_load1 = time.perf_counter()

    t0 = time.time()
    sim_result: Optional[Dict[str, Any]] = None
    try:
        from utils.airside_sim import run_simulation

        t_run0 = time.perf_counter()
        cpu_run0 = time.process_time()
        progress_counts: Dict[str, Any] = {
            "calls": 0,
            "writes": 0,
            "last_write": 0.0,
            "last_pct": -1,
        }

        def _write_progress(
            current_time: float, total_time: float, _sim_time_abs: Optional[float]
        ) -> None:
            progress_counts["calls"] += 1
            if progress_path is None:
                return
            now = time.perf_counter()
            pct_raw = (
                100 * float(current_time) / float(total_time)
                if float(total_time) > 0
                else 0
            )
            pct = int(pct_raw)
            pct = max(0, min(100, pct))
            step = float(progress_step_percent) if progress_step_percent else 0.0
            if math.isfinite(step) and step > 0:
                pct = (
                    100
                    if pct_raw >= 100.0
                    else int(max(0.0, min(100.0, math.floor(pct_raw / step) * step)))
                )
                if pct <= int(progress_counts["last_pct"]) and pct < 100:
                    return
            elif progress_counts["last_write"] and now - progress_counts["last_write"] < 1.0:
                return
            progress_counts["last_pct"] = pct
            progress_counts["last_write"] = now
            row = {
                "percent": pct,
                "elapsedSec": max(0.0, now - t_run0),
                "current": float(current_time),
                "total": float(total_time),
            }
            try:
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = progress_path.with_suffix(progress_path.suffix + ".tmp")
                tmp.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
                tmp.replace(progress_path)
                progress_counts["writes"] += 1
            except Exception:
                pass

        sim_result = run_simulation(
            sim_input,
            dt=float(dt),
            progress_cb=(_write_progress if progress_path is not None else None),
            progress_step_percent=(
                float(progress_step_percent) if progress_path is not None else 0.0
            ),
        )
        cpu_run1 = time.process_time()
        t_run1 = time.perf_counter()
    except Exception as e:
        print(f"run: runtime failure: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    dt_wall = time.time() - t0
    assert sim_result is not None
    output = dict(sim_result) if isinstance(sim_result, dict) else {}
    if compact_output:
        output.pop("flight_edge_paths", None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_dump0 = time.perf_counter()
    payload = json.dumps(output, ensure_ascii=False, indent=2, default=str)
    t_dump1 = time.perf_counter()
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(out_path)
    t_write1 = time.perf_counter()
    payload_bytes = len(payload.encode("utf-8"))
    print(f"run: wrote {out_path} ({dt_wall:.2f}s)")
    metrics_obj = {
        "ok": True,
        "resultStem": str(stem or ""),
        "pythonExecutable": sys.executable,
        "pythonHashSeed": os.environ.get("PYTHONHASHSEED", ""),
        "processPriorityClass": _process_priority_label(),
        "ompNumThreads": os.environ.get("OMP_NUM_THREADS", ""),
        "pythonPath": os.environ.get("PYTHONPATH", ""),
        "inputSha256": input_sha256,
        "inputLoadWallSec": round(t_load1 - t_load0, 6),
        "runSimulationWallSec": round(t_run1 - t_run0, 6),
        "runSimulationCpuSec": round(cpu_run1 - cpu_run0, 6),
        "progressCbCalls": int(progress_counts["calls"]),
        "progressWrites": int(progress_counts["writes"]),
        "jsonDumpsWallSec": round(t_dump1 - t_dump0, 6),
        "resultWriteWallSec": round(t_write1 - t_dump1, 6),
        "payloadUtf8Bytes": int(payload_bytes),
    }
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_tmp = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
        metrics_tmp.write_text(json.dumps(metrics_obj, ensure_ascii=False), encoding="utf-8")
        metrics_tmp.replace(metrics_path)
    if metrics_json:
        print(json.dumps(metrics_obj, ensure_ascii=False), flush=True)

    if no_validate:
        return 0

    issues = validate_sim_result(sim_input, output)
    if issues:
        for iss in issues[:50]:
            print(f"FAIL {iss.code}: {iss.message}", file=sys.stderr)
        if len(issues) > 50:
            print(f"... {len(issues)-50} more issues", file=sys.stderr)
        return 1

    print("PASS run+validate")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(_ROOT / "data" / "Result_storage" / "default_layout_sim_input.json"),
        help="sim_input json path",
    )
    parser.add_argument(
        "--output",
        default=str(_ROOT / "data" / "Result_storage" / "_validation_sim_result.json"),
        help="sim_result json path (harness output)",
    )
    parser.add_argument("--dt", type=float, default=5.0, help="simulation dt step (seconds)")
    parser.add_argument("--no-validate", action="store_true", help="skip validate step")
    parser.add_argument("--progress", default="", help="optional progress JSON path")
    parser.add_argument(
        "--progress-step-percent",
        type=float,
        default=5.0,
        help="minimum percent step between run_simulation progress callbacks",
    )
    parser.add_argument("--stem", default="", help="result stem for diagnostics")
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="drop transient debug-only result fields before writing output",
    )
    parser.add_argument(
        "--metrics-json",
        action="store_true",
        help="print machine-readable run metrics as the final stdout line",
    )
    parser.add_argument("--metrics-file", default="", help="optional metrics JSON output path")
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.output)
    progress_path = Path(args.progress).resolve() if str(args.progress or "").strip() else None
    metrics_path = Path(args.metrics_file).resolve() if str(args.metrics_file or "").strip() else None

    return run_simulation_job(
        input_path=in_path,
        output_path=out_path,
        progress_path=progress_path,
        metrics_path=metrics_path,
        stem=str(args.stem or ""),
        dt=float(args.dt),
        no_validate=bool(args.no_validate),
        compact_output=bool(args.compact_output),
        progress_step_percent=float(args.progress_step_percent),
        metrics_json=bool(args.metrics_json),
    )


if __name__ == "__main__":
    raise SystemExit(main())

