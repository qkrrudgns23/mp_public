from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


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


def _load_json_object(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"expected JSON object at top-level: {path}")
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="sim_input JSON path")
    parser.add_argument("--output", required=True, help="sim_result JSON path")
    parser.add_argument("--stem", default="", help="result stem for diagnostics")
    parser.add_argument("--dt", type=float, default=1.0, help="simulation dt seconds")
    parser.add_argument("--progress", default="", help="optional progress JSON path")
    args = parser.parse_args(argv)

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    progress_path = Path(args.progress).resolve() if str(args.progress or "").strip() else None

    t_load0 = time.perf_counter()
    input_text = in_path.read_text(encoding="utf-8")
    input_sha256 = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
    obj = json.loads(input_text)
    if not isinstance(obj, dict):
        raise TypeError(f"expected JSON object at top-level: {in_path}")
    layout = obj
    t_load1 = time.perf_counter()

    from utils.airside_sim import run_simulation

    t_run0 = time.perf_counter()
    cpu_run0 = time.process_time()
    progress_counts = {"calls": 0, "writes": 0, "last_write": 0.0}

    def _write_progress(
        current_time: float, total_time: float, _sim_time_abs: Optional[float]
    ) -> None:
        progress_counts["calls"] += 1
        if progress_path is None:
            return
        now = time.perf_counter()
        if progress_counts["last_write"] and now - progress_counts["last_write"] < 1.0:
            return
        progress_counts["last_write"] = now
        pct = int(100 * float(current_time) / float(total_time)) if float(total_time) > 0 else 0
        pct = max(0, min(100, pct))
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

    result = run_simulation(
        layout,
        dt=float(args.dt),
        progress_cb=(_write_progress if progress_path is not None else None),
    )
    cpu_run1 = time.process_time()
    t_run1 = time.perf_counter()

    output = dict(result) if isinstance(result, dict) else {}
    output.pop("flight_edge_paths", None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_dump0 = time.perf_counter()
    payload = json.dumps(output, ensure_ascii=False, indent=2, default=str)
    t_dump1 = time.perf_counter()
    out_path.write_text(payload, encoding="utf-8")
    t_write1 = time.perf_counter()

    print(
        json.dumps(
            {
                "ok": True,
                "resultStem": str(args.stem or ""),
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
                "payloadUtf8Bytes": len(payload.encode("utf-8")),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
