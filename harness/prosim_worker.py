from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict


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
    args = parser.parse_args(argv)

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()

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
    result = run_simulation(layout, dt=float(args.dt), progress_cb=None)
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
