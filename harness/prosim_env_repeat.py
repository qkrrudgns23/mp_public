"""
Repeat harness.run with the **same subprocess contract** as ``harness.prosim_job_worker``.

This cannot reproduce Chromium WebGL/tab GPU contention exactly, but matches:
``PYTHONHASHSEED`` (from ``PROSIM_HASH_SEED`` or ``1``), Windows
``HIGH_PRIORITY_CLASS`` child, ``--compact-output``, ``--no-validate``, 5%
progress stepping, metrics file writing.

Examples:

    python -m harness.smoke
    python -m harness.prosim_env_repeat --runs 5
    python -m harness.prosim_env_repeat --runs 10 --input data/Result_storage/default_layout_sim_input.json
    python -m harness.prosim_env_repeat --runs 8 --warmup 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict


_ROOT = Path(__file__).resolve().parents[1]
_RESULT_STORAGE = (_ROOT / "data" / "Result_storage").resolve()


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(_RESULT_STORAGE / "default_layout_sim_input.json"),
        help="sim_input json path",
    )
    parser.add_argument("--runs", type=int, default=5, help="number of sequential runs")
    parser.add_argument(
        "--stem-prefix",
        default="_prosim_env_bench",
        help="temporary result stem prefix (files under Result_storage)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="discard first N runs from printed stats only (runs still executed)",
    )
    parser.add_argument(
        "--progress-step-percent",
        type=float,
        default=5.0,
        help="same as ProSim terminal job worker",
    )
    args = parser.parse_args(argv)

    runs = max(1, int(args.runs))
    warmup = max(0, int(args.warmup))
    if warmup >= runs:
        print("prosim-env-repeat: warmup must be < runs", file=sys.stderr)
        return 2

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        print(f"prosim-env-repeat: input not found: {in_path}", file=sys.stderr)
        return 2
    input_sha256 = hashlib.sha256(in_path.read_bytes()).hexdigest()
    print(f"Input: {in_path}")
    print(f"SHA256: {input_sha256}")
    print(f"PROSIM_HASH_SEED: {_harness_run_env().get('PYTHONHASHSEED', '')}")
    print(f"runs: {runs}  warmup (excluded from stats): {warmup}", flush=True)

    wall_secs: list[float] = []
    run_wall_secs: list[float] = []
    cpu_secs: list[float] = []

    prefix = str(args.stem_prefix or "").strip() or "_prosim_env_bench"
    for i in range(runs):
        stem = f"{prefix}_{i}_{int(time.time() * 1000)}"
        out_path = _RESULT_STORAGE / f"{stem}_sim_result.json"
        progress_path = _RESULT_STORAGE / f".{stem}_prosim_progress.json"
        metrics_path = _RESULT_STORAGE / f".{stem}_prosim_metrics.json"

        for p in (out_path, progress_path, metrics_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

        cmd = [
            sys.executable,
            "-m",
            "harness.run",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "--no-validate",
            "--compact-output",
            "--stem",
            stem,
            "--progress",
            str(progress_path),
            "--progress-step-percent",
            str(float(args.progress_step_percent)),
            "--metrics-file",
            str(metrics_path),
        ]

        tw0 = time.perf_counter()
        rc = subprocess.call(
            cmd,
            cwd=str(_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_harness_run_env(),
            **_harness_run_popen_extra(),
        )
        outer_wall = time.perf_counter() - tw0

        if rc != 0:
            print(f"run {i + 1}: FAILED rc={rc}", file=sys.stderr, flush=True)
            return rc

        m: Dict[str, Any] = {}
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not isinstance(m, dict):
                m = {}
        except Exception as exc:
            print(f"run {i + 1}: metrics read failed: {exc}", file=sys.stderr, flush=True)

        rs = float(m.get("runSimulationWallSec") or 0.0)
        cs = float(m.get("runSimulationCpuSec") or 0.0)
        run_wall_secs.append(rs)
        wall_secs.append(outer_wall)
        cpu_secs.append(cs)
        phs = str(m.get("pythonHashSeed") or "")
        ppc = str(m.get("processPriorityClass") or "")
        print(
            f"run {i + 1:2d}  subprocess_outer_wall={outer_wall:.3f}s  "
            f"runSimulationWallSec={rs:.3f}s  cpu={cs:.3f}s  prio={ppc}  hash_seed={phs}",
            flush=True,
        )

        for p in (out_path, progress_path, metrics_path):
            try:
                p.unlink()
            except OSError:
                pass

    for_stats = run_wall_secs[warmup:]
    print("--- stats (runSimulationWallSec) ---", flush=True)
    print(
        f"all_runs  n={len(run_wall_secs)}  min={min(run_wall_secs):.3f}  max={max(run_wall_secs):.3f}  "
        f"mean={mean(run_wall_secs):.3f}  median={median(run_wall_secs):.3f}",
        flush=True,
    )
    if warmup:
        print(
            f"after_warmup  n={len(for_stats)}  (dropped first {warmup})  "
            f"min={min(for_stats):.3f}  max={max(for_stats):.3f}  "
            f"mean={mean(for_stats):.3f}  median={median(for_stats):.3f}",
            flush=True,
        )
    print("(subprocess_outer_wall spans include interpreter startup/teardown overhead)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
