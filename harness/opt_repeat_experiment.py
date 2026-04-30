"""Repeat ``golden_opt_cycle`` to measure wall-time stability (median / spread).

Runs ``harness.smoke`` once (unless ``--skip-smoke``), then ``N`` cycles with
``--skip-smoke`` so each repetition is dominated by simulation + golden compare.

Usage::

    python -m harness.opt_repeat_experiment --reps 15
    python -m harness.opt_repeat_experiment --reps 8 --skip-smoke
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _run_once(tag: str, skip_smoke: bool) -> tuple[float | None, int]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "harness.golden_opt_cycle",
            "--tag",
            tag,
            *([] if not skip_smoke else ["--skip-smoke"]),
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    sums: float | None = None
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("wall_sec:") and "sum=" in line:
            try:
                part = line.split("sum=", 1)[1]
                num = part.replace("s", "").strip()
                sums = float(num)
            except (IndexError, ValueError):
                sums = None
            break
    return sums, proc.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Repeat golden triple + median wall sum.")
    ap.add_argument("--reps", type=int, default=10, help="number of cycles after smoke (default 10)")
    ap.add_argument(
        "--skip-smoke",
        action="store_true",
        help="skip smoke on first cycle too (all cycles use --skip-smoke)",
    )
    ns = ap.parse_args(argv)
    if ns.reps < 1:
        print("--reps must be >= 1", file=sys.stderr)
        return 2

    samples: list[float] = []
    failed = 0
    for i in range(ns.reps):
        tag = f"opt_rep_{i}"
        skip = ns.skip_smoke or (i > 0)
        s, rc = _run_once(tag, skip_smoke=skip)
        if rc != 0:
            failed += 1
            print(f"rep {i + 1}/{ns.reps}: FAIL rc={rc}", file=sys.stderr)
            if s is not None:
                print(f"  (parsed sum={s:.2f}s before fail)", file=sys.stderr)
            continue
        if s is None:
            print(f"rep {i + 1}/{ns.reps}: PASS but could not parse wall sum", file=sys.stderr)
        else:
            samples.append(s)
            print(f"rep {i + 1}/{ns.reps}: PASS sum_wall={s:.2f}s")

    print("---")
    print(f"pass_reps={len(samples)} fail_reps={failed} total={ns.reps}")
    if samples:
        med = statistics.median(samples)
        print(
            f"sum_wall_s: min={min(samples):.2f} max={max(samples):.2f} "
            f"median={med:.2f} mean={statistics.mean(samples):.2f}"
        )
        if len(samples) > 1:
            print(f"stdev={statistics.stdev(samples):.2f}")
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
