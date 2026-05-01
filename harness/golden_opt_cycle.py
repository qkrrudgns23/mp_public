"""One optimisation verification cycle after editing ``utils/airside_sim.py``.

Runs: smoke → ``harness.run`` (three pairs) → ``golden_compare`` (three pairs).
Exit 0 only if **all three** golden PASS (default strict; optional numeric leaf tolerances forwarded).

Intended workflow (repeat manually or from an agent loop):
1. Patch ``utils/airside_sim.py`` toward performance or behaviour fix.
2. ``python -m harness.golden_opt_cycle [--tag mytry1]``
3. On FAIL: revert the patch(es). On PASS: optionally record wall times vs baseline.

This replaces "timing-only loops" without a golden gate -- those measure OS noise,
not correctness of code changes.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

_ROOT = Path(__file__).resolve().parents[1]

_PAIRS: List[Tuple[str, str, str]] = [
    ("default_layout", "data/Result_storage/default_layout_sim_input.json", "data/Result_storage/default_layout_sim_result.json"),
    ("large_flight", "data/Result_storage/large_flight_sim_input.json", "data/Result_storage/large_flight_sim_result.json"),
    ("MNL_OSM", "data/Result_storage/MNL_OSM_sim_input.json", "data/Result_storage/MNL_OSM_sim_result.json"),
]

_RE_WALL = re.compile(r"\(([0-9.]+)s\)")


def _must_ok(code: int, label: str) -> None:
    if code != 0:
        print(f"{label}: failed rc={code}", file=sys.stderr)
        raise SystemExit(1)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Single cycle: smoke + 3x run + 3x golden_compare (PASS only when compare passes).",
    )
    ap.add_argument(
        "--tag",
        default="cycle",
        help="suffix for output JSON paths under data/Result_storage/ (avoid clobber)",
    )
    ap.add_argument("--skip-smoke", action="store_true", help="skip harness.smoke")
    ap.add_argument(
        "--float-rtol",
        type=float,
        default=0.0,
        help="golden_compare numeric leaf --float-rtol (default 0: strict equality)",
    )
    ap.add_argument(
        "--float-atol",
        type=float,
        default=0.0,
        help="golden_compare numeric leaf --float-atol (default 0: strict equality)",
    )
    ns = ap.parse_args(argv)

    tag = "".join(c if c.isalnum() or c in "_-" else "_" for c in ns.tag.strip() or "cycle")

    if not ns.skip_smoke:
        smoke = subprocess.run(
            [sys.executable, "-m", "harness.smoke"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
        )
        if smoke.stderr:
            sys.stderr.write(smoke.stderr)
        if smoke.stdout:
            sys.stdout.write(smoke.stdout)
        _must_ok(smoke.returncode, "harness.smoke")

    times: dict[str, float] = {}
    all_pass = True
    print("=== golden_opt_cycle: run + golden_compare ===")
    rtol = float(ns.float_rtol)
    atol = float(ns.float_atol)
    if rtol != 0.0 or atol != 0.0:
        print(f"(numeric leaf rtol={rtol:g} atol={atol:g})", file=sys.stderr)

    for pair_id, inp_rel, golden_rel in _PAIRS:
        inp = _ROOT / inp_rel
        golden = _ROOT / golden_rel
        out_rel = f"data/Result_storage/_golden_opt_{tag}_{pair_id}_out.json"
        out = _ROOT / out_rel

        run_p = subprocess.run(
            [
                sys.executable,
                "-m",
                "harness.run",
                "--input",
                str(inp),
                "--output",
                str(out),
                "--no-validate",
            ],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
        )
        if run_p.stdout:
            print(run_p.stdout.rstrip())
        if run_p.stderr:
            sys.stderr.write(run_p.stderr)
        _must_ok(run_p.returncode, f"harness.run {pair_id}")

        m = _RE_WALL.search(run_p.stdout or "")
        if m:
            times[pair_id] = float(m.group(1))

        gc_cmd: List[str] = [
            sys.executable,
            "-m",
            "harness.golden_compare",
            str(golden),
            str(out),
            "--pair-id",
            pair_id,
        ]
        if rtol != 0.0:
            gc_cmd.extend(["--float-rtol", f"{rtol}"])
        if atol != 0.0:
            gc_cmd.extend(["--float-atol", f"{atol}"])
        gc = subprocess.run(gc_cmd, cwd=str(_ROOT), capture_output=True, text=True)
        if gc.stdout:
            print(gc.stdout.rstrip())
        if gc.stderr:
            sys.stderr.write(gc.stderr)
        if gc.returncode != 0:
            all_pass = False
            print(f"FAIL golden gate: {pair_id}", file=sys.stderr)

    sum_w = sum(times.values()) if len(times) == len(_PAIRS) else float("nan")
    print(f"wall_sec: {times} sum={sum_w:.2f}s" if sum_w == sum_w else f"wall_sec: {times}")
    if not all_pass:
        print(
            "\nOverall: FAIL - revert airside_sim.py changes (or fix until this script exits 0).",
            file=sys.stderr,
        )
        return 1

    print("\nOverall: PASS - golden triple locked; OK to adopt this patch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
