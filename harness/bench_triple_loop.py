"""Wall-time variability probe only (same binary, repeated runs).

**Do not confuse with an optimisation regime.** Correctness is not exercised after
each round unless ``--golden-first`` is set (golden only round 0). Repeated timing
without code changes estimates OS jitter, not regressions.

For perf work with golden lock, use::

    python -m harness.golden_opt_cycle [--tag LABEL]

once after each ``airside_sim.py`` edit (smoke + 3x run + 3x golden_compare).
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]

_PAIRS: List[Tuple[str, str, str]] = [
    ("default_layout", "data/Result_storage/default_layout_sim_input.json", "data/Result_storage/default_layout_sim_result.json"),
    ("large_flight", "data/Result_storage/large_flight_sim_input.json", "data/Result_storage/large_flight_sim_result.json"),
    ("MNL_OSM", "data/Result_storage/MNL_OSM_sim_input.json", "data/Result_storage/MNL_OSM_sim_result.json"),
]

_RE_WALL = re.compile(r"\(([0-9.]+)s\)")


def _run_pair(pair_id: str, inp_rel: str, out_rel: str) -> float:
    inp = _ROOT / inp_rel
    out = _ROOT / out_rel
    p = subprocess.run(
        [sys.executable, "-m", "harness.run", "--input", str(inp), "--output", str(out), "--no-validate"],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        sys.stderr.write(p.stderr or "")
        sys.stderr.write(p.stdout or "")
        raise RuntimeError(f"harness.run failed {pair_id} rc={p.returncode}")
    m = _RE_WALL.search(p.stdout or "")
    if not m:
        raise RuntimeError(f"no wall time in stdout for {pair_id}: {(p.stdout or '')[-400:]}")
    return float(m.group(1))


def _golden(pair_id: str, golden_rel: str, actual_rel: str) -> None:
    g = subprocess.run(
        [
            sys.executable,
            "-m",
            "harness.golden_compare",
            str(_ROOT / golden_rel),
            str(_ROOT / actual_rel),
            "--pair-id",
            pair_id,
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    if g.returncode != 0:
        sys.stderr.write(g.stderr or "")
        sys.stderr.write(g.stdout or "")
        raise RuntimeError(f"golden_compare failed for {pair_id}")
    print((g.stdout or "").strip())


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Same-code repeated timing (--golden-first compares only round 1). Prefer golden_opt_cycle per code edit.",
    )
    ap.add_argument("--rounds", type=int, default=20, help="number of timing loops (default 20)")
    ap.add_argument("--golden-first", action="store_true", help="run golden_compare on round 0 outputs")
    args = ap.parse_args(argv)

    if args.rounds < 1:
        print("bench_triple_loop: rounds must be >= 1", file=sys.stderr)
        return 2

    sums: List[float] = []
    per_pair: Dict[str, List[float]] = {pid: [] for pid, _, _ in _PAIRS}

    for r in range(args.rounds):
        row: Dict[str, float] = {}
        for pid, inp, gold in _PAIRS:
            outp = f"data/Result_storage/_bench_loop_r{r}_{pid}.json"
            wall = _run_pair(pid, inp, outp)
            row[pid] = wall
            per_pair[pid].append(wall)
            if args.golden_first and r == 0:
                _golden(pid, gold, outp)

        s = sum(row[p] for p, _, _ in _PAIRS)
        sums.append(s)
        print(
            f"round {r + 1}/{args.rounds}: "
            + " ".join(f"{p}={row[p]:.2f}s" for p, _, _ in _PAIRS)
            + f" | sum={s:.2f}s"
        )

    print("--- aggregate over rounds ---")
    print(f"sum_wall: min={min(sums):.2f}s max={max(sums):.2f}s mean={statistics.mean(sums):.2f}s median={statistics.median(sums):.2f}s")
    for pid, _, _ in _PAIRS:
        v = per_pair[pid]
        print(f"  {pid}: min={min(v):.2f}s max={max(v):.2f}s mean={statistics.mean(v):.2f}s median={statistics.median(v):.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
