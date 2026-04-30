"""One-off wall-time bench: BASELINE vs OPT1 vs OPT2 (backed-up airside_sim).

Restores utils/airside_sim.py to OPT2 backup at end.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
_AIR = _ROOT / "utils" / "airside_sim.py"
_BACKUP = _ROOT / "data" / "Result_storage" / "_wall_bench_OPT2_airside_sim.py"

_PAIRS: List[Tuple[str, str, str]] = [
    ("default_layout", "data/Result_storage/default_layout_sim_input.json", "data/Result_storage/_perf_wbench_dl.json"),
    ("large_flight", "data/Result_storage/large_flight_sim_input.json", "data/Result_storage/_perf_wbench_lf.json"),
    ("MNL_OSM", "data/Result_storage/MNL_OSM_sim_input.json", "data/Result_storage/_perf_wbench_mnl.json"),
]


def _run_pair(inp: str, outp: str) -> float:
    p = subprocess.run(
        [sys.executable, "-m", "harness.run", "--input", str(_ROOT / inp), "--output", str(_ROOT / outp), "--no-validate"],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        sys.stderr.write(p.stderr or "")
        sys.stderr.write(p.stdout or "")
        raise RuntimeError(f"harness.run failed rc={p.returncode}")
    m = re.search(r"\(([0-9.]+)s\)", p.stdout or "")
    if not m:
        raise RuntimeError(f"no time in stdout tail: {(p.stdout or '')[-600:]}")
    return float(m.group(1))


def _runs_min(pair_id: str, inp: str, outp: str, n: int = 3) -> Tuple[float, List[float]]:
    vals = [_run_pair(inp, outp) for _ in range(n)]
    return min(vals), vals


def main() -> int:
    if not _BACKUP.exists():
        print(f"wall_bench_triple: backup missing: {_BACKUP} (cp utils/airside_sim.py first)", file=sys.stderr)
        return 2

    results: Dict[str, Dict[str, Tuple[float, List[float]]]] = {}

    for tag, rev in [("BASELINE", "221c99b"), ("OPT1", "f940bae")]:
        subprocess.run(["git", "checkout", rev, "--", "utils/airside_sim.py"], cwd=str(_ROOT), check=True)
        subprocess.run([sys.executable, "-m", "harness.smoke"], cwd=str(_ROOT), check=True)
        print(f"=== {tag} (git {rev}) ===", flush=True)
        results[tag] = {}
        for pid, inp, outp in _PAIRS:
            mn, vals = _runs_min(pid, inp, outp, 3)
            results[tag][pid] = (mn, vals)
            print(f"  {pid}: min={mn:.2f}s  runs={vals}", flush=True)

    shutil.copyfile(_BACKUP, _AIR)
    subprocess.run([sys.executable, "-m", "harness.smoke"], cwd=str(_ROOT), check=True)
    print("=== OPT2 (backup file: LOOP2-6) ===", flush=True)
    results["OPT2"] = {}
    for pid, inp, outp in _PAIRS:
        mn, vals = _runs_min(pid, inp, outp, 3)
        results["OPT2"][pid] = (mn, vals)
        print(f"  {pid}: min={mn:.2f}s  runs={vals}", flush=True)

    shutil.copyfile(_BACKUP, _AIR)

    def pct(new: float, old: float) -> float:
        return 100.0 * (old - new) / old if old > 0 else 0.0

    print("\n--- SUMMARY (min of 3 runs) ---", flush=True)
    for pid, _, _ in _PAIRS:
        b = results["BASELINE"][pid][0]
        o1 = results["OPT1"][pid][0]
        o2 = results["OPT2"][pid][0]
        print(
            f"{pid}: base={b:.2f}s  opt1={o1:.2f}s ({pct(o1, b):+.1f}% vs base)  "
            f"opt2={o2:.2f}s ({pct(o2, b):+.1f}% vs base, {pct(o2, o1):+.1f}% vs opt1)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
