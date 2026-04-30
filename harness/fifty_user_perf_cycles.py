"""50 user-specified perf cycles on ``utils/airside_sim.py``.

One cycle::
  - Take accepted snapshot + apply marathon patch at rotating index (no-op if no match).
  - Run golden triple three times on disk; median wall seconds per scenario.
  - ``golden_compare`` vs committed goldens using run #3 outputs.
  - PASS and not slower vs last accepted -> accept snapshot, write ``*_<N>sec.json`` copies.
  - FAIL golden or slower total wall -> revert disk to accepted snapshot.

No silent fallbacks: patch apply uses ``golden_opt_marathon_steps`` only.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
_AIR = _ROOT / "utils" / "airside_sim.py"

_PAIRS: List[Tuple[str, str, str]] = [
    ("default_layout", "data/Result_storage/default_layout_sim_input.json", "data/Result_storage/default_layout_sim_result.json"),
    ("large_flight", "data/Result_storage/large_flight_sim_input.json", "data/Result_storage/large_flight_sim_result.json"),
    ("MNL_OSM", "data/Result_storage/MNL_OSM_sim_input.json", "data/Result_storage/MNL_OSM_sim_result.json"),
]


def _run_sim_pair(inp: Path, out: Path, tag: str) -> Tuple[float, int]:
    t0 = perf_counter()
    p = subprocess.run(
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
    dt = perf_counter() - t0
    if p.stderr:
        sys.stderr.write(p.stderr)
    if p.stdout:
        print(p.stdout.rstrip())
    if p.returncode != 0:
        print(f"fifty_cycles: harness.run FAILED {tag} rc={p.returncode}", file=sys.stderr)
    return dt, p.returncode


def _triple_times_and_outputs(
    work: Path,
    repetition: int,
) -> Tuple[Dict[str, float], Dict[str, Path], List[int]]:
    times: Dict[str, float] = {}
    outs: Dict[str, Path] = {}
    rcs: List[int] = []
    for pair_id, inp_rel, _golden_rel in _PAIRS:
        inp = _ROOT / inp_rel
        outp = work / f"{pair_id}_rep{repetition}.json"
        dt, rc = _run_sim_pair(inp, outp, f"{pair_id} rep={repetition}")
        times[pair_id] = dt
        outs[pair_id] = outp
        rcs.append(rc)
    return times, outs, rcs


def _golden_compare_output(golden_rel: str, outp: Path, pair_id: str) -> int:
    gc = subprocess.run(
        [
            sys.executable,
            "-m",
            "harness.golden_compare",
            str(_ROOT / golden_rel),
            str(outp),
            "--pair-id",
            pair_id,
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    if gc.stdout:
        print(gc.stdout.rstrip())
    if gc.stderr:
        sys.stderr.write(gc.stderr)
    return gc.returncode


def main(argv: List[str] | None = None) -> int:
    from harness import golden_opt_marathon_steps as steps

    ap = argparse.ArgumentParser(description="50 accept/revert cycles: patch + 3 triples + median + golden + save.")
    ap.add_argument("--cycles", type=int, default=50, help="number of cycles (default 50)")
    ns = ap.parse_args(argv)
    n_cycles = int(ns.cycles)
    if n_cycles < 1:
        print("fifty_cycles: cycles must be >= 1", file=sys.stderr)
        return 2

    nm = steps.patch_count()
    accepted_snap = _AIR.read_text(encoding="utf-8")
    sum_last_accepted: float | None = None
    accepted_count = 0
    revert_count = 0

    for c in range(1, n_cycles + 1):
        idx = (c - 1) % nm
        trial_snap, edited, meta = steps.apply_patch_at_index(idx, accepted_snap)
        _AIR.write_text(trial_snap, encoding="utf-8")
        if edited:
            print(f"\n=== cycle {c}/{n_cycles}: APPLY idx={idx} {meta}")
        else:
            print(f"\n=== cycle {c}/{n_cycles}: SKIP PATCH idx={idx} {meta} (trial == accepted content)")

        series_per_pair: Dict[str, List[float]] = {p[0]: [] for p in _PAIRS}
        last_outs: Dict[str, Path] = {}
        all_rc0 = True

        rs_dir = _ROOT / "data" / "Result_storage"
        rs_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="_fifty_cyc_", dir=str(rs_dir)) as td:
            tdir = Path(td)
            for rep in range(1, 4):
                tmap, omap, rcs = _triple_times_and_outputs(tdir, rep)
                if any(rc != 0 for rc in rcs):
                    all_rc0 = False
                    print(f"fifty_cycles: cycle {c} repetition {rep} harness.run FAILURE", file=sys.stderr)
                    break
                for pair_id in series_per_pair.keys():
                    series_per_pair[pair_id].append(tmap[pair_id])
                if rep == 3:
                    last_outs = dict(omap)

            if not all_rc0:
                _AIR.write_text(accepted_snap, encoding="utf-8")
                revert_count += 1
                print(f"cycle {c}: REVERT run failure.")
                continue

            medians: Dict[str, float] = {
                pid: float(statistics.median(times)) for pid, times in series_per_pair.items()
            }

            all_golden0 = True
            for pair_id, _inp_rel, golden_rel in _PAIRS:
                gc_rc = _golden_compare_output(golden_rel, last_outs[pair_id], pair_id)
                if gc_rc != 0:
                    all_golden0 = False
                    print(f"cycle {c}: FAIL golden {pair_id}", file=sys.stderr)

            if not all_golden0:
                _AIR.write_text(accepted_snap, encoding="utf-8")
                revert_count += 1
                print(f"cycle {c}: REVERT golden mismatch.")
                continue

            sum_m = sum(medians[p[0]] for p in _PAIRS)
            if sum_last_accepted is not None and sum_m > sum_last_accepted + 1e-9:
                _AIR.write_text(accepted_snap, encoding="utf-8")
                revert_count += 1
                print(
                    f"cycle {c}: REVERT slower sum_median_sec={sum_m:.3f} "
                    f"> accepted {float(sum_last_accepted):.3f}"
                )
                continue

            for pair_id, inp_rel, golden_rel in _PAIRS:
                m = medians[pair_id]
                n_sec = int(round(m))
                inp = _ROOT / inp_rel
                golden_path = _ROOT / golden_rel
                dst_in = rs_dir / f"{inp.stem}_{n_sec}sec.json"
                dst_out = rs_dir / f"{golden_path.stem}_{n_sec}sec.json"
                shutil.copy2(inp, dst_in)
                shutil.copy2(last_outs[pair_id], dst_out)
                print(f"cycle {c}: wrote {dst_in.relative_to(_ROOT)!s}")
                print(f"cycle {c}: wrote {dst_out.relative_to(_ROOT)!s} (median {pair_id}={m:.3f}s -> suffix {n_sec}sec)")

            accepted_snap = trial_snap
            _AIR.write_text(accepted_snap, encoding="utf-8")
            sum_last_accepted = sum_m
            accepted_count += 1
            print(f"cycle {c}: ACCEPT sum_median_sec(sum of medians)={sum_m:.3f}s")

    print(
        f"\nfifty_cycles: DONE cycles={n_cycles} accepts={accepted_count} "
        f"reverts={revert_count} patches_in_registry={nm}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
