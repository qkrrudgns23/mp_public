"""Golden triple harness.run with total wall time; save inputs + outputs under suffixed filenames.

Measures wall seconds for sequential ``python -m harness.run`` across the three canonical
golden pairs (smoke skipped here; run smoke separately if desired). Runs
``golden_compare`` for each produced output vs the committed golden result. On PASS,
copies each input and fresh output into ``data/Result_storage/`` with the same basename
stem plus ``_{total_wall_sec:.3f}s.json``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import List, Tuple

_ROOT = Path(__file__).resolve().parents[1]

_PAIRS: List[Tuple[str, str, str]] = [
    ("default_layout", "data/Result_storage/default_layout_sim_input.json", "data/Result_storage/default_layout_sim_result.json"),
    ("large_flight", "data/Result_storage/large_flight_sim_input.json", "data/Result_storage/large_flight_sim_result.json"),
    ("MNL_OSM", "data/Result_storage/MNL_OSM_sim_input.json", "data/Result_storage/MNL_OSM_sim_result.json"),
]


def _must_ok(rc: int, label: str) -> None:
    if rc != 0:
        print(f"{label}: failed rc={rc}", file=sys.stderr)
        raise SystemExit(1)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Triple run + golden + save snapshots with total wall-time suffix.")
    ap.add_argument(
        "--keep-temp-output",
        action="store_true",
        help="do not delete temporary run outputs after successful compare",
    )
    ns = ap.parse_args(argv)

    temp_dir = Path(tempfile.mkdtemp(prefix="_triple_snap_", dir=str(_ROOT / "data" / "Result_storage")))

    t0 = perf_counter()
    out_paths: list[Tuple[str, Path, Path, Path]] = []
    for pair_id, inp_rel, golden_rel in _PAIRS:
        inp = _ROOT / inp_rel
        golden = _ROOT / golden_rel
        outp = temp_dir / f"{pair_id}_out.json"

        rp = subprocess.run(
            [
                sys.executable,
                "-m",
                "harness.run",
                "--input",
                str(inp),
                "--output",
                str(outp),
                "--no-validate",
            ],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
        )
        if rp.stdout:
            print(rp.stdout.rstrip())
        if rp.stderr:
            sys.stderr.write(rp.stderr)
        _must_ok(rp.returncode, f"harness.run {pair_id}")
        out_paths.append((pair_id, inp, golden, outp))

    total_wall = perf_counter() - t0
    suffix = f"{total_wall:.3f}s"

    rs_dir = _ROOT / "data" / "Result_storage"
    rs_dir.mkdir(parents=True, exist_ok=True)

    all_pass = True
    print(f"\ntriple_timed_golden_snap: total_wall_sec={total_wall:.3f} golden_compare ...")
    for pair_id, inp, golden, outp in out_paths:
        gc = subprocess.run(
            [
                sys.executable,
                "-m",
                "harness.golden_compare",
                str(golden),
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
        if gc.returncode != 0:
            all_pass = False
            print(f"FAIL golden_compare: {pair_id}", file=sys.stderr)

    if not all_pass:
        if not ns.keep_temp_output:
            shutil.rmtree(temp_dir, ignore_errors=True)
        print("\nOverall FAIL: snapshots not written.", file=sys.stderr)
        return 1

    for _pair_id, inp, golden, outp in out_paths:
        inp_stem = inp.stem
        out_stem = golden.stem
        dest_inp = rs_dir / f"{inp_stem}_{suffix}.json"
        dest_out = rs_dir / f"{out_stem}_{suffix}.json"
        shutil.copy2(inp, dest_inp)
        shutil.copy2(outp, dest_out)
        print(f"wrote {dest_inp.relative_to(_ROOT)!s}")
        print(f"wrote {dest_out.relative_to(_ROOT)!s}")

    if not ns.keep_temp_output:
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        print(f"kept temp dir: {temp_dir.relative_to(_ROOT)!s}")

    print(f"\ntriple_timed_golden_snap: DONE total_wall_sec={total_wall:.3f} suffix=_{suffix}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
