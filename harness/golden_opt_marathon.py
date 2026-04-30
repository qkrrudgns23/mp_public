"""Run up to N optimisation iterations: apply one PAT patch, then golden_opt_cycle.

After each accepted patch definition in ``golden_opt_marathon_steps``, this runs::
  smoke -> 3x harness.run -> 3x golden_compare.

If golden fails for that iteration, ``utils/airside_sim.py`` is reverted for that patch.
Iterations beyond ``golden_opt_marathon_steps.patch_count()`` SKIP work (nothing to patch)
until you extend the registry.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from harness import golden_opt_marathon_steps as steps

_ROOT = Path(__file__).resolve().parents[1]
_AIR = _ROOT / "utils" / "airside_sim.py"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Golden-locked iterative perf patches (repeat N times).")
    ap.add_argument("--iterations", type=int, default=50, help="attempted iterations (default 50)")
    ap.add_argument(
        "--abort-on-registry-exhaust",
        action="store_true",
        help="exit 5 when iteration index exceeds available patches instead of SKIP logging",
    )
    ns = ap.parse_args(argv)
    iterations = ns.iterations
    if iterations < 1:
        print("golden_opt_marathon: iterations must be >= 1", file=sys.stderr)
        return 2

    text_cur = _AIR.read_text(encoding="utf-8")
    nm = steps.patch_count()
    adopted = 0

    print(
        f"golden_opt_marathon: AIR={_AIR.relative_to(_ROOT)!s} patches_in_registry={nm} iterations_requested={iterations}"
    )

    for it in range(1, iterations + 1):
        idx = it - 1
        if idx >= nm:
            msg = (
                f"iteration {it}/{iterations}: SKIP (no patch at index {idx}; "
                f"extend harness/golden_opt_marathon_steps.py - currently {nm} patches)"
            )
            if ns.abort_on_registry_exhaust:
                print(msg, file=sys.stderr)
                return 5
            print(msg)
            continue

        new_t, edited, meta = steps.apply_patch_at_index(idx, text_cur)
        if not edited:
            print(f"iteration {it}/{iterations}: SKIP {meta}")
            continue

        snap = text_cur
        print(f"iteration {it}/{iterations}: APPLY {meta}")
        _AIR.write_text(new_t, encoding="utf-8")

        gc = subprocess.run(
            [sys.executable, "-m", "harness.golden_opt_cycle", "--tag", f"marathon_{it}"],
            cwd=str(_ROOT),
        )

        if gc.returncode != 0:
            _AIR.write_text(snap, encoding="utf-8")
            print(f"iteration {it}: GOLDEN_CYCLE FAIL rc={gc.returncode} - reverted this patch.", file=sys.stderr)
            text_cur = snap
            return 1

        adopted += 1
        print(f"iteration {it}: adopted OK (total patches adopted this marathon: {adopted})\n")

        text_cur = new_t

    print(f"golden_opt_marathon: DONE adopted_new_deltas={adopted}/{nm} (skipped pre-applied/already-done steps as SKIP)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
