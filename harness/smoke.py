from __future__ import annotations

import py_compile
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def _must_exist(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        print("smoke: no args supported", file=sys.stderr)
        return 2

    info_path = _ROOT / "data" / "Info_storage" / "Information.json"
    _must_exist(info_path)

    targets = [
        _ROOT / "utils" / "airside_sim.py",
        _ROOT / "harness" / "run.py",
        _ROOT / "harness" / "validate.py",
        _ROOT / "harness" / "golden_compare.py",
        _ROOT / "harness" / "golden_opt_cycle.py",
        _ROOT / "harness" / "golden_opt_marathon.py",
        _ROOT / "harness" / "golden_opt_marathon_steps.py",
        _ROOT / "harness" / "triple_timed_golden_snap.py",
        _ROOT / "harness" / "bench_triple_loop.py",
        _ROOT / "harness" / "smoke.py",
        _ROOT / "harness" / "multi_apron_regression.py",
    ]
    for t in targets:
        _must_exist(t)
        py_compile.compile(str(t), doraise=True)

    from utils.airside_sim import run_simulation  # noqa: F401

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

