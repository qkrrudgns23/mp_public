from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

from harness.validate import validate_sim_result


_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"expected json object at top-level: {path}")
    return obj


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(_ROOT / "data" / "Result_storage" / "default_layout_sim_input.json"),
        help="sim_input json path",
    )
    parser.add_argument(
        "--output",
        default=str(_ROOT / "data" / "Result_storage" / "_validation_sim_result.json"),
        help="sim_result json path (harness output)",
    )
    parser.add_argument("--dt", type=float, default=1.0, help="simulation dt step (seconds)")
    parser.add_argument("--no-validate", action="store_true", help="skip validate step")
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"run: input not found: {in_path}", file=sys.stderr)
        return 2

    try:
        sim_input = _load_json(in_path)
    except Exception as e:
        print(f"run: failed to load input json: {e}", file=sys.stderr)
        return 2

    t0 = time.time()
    try:
        from utils.airside_sim import run_simulation

        sim_result = run_simulation(sim_input, dt=float(args.dt))
    except Exception as e:
        print(f"run: runtime failure: {type(e).__name__}: {e}", file=sys.stderr)
        raise
    finally:
        dt_wall = time.time() - t0

    _write_json_atomic(out_path, sim_result)
    print(f"run: wrote {out_path} ({dt_wall:.2f}s)")

    if args.no_validate:
        return 0

    issues = validate_sim_result(sim_input, sim_result if isinstance(sim_result, dict) else {})
    if issues:
        for iss in issues[:50]:
            print(f"FAIL {iss.code}: {iss.message}", file=sys.stderr)
        if len(issues) > 50:
            print(f"... {len(issues)-50} more issues", file=sys.stderr)
        return 1

    print("PASS run+validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

