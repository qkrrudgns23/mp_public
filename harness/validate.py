from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def validate_sim_result(sim_input: Dict[str, Any], sim_result: Dict[str, Any]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    required_top = [
        "baseDate",
        "positions",
        "schedule",
        "flights_detail",
        "deadlock_resolve_event_count",
    ]
    for k in required_top:
        if k not in sim_result:
            issues.append(ValidationIssue("missing_key", f"sim_result missing top-level key: {k}"))

    flights = _as_list(sim_input.get("flights"))
    flight_ids: List[str] = []
    for i, f in enumerate(flights):
        fid = str(_as_dict(f).get("id", "")).strip()
        if not fid:
            issues.append(ValidationIssue("input_flight_id", f"sim_input.flights[{i}] missing id"))
            continue
        flight_ids.append(fid)

    pos = _as_dict(sim_result.get("positions"))
    sched = _as_list(sim_result.get("schedule"))
    detail = _as_list(sim_result.get("flights_detail"))

    if isinstance(pos, dict):
        missing_pos = [fid for fid in flight_ids if fid not in pos]
        if missing_pos:
            issues.append(
                ValidationIssue(
                    "positions_missing",
                    f"positions missing {len(missing_pos)} flight ids (sample: {missing_pos[:3]})",
                )
            )
        for fid in flight_ids:
            tr = pos.get(fid)
            td = _as_dict(tr)
            if td.get("format") != "compact_v2":
                issues.append(ValidationIssue("positions_format", f"positions[{fid}] must be compact_v2"))
                continue
            arrays = [td.get(k) for k in ("t", "x", "y", "v")]
            if not all(isinstance(a, list) for a in arrays):
                issues.append(ValidationIssue("positions_arrays", f"positions[{fid}] missing t/x/y/v arrays"))
                continue
            lens = {len(a) for a in arrays if isinstance(a, list)}
            if len(lens) != 1:
                issues.append(ValidationIssue("positions_arrays", f"positions[{fid}] t/x/y/v length mismatch"))
    else:
        issues.append(ValidationIssue("positions_type", "positions must be an object/dict"))

    sched_by_id: Set[str] = set()
    for i, row in enumerate(sched):
        rd = _as_dict(row)
        fid = str(rd.get("flight_id", "")).strip()
        if not fid:
            issues.append(ValidationIssue("schedule_row", f"schedule[{i}] missing flight_id"))
            continue
        sched_by_id.add(fid)
    missing_sched = [fid for fid in flight_ids if fid not in sched_by_id]
    if missing_sched:
        issues.append(
            ValidationIssue(
                "schedule_missing",
                f"schedule missing {len(missing_sched)} flight ids (sample: {missing_sched[:3]})",
            )
        )

    detail_by_id: Set[str] = set()
    for i, row in enumerate(detail):
        rd = _as_dict(row)
        fid = str(rd.get("flight_id", "")).strip()
        if not fid:
            issues.append(ValidationIssue("detail_row", f"flights_detail[{i}] missing flight_id"))
            continue
        detail_by_id.add(fid)
        if "ok" not in rd:
            issues.append(ValidationIssue("detail_row", f"flights_detail[{i}] missing ok"))
    missing_detail = [fid for fid in flight_ids if fid not in detail_by_id]
    if missing_detail:
        issues.append(
            ValidationIssue(
                "detail_missing",
                f"flights_detail missing {len(missing_detail)} flight ids (sample: {missing_detail[:3]})",
            )
        )

    dc = sim_result.get("deadlock_resolve_event_count")
    if not isinstance(dc, int):
        try:
            int(dc)
        except Exception:
            issues.append(
                ValidationIssue(
                    "deadlock_count_type",
                    "deadlock_resolve_event_count must be an int-like value",
                )
            )

    return issues


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", dest="input_path", required=True)
    p.add_argument("--result", dest="result_path", required=True)
    args = p.parse_args(argv)

    in_path = Path(args.input_path)
    out_path = Path(args.result_path)
    if not in_path.exists():
        print(f"validate: input not found: {in_path}", file=sys.stderr)
        return 2
    if not out_path.exists():
        print(f"validate: result not found: {out_path}", file=sys.stderr)
        return 2

    try:
        sim_input = _as_dict(_load_json(in_path))
    except Exception as e:
        print(f"validate: failed to load input json: {e}", file=sys.stderr)
        return 2
    try:
        sim_result = _as_dict(_load_json(out_path))
    except Exception as e:
        print(f"validate: failed to load result json: {e}", file=sys.stderr)
        return 2

    issues = validate_sim_result(sim_input, sim_result)
    if issues:
        for iss in issues[:50]:
            print(f"FAIL {iss.code}: {iss.message}", file=sys.stderr)
        if len(issues) > 50:
            print(f"... {len(issues)-50} more issues", file=sys.stderr)
        return 1

    print("PASS validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

