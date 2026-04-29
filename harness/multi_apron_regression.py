from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


_ROOT = Path(__file__).resolve().parents[1]


def _load_default_layout() -> Dict[str, Any]:
    path = _ROOT / "data" / "Result_storage" / "default_layout_sim_input.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError("default layout sim input must be a JSON object")
    layout = obj.get("layout") if isinstance(obj.get("layout"), dict) else obj
    if not isinstance(layout, dict):
        raise TypeError("sim input layout must be a JSON object")
    return layout


def _stand_ids(layout: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key in ("pbbStands", "remoteStands", "tempStands"):
        rows = layout.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("id") is not None:
                sid = str(row.get("id")).strip()
                if sid and sid not in out:
                    out.append(sid)
    return out


def _first_flight(layout: Dict[str, Any]) -> Dict[str, Any]:
    flights = layout.get("flights")
    if not isinstance(flights, list) or not flights:
        raise ValueError("default layout has no flights")
    f0 = flights[0]
    if not isinstance(f0, dict):
        raise TypeError("first flight must be an object")
    return f0


def _with_segments(flight: Dict[str, Any], stands: List[str], count: int) -> Dict[str, Any]:
    f = copy.deepcopy(flight)
    if len(stands) < count:
        raise ValueError(f"need at least {count} stands for regression")
    sibt = float(f.get("sibtMin", f.get("timeMin", 0.0)))
    sobt = float(f.get("sobtMin", sibt + count * 40.0))
    if sobt <= sibt:
        sobt = sibt + count * 40.0
    width = (sobt - sibt) / float(count)
    segs = []
    for i in range(count):
        segs.append(
            {
                "standId": stands[i],
                "sibtMin": sibt + width * i,
                "sobtMin": sibt + width * (i + 1),
            }
        )
    f["apronStaySegments"] = segs
    f["arrApronId"] = stands[0]
    f["depApronId"] = stands[count - 1]
    f["standId"] = stands[count - 1]
    tok = f.get("token") if isinstance(f.get("token"), dict) else {}
    tok = dict(tok)
    tok["apronId"] = stands[count - 1]
    f["token"] = tok
    return f


def main(argv: List[str] | None = None) -> int:
    if argv:
        print("multi_apron_regression: no args supported", file=sys.stderr)
        return 2

    from utils import airside_sim as sim

    layout = _load_default_layout()
    info = sim._load_information_json()
    cell_size = float(layout.get("grid", {}).get("cellSize", 20.0))
    ppm = sim._layout_pixels_per_meter(info)
    reverse_cost, merge_r, taxiway_h = sim._path_search_params(info)
    base_flight = _first_flight(layout)
    stands = _stand_ids(layout)
    if len(stands) < 3:
        print("SKIP multi_apron_regression: need at least 3 stands")
        return 0

    # N=1 must preserve the historical phased leg contract and coordinate wrapper.
    legs1 = sim._extract_point_to_path_legs(base_flight, layout, cell_size, information=info)
    pts1 = sim.extract_point_to_paths(base_flight, layout, cell_size, information=info)
    if [leg for leg, _phase, _sid in legs1] != pts1:
        raise AssertionError("N=1 coordinate wrapper diverged from phased leg builder")
    phases1 = [phase for _leg, phase, _sid in legs1]
    if phases1 != list(sim._EXTRACT_LEG_PHASES[: len(phases1)]):
        raise AssertionError(f"N=1 phase contract changed: {phases1}")

    # Same-stand visual splits must collapse to one logical apron stay.
    same = _with_segments(base_flight, [stands[0], stands[0]], 2)
    same_segments = sim._apron_stay_segments_from_flight(same)
    if len(same_segments) != 1:
        raise AssertionError(f"same-stand adjacent split should collapse to N=1, got {same_segments}")

    # N=2 / N=3 distinct stands should expand path phases and schedule list fields.
    for n in (2, 3):
        f = _with_segments(base_flight, stands[:n], n)
        legs = sim._extract_point_to_path_legs(f, layout, cell_size, information=info)
        phases = [phase for _leg, phase, _sid in legs]
        expected_arr = n
        expected_push = n
        if phases.count(sim.PHASE_ARR_TAXI) != expected_arr:
            raise AssertionError(f"N={n} expected {expected_arr} Arr_taxi legs, got {phases}")
        if phases.count(sim.PHASE_PUSHBACK) != expected_push:
            raise AssertionError(f"N={n} expected {expected_push} Pushback legs, got {phases}")
        prep = sim.prepare_flight_path(f, layout, cell_size, reverse_cost, merge_r, taxiway_h, info)
        if not prep.logical_edge_list:
            raise AssertionError(f"N={n} did not produce logical edges")
        eibt_vals = [1000.0 + i * 1000.0 for i in range(n)]
        eobt_vals = [None] * n
        push_vals = [1100.0 + i * 1000.0 for i in range(n)]
        row = sim._build_schedule_row(
            f,
            str(f.get("id", "")),
            prep,
            ppm,
            "2026-03-31",
            actual_apron_inblocks_abs_sec_list=eibt_vals,
            actual_apron_offblocks_abs_sec_list=eobt_vals,
            pushback_finished_abs_sec_list=push_vals,
        )
        if len(row.get("STANDS") or []) != n:
            raise AssertionError(f"N={n} STANDS length mismatch: {row.get('STANDS')}")
        if len(row.get("EIBT_LIST") or []) != n:
            raise AssertionError(f"N={n} EIBT_LIST length mismatch: {row.get('EIBT_LIST')}")
        if row.get("EIBT") != int(eibt_vals[0]):
            raise AssertionError(f"N={n} EIBT alias should use first EIBT_LIST")
        if row.get("E_PUSH_FINISHED") != int(push_vals[-1]):
            raise AssertionError(f"N={n} E_PUSH_FINISHED alias should use last list value")
        if row.get("EOBT") != (row.get("EOBT_LIST") or [None])[-1]:
            raise AssertionError(f"N={n} EOBT alias should use last EOBT_LIST")

    print("PASS multi_apron_regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
