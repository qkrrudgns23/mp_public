"""
One-off: MNL OSM, flight id_hmt0zdqsd, intersection N297 block reason at t~29550.
Run: python -m harness.debug_mnl_n297_ycp
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils import airside_sim

TARGET = "id_hmt0zdqsd"
IIS = "N297"
# 08:12:20 .. 08:12:50 (ref day base)
T_LO = 29540.0
T_HI = 29590.0
_MAX_LOG = 12
_log_n = 0
_orig = airside_sim.can_reserve_path


def _wrap(*args: Any, **kwargs: Any) -> Tuple[bool, str]:
    global _log_n
    out = _orig(*args, **kwargs)
    agent = args[0]
    t_abs = float(args[5] if len(args) > 5 else kwargs.get("sim_time", -1e30))
    ok, reason = out[0], out[1]
    if (
        _log_n < _MAX_LOG
        and (not ok)
        and str(agent.id) == TARGET
        and f"intersection:{IIS}" in str(reason)
        and T_LO <= t_abs <= T_HI
    ):
        control_state = args[2]
        agents = args[3]
        lookahead = args[1] if len(args) > 1 else []
        ir = control_state.intersection_resources.get(IIS)
        _log_n += 1
        print("=" * 72, flush=True)
        print("t_abs", t_abs, "HMS", f"{int(t_abs)//3600:02d}:{(int(t_abs)//60)%60:02d}:{int(t_abs)%60:02d}")
        print("ok", ok, "reason", reason, flush=True)
        print("lookahead[0:5]", (lookahead or [])[:5], flush=True)
        if ir is not None:
            print(
                f"  {IIS} capacity={ir.capacity} forced_open={ir.forced_open}",
                flush=True,
            )
            print("  occupied_by", list(ir.occupied_by), flush=True)
            print("  reserved_by", list(ir.reserved_by), flush=True)
        for ag in agents:
            st = control_state.agent_states.get(ag.id)
            if not st:
                continue
            ri = st.reserved_intersections or []
            if any(str(x) == IIS for x in ri) and str(ag.id) != TARGET:
                ph = str(ag.edge_phases[0]) if ag.edge_phases else "?"
                e0 = str(ag.edge_ids[0]) if ag.edge_ids else "?"
                print(
                    f"  other res N297: {ag.id} ph={ph} e0={e0} clear={st.clearance!r} wr={st.wait_reason!r}",
                    flush=True,
                )
    return out


def main() -> int:
    airside_sim.can_reserve_path = _wrap
    p = _ROOT / "data" / "Result_storage" / "MNL_OSM_sim_input.json"
    with p.open("r", encoding="utf-8") as f:
        layout = json.load(f)
    t0 = time.time()
    print("loading + run_simulation", p, flush=True)
    airside_sim.run_simulation(layout, dt=1.0)
    print("wall_s", time.time() - t0, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
