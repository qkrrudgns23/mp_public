"""Sequential text patches for ``golden_opt_marathon`` (one golden cycle per applied patch).

Each ``old`` must match **exactly once** before that step runs. If ``old`` is absent
(already merged by hand / earlier marathon), that step becomes a noop (no file write,
no subprocess).
"""

from __future__ import annotations

from typing import List, Tuple

Step = Tuple[str, str, str]

_PATCHES: List[Step] = [
    (
        "_current_edge_separation_ok: agent_states.get local",
        """    rw_lag = float(runway_release_lag_sec)
    eid_sp = str(er.edge_id)
    for o in agents:
        if not o.edge_ids or str(o.edge_ids[0]) != eid_sp:
            continue
        if o.id == agent.id or not o.segment_endpoints:
            continue
        o_td = _arr_touchdown_motion_abs_sec(
            o, agents, rw_lag, control_state=control_state
        )
        if o_td is not None and t_eff + 1e-9 < float(o_td):
            continue
        st_o = control_state.agent_states.get(o.id)
""",
        """    rw_lag = float(runway_release_lag_sec)
    eid_sp = str(er.edge_id)
    agent_states_get_sep = control_state.agent_states.get
    for o in agents:
        if not o.edge_ids or str(o.edge_ids[0]) != eid_sp:
            continue
        if o.id == agent.id or not o.segment_endpoints:
            continue
        o_td = _arr_touchdown_motion_abs_sec(
            o, agents, rw_lag, control_state=control_state
        )
        if o_td is not None and t_eff + 1e-9 < float(o_td):
            continue
        st_o = agent_states_get_sep(o.id)
""",
    ),
    (
        "resolve_head_on_conflict: agent_states binder",
        """    if not detect_head_on_conflict(agent_a, agent_b, control_state):
        return
    eid0 = str(agent_a.edge_ids[0])
    sta = control_state.agent_states.get(agent_a.id)
    stb = control_state.agent_states.get(agent_b.id)
""",
        """    if not detect_head_on_conflict(agent_a, agent_b, control_state):
        return
    eid0 = str(agent_a.edge_ids[0])
    aget_h = control_state.agent_states.get
    sta = aget_h(agent_a.id)
    stb = aget_h(agent_b.id)
""",
    ),
    (
        "resolve_head_on_conflict: loser lookup",
        """    else:
        loser = agent_a
    st = control_state.agent_states.get(loser.id)
""",
        """    else:
        loser = agent_a
    st = aget_h(loser.id)
""",
    ),
    (
        "_reroute_all_moving_flights_after_temp_park_arrival: loop get",
        """    t_abs = float(sim_time_abs)
    for ag in sorted(agents, key=lambda a: str(a.id)):
        if not ag.edge_ids or not ag.edge_phases:
            continue
        if str(ag.edge_phases[0]) == PHASE_LANDING:
            continue
        st = control_state.agent_states.get(ag.id)
""",
        """    t_abs = float(sim_time_abs)
    agent_states_get_tp = control_state.agent_states.get
    for ag in sorted(agents, key=lambda a: str(a.id)):
        if not ag.edge_ids or not ag.edge_phases:
            continue
        if str(ag.edge_phases[0]) == PHASE_LANDING:
            continue
        st = agent_states_get_tp(ag.id)
""",
    ),
    (
        "should_reroute_agent: binder",
        """def should_reroute_agent(
    agent: Flight,
    control_state: SimulationControlState,
    sim_time: float,
    information: Dict[str, Any],
) -> bool:
    st = control_state.agent_states.get(agent.id)
""",
        """def should_reroute_agent(
    agent: Flight,
    control_state: SimulationControlState,
    sim_time: float,
    information: Dict[str, Any],
) -> bool:
    agent_states_get_sr = control_state.agent_states.get
    st = agent_states_get_sr(agent.id)
""",
    ),
    (
        "_try_one_aggressive_deadlock_reroute: wait_rank binder",
        """    agents_by_id: Dict[str, Flight] = {str(a.id): a for a in agents}

    def _wait_rank(fid: str) -> float:
        st = control_state.agent_states.get(fid)
""",
        """    agents_by_id: Dict[str, Flight] = {str(a.id): a for a in agents}
    agent_states_get_dl = control_state.agent_states.get

    def _wait_rank(fid: str) -> float:
        st = agent_states_get_dl(fid)
""",
    ),
    (
        "_update_deadlock_stagnation_probe: loop binder",
        """def _update_deadlock_stagnation_probe(
    control_state: SimulationControlState,
    agents: List[Flight],
    sim_time: float,
    pixels_per_meter: float,
    runway_release_lag_sec: float = 0.0,
) -> None:
    ppm = max(float(pixels_per_meter), 1e-9)
    t = float(sim_time)
    for ag in agents:
        st = control_state.agent_states.get(ag.id)
""",
        """def _update_deadlock_stagnation_probe(
    control_state: SimulationControlState,
    agents: List[Flight],
    sim_time: float,
    pixels_per_meter: float,
    runway_release_lag_sec: float = 0.0,
) -> None:
    ppm = max(float(pixels_per_meter), 1e-9)
    t = float(sim_time)
    agent_states_get_us = control_state.agent_states.get
    for ag in agents:
        st = agent_states_get_us(ag.id)
""",
    ),
    (
        "_apply_same_direction_following_caps: binds",
        """def _apply_same_direction_following_caps(
    control_state: SimulationControlState,
    agents: List[Flight],
    ppm: float,
    sim_time_abs: float,
) -> None:
    by_edge: Dict[str, List[Flight]] = {}
    t_eff = float(sim_time_abs)
    for ag in agents:
        if not ag.edge_ids or ag.control_halt:
            continue
        st = control_state.agent_states.get(ag.id)
""",
        """def _apply_same_direction_following_caps(
    control_state: SimulationControlState,
    agents: List[Flight],
    ppm: float,
    sim_time_abs: float,
) -> None:
    by_edge: Dict[str, List[Flight]] = {}
    t_eff = float(sim_time_abs)
    agent_states_get_sf = control_state.agent_states.get
    edge_resources_get_sf = control_state.edge_resources.get
    for ag in agents:
        if not ag.edge_ids or ag.control_halt:
            continue
        st = agent_states_get_sf(ag.id)
""",
    ),
    (
        "_apply_same_direction_following_caps: er lookup",
        """            if not detect_same_direction_conflict(ag_f, ag_l, control_state):
                continue
            gap_m = along_l - along_f
            er = control_state.edge_resources.get(eid)
            min_sep = float(er.min_separation_m) if er else DEFAULT_MIN_SEPARATION_M
""",
        """            if not detect_same_direction_conflict(ag_f, ag_l, control_state):
                continue
            gap_m = along_l - along_f
            er = edge_resources_get_sf(eid)
            min_sep = float(er.min_separation_m) if er else DEFAULT_MIN_SEPARATION_M
""",
    ),
    (
        "compute_following_speed: edge_resources.get binder",
        """def compute_following_speed(
    follower: Flight,
    leader: Flight,
    control_state: SimulationControlState,
    ppm: float,
) -> float:
    er_id = str(follower.edge_ids[0]) if follower.edge_ids else ""
    er = control_state.edge_resources.get(er_id)
""",
        """def compute_following_speed(
    follower: Flight,
    leader: Flight,
    control_state: SimulationControlState,
    ppm: float,
) -> float:
    er_get_cfs = control_state.edge_resources.get
    er_id = str(follower.edge_ids[0]) if follower.edge_ids else ""
    er = er_get_cfs(er_id)
""",
    ),
    (
        "can_enter_intersection: intersection_resources.get binder",
        """def can_enter_intersection(
    agent: Flight, intersection_id: str, control_state: SimulationControlState
) -> bool:
    ir = control_state.intersection_resources.get(intersection_id)
""",
        """def can_enter_intersection(
    agent: Flight, intersection_id: str, control_state: SimulationControlState
) -> bool:
    ir_get_cei = control_state.intersection_resources.get
    ir = ir_get_cei(intersection_id)
""",
    ),
    (
        "reserve_intersection: intersection_resources.get binder",
        """def reserve_intersection(
    agent: Flight,
    intersection_id: str,
    control_state: SimulationControlState,
    sim_time: float,
) -> None:
    del sim_time
    ir = control_state.intersection_resources.get(intersection_id)
""",
        """def reserve_intersection(
    agent: Flight,
    intersection_id: str,
    control_state: SimulationControlState,
    sim_time: float,
) -> None:
    del sim_time
    ir_get_rs = control_state.intersection_resources.get
    ir = ir_get_rs(intersection_id)
""",
    ),
    (
        "_apply_prep_to_agent: agent_states binder",
        """def _apply_prep_to_agent(
    agent: Flight,
    prep: PreparedFlightPath,
    control_state: SimulationControlState,
    sim_time: float,
) -> None:
    st = control_state.agent_states.get(agent.id)
""",
        """def _apply_prep_to_agent(
    agent: Flight,
    prep: PreparedFlightPath,
    control_state: SimulationControlState,
    sim_time: float,
) -> None:
    agent_states_get_ap = control_state.agent_states.get
    st = agent_states_get_ap(agent.id)
""",
    ),
    (
        "_agent_occupies_temp_stand_slot: agent_states binder",
        """    tid = str(ag.temp_stand_id or "").strip()
    if not tid:
        return None
    st0 = control_state.agent_states.get(ag.id)
""",
        """    tid = str(ag.temp_stand_id or "").strip()
    if not tid:
        return None
    agent_states_get_ts = control_state.agent_states.get
    st0 = agent_states_get_ts(ag.id)
""",
    ),
    (
        "compare_agents: agent_states binder",
        """    if pa > pb:
        return 1
    sa = control_state.agent_states.get(agent_a.id)
    sb = control_state.agent_states.get(agent_b.id)
""",
        """    if pa > pb:
        return 1
    agent_states_ca = control_state.agent_states.get
    sa = agent_states_ca(agent_a.id)
    sb = agent_states_ca(agent_b.id)
""",
    ),
    (
        "_target_apron_stand_occupied_by_other: stand_resources binder",
        """    sid = str(agent.apron_stand_id or "").strip()
    if not sid:
        return False
    sr = control_state.stand_resources.get(sid)
    if sr is None:
        return False
""",
        """    sid = str(agent.apron_stand_id or "").strip()
    if not sid:
        return False
    stand_get_tgt = control_state.stand_resources.get
    sr = stand_get_tgt(sid)
    if sr is None:
        return False
""",
    ),
    (
        "_stand_arrival_book_if_pipeline_proceed: agent_states binder",
        """    if not ag.edge_phases or str(ag.edge_phases[0]) != PHASE_ARR_TAXI:
        return
    st = control_state.agent_states.get(ag.id)
    if st is None or not st.reserved_edges:
        return
""",
        """    if not ag.edge_phases or str(ag.edge_phases[0]) != PHASE_ARR_TAXI:
        return
    agent_states_sb = control_state.agent_states.get
    st = agent_states_sb(ag.id)
    if st is None or not st.reserved_edges:
        return
""",
    ),
    (
        "_stand_arrival_book_if_pipeline_proceed: stand_resources binder",
        """    if not has_apron:
        return
    sr = control_state.stand_resources.get(sid)
    if sr is None or ag.id in sr.occupied_by:
        return
""",
        """    if not has_apron:
        return
    stand_res_sb = control_state.stand_resources.get
    sr = stand_res_sb(sid)
    if sr is None or ag.id in sr.occupied_by:
        return
""",
    ),
    (
        "_agent_current_runway_id: edge_resources binder",
        """    if not ag.edge_ids:
        return None
    er0 = control_state.edge_resources.get(str(ag.edge_ids[0]))
""",
        """    if not ag.edge_ids:
        return None
    edge_get_arc = control_state.edge_resources.get
    er0 = edge_get_arc(str(ag.edge_ids[0]))
""",
    ),
    (
        "_layout_edge_path_type: edge_resources binder",
        """    if ptn:
        out = ptn.get(key)
        if out is not None:
            return out
    er = control_state.edge_resources.get(key)
""",
        """    if ptn:
        out = ptn.get(key)
        if out is not None:
            return out
    er_get_lept = control_state.edge_resources.get
    er = er_get_lept(key)
""",
    ),
    (
        "detect_head_on_conflict: edge_resources binder",
        """        return False
    eid = str(agent_a.edge_ids[0])
    er = control_state.edge_resources.get(eid)
    if er is None or er.direction_mode != "bidirectional":
""",
        """        return False
    eid = str(agent_a.edge_ids[0])
    er_get_dhc = control_state.edge_resources.get
    er = er_get_dhc(eid)
    if er is None or er.direction_mode != "bidirectional":
""",
    ),
    (
        "reserve_path: agent_states binder",
        """def reserve_path(
    agent: Flight,
    lookahead: List[str],
    control_state: SimulationControlState,
    sim_time: float,
    reservation_depth: int = RESERV_DEPTH_ARR_TAXI,
) -> None:
    st = control_state.agent_states.get(agent.id)
""",
        """def reserve_path(
    agent: Flight,
    lookahead: List[str],
    control_state: SimulationControlState,
    sim_time: float,
    reservation_depth: int = RESERV_DEPTH_ARR_TAXI,
) -> None:
    agent_states_get_rp = control_state.agent_states.get
    st = agent_states_get_rp(agent.id)
""",
    ),
    (
        "_yield_penalized_layout_edges_for_reroute: agent_states binder",
        """    out: set[str] = set()
    for eid, er in control_state.edge_resources.items():
        for oid in er.occupied_by:
            if oid == exclude_flight_id:
                continue
            st_o = control_state.agent_states.get(oid)
""",
        """    out: set[str] = set()
    agent_states_get_yp = control_state.agent_states.get
    for eid, er in control_state.edge_resources.items():
        for oid in er.occupied_by:
            if oid == exclude_flight_id:
                continue
            st_o = agent_states_get_yp(oid)
""",
    ),
    (
        "_estimate_remaining_route_length_m: edge_resources binder",
        """def _estimate_remaining_route_length_m(
    agent: Flight,
    control_state: SimulationControlState,
) -> float:
    s = 0.0
    for eid in agent.edge_ids:
        er = control_state.edge_resources.get(str(eid))
""",
        """def _estimate_remaining_route_length_m(
    agent: Flight,
    control_state: SimulationControlState,
) -> float:
    s = 0.0
    er_get_est = control_state.edge_resources.get
    for eid in agent.edge_ids:
        er = er_get_est(str(eid))
""",
    ),
    (
        "_apply_reroute_prepared_flight_state: agent_states binder",
        """def _apply_reroute_prepared_flight_state(
    agent: Flight,
    prep: PreparedFlightPath,
    control_state: SimulationControlState,
    sim_time: float,
) -> None:
    st = control_state.agent_states.get(agent.id)
""",
        """def _apply_reroute_prepared_flight_state(
    agent: Flight,
    prep: PreparedFlightPath,
    control_state: SimulationControlState,
    sim_time: float,
) -> None:
    agent_states_get_rrp = control_state.agent_states.get
    st = agent_states_get_rrp(agent.id)
""",
    ),
    (
        "detect_deadlock: agent_states binder",
        """    out: List[str] = []
    for ag in agents:
        st = control_state.agent_states.get(ag.id)
""",
        """    out: List[str] = []
    agent_states_get_dd = control_state.agent_states.get
    for ag in agents:
        st = agent_states_get_dd(ag.id)
""",
    ),
    (
        "resolve_deadlock: agent_states binder",
        """    wait_snap: Dict[str, float] = {}
    stall_snap: Dict[str, Optional[float]] = {}
    for fid in id_set:
        st = control_state.agent_states.get(fid)
""",
        """    wait_snap: Dict[str, float] = {}
    stall_snap: Dict[str, Optional[float]] = {}
    agent_states_get_rd = control_state.agent_states.get
    for fid in id_set:
        st = agent_states_get_rd(fid)
""",
    ),
    (
        "run_simulation history tick: reuse st_h for dbg state",
        """            _dst_snap = _destination_stand_history_snap(
                ag, control_state, agents, float(current_time_abs), stand_cooldown_index
            )
            _st_dbg = control_state.agent_states.get(ag.id)
            _eid0 = str(ag.edge_ids[0]) if ag.edge_ids else ""
""",
        """            _dst_snap = _destination_stand_history_snap(
                ag, control_state, agents, float(current_time_abs), stand_cooldown_index
            )
            _st_dbg = st_h
            _eid0 = str(ag.edge_ids[0]) if ag.edge_ids else ""
""",
    ),
    (
        "run_simulation history tick: reuse st_h for wait accumulator",
        """            )
            st_w = control_state.agent_states.get(ag.id)
            if st_w and st_w.clearance in ("WAIT", "YIELD"):
""",
        """            )
            st_w = st_h
            if st_w and st_w.clearance in ("WAIT", "YIELD"):
""",
    ),
    (
        "run_simulation history tick: edge_resources binder for runway diag",
        """            _rw0: Optional[str] = None
            if _eid0:
                _er0 = control_state.edge_resources.get(_eid0)
""",
        """            _rw0: Optional[str] = None
            _edge_hist = control_state.edge_resources.get
            if _eid0:
                _er0 = _edge_hist(_eid0)
""",
    ),
    (
        "_stand_pipeline_allows_apron_inblocks_stamp: stand_resources binder",
        """    sid = str(ag.apron_stand_id or "").strip()
    if not sid:
        return False
    sr = control_state.stand_resources.get(sid)
    if sr is None:
        return True
""",
        """    sid = str(ag.apron_stand_id or "").strip()
    if not sid:
        return False
    stand_get_pl = control_state.stand_resources.get
    sr = stand_get_pl(sid)
    if sr is None:
        return True
""",
    ),
    (
        "_destination_stand_history_snap: stand_resources binder",
        """    if not sid:
        return None
    booked = int(control_state.stand_arrival_book_snapshot.get(sid, 0))
    cd = _stand_pushback_clearance_cooldown_active(
        sid, str(ag.id), agents, float(t_abs), stand_cooldown_index
    )
    sr = control_state.stand_resources.get(sid)
""",
        """    if not sid:
        return None
    booked = int(control_state.stand_arrival_book_snapshot.get(sid, 0))
    cd = _stand_pushback_clearance_cooldown_active(
        sid, str(ag.id), agents, float(t_abs), stand_cooldown_index
    )
    stand_get_dsh = control_state.stand_resources.get
    sr = stand_get_dsh(sid)
""",
    ),
]


def patch_count() -> int:
    return len(_PATCHES)


def apply_patch_at_index(patch_idx: int, text: str) -> tuple[str, bool, str]:
    """Apply patch ``patch_idx`` if in range and ``old`` matches once. Returns (text, edited?, name_or_note)."""
    if patch_idx < 0 or patch_idx >= len(_PATCHES):
        return text, False, ""
    name, old, new = _PATCHES[patch_idx]
    if old not in text:
        return text, False, f"{name} [skip:no_match]"
    cnt = text.count(old)
    if cnt != 1:
        raise ValueError(f"patch[{patch_idx}] {name!r}: expected 1 occurrence, got {cnt}")
    return text.replace(old, new, 1), True, name
