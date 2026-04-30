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
    (
        "_agent_occupies_apron_stand_slot: agent_states binder twice",
        """    tt = float(t_abs)
    st0 = control_state.agent_states.get(ag.id)
    if st0 is not None and _agent_deadlock_ghost_at_time(st0, tt):
""",
        """    tt = float(t_abs)
    agent_states_occ_ap = control_state.agent_states.get
    st0 = agent_states_occ_ap(ag.id)
    if st0 is not None and _agent_deadlock_ghost_at_time(st0, tt):
""",
    ),
    (
        "_agent_occupies_apron_stand_slot: second lookup via binder",
        """    if ph0 == PHASE_ARR_TAXI and pt0 == "apron_link":
        st = control_state.agent_states.get(ag.id)
        if (
""",
        """    if ph0 == PHASE_ARR_TAXI and pt0 == "apron_link":
        st = agent_states_occ_ap(ag.id)
        if (
""",
    ),
    (
        "_try_stamp_actual_apron_inblocks_from_stand_position: agent_states binder",
        """    if abs(float(ag.velocity_ms)) > v_max + 1e-9:
        return
    st_ag = control_state.agent_states.get(ag.id)
""",
        """    if abs(float(ag.velocity_ms)) > v_max + 1e-9:
        return
    agent_states_stamp_ib = control_state.agent_states.get
    st_ag = agent_states_stamp_ib(ag.id)
""",
    ),
    (
        "_try_reroute_agent_off_path_block: agent_states binder",
        """    if str(agent.edge_phases[0]) in (
        PHASE_LANDING,
        PHASE_PUSHBACK,
        PHASE_HOLDING_LINEUP,
        PHASE_LINEUP_DEPARTURE,
    ):
        return False
    st = control_state.agent_states.get(agent.id)
""",
        """    if str(agent.edge_phases[0]) in (
        PHASE_LANDING,
        PHASE_PUSHBACK,
        PHASE_HOLDING_LINEUP,
        PHASE_LINEUP_DEPARTURE,
    ):
        return False
    agent_states_reroute_op = control_state.agent_states.get
    st = agent_states_reroute_op(agent.id)
""",
    ),
    (
        "run_simulation history loop: hoist agent_states and edge_resources binders",
        """        _refresh_touchdown_motion_cache(control_state, agents, rw_release_lag)
        stand_cooldown_index = _build_stand_pushback_clearance_index(agents)
        for ag in agents:
            # Always record history (even before touchdown / during runway-separation hold).
""",
        """        _refresh_touchdown_motion_cache(control_state, agents, rw_release_lag)
        stand_cooldown_index = _build_stand_pushback_clearance_index(agents)
        agent_states_hist_tick = control_state.agent_states.get
        edge_hist_tick = control_state.edge_resources.get
        for ag in agents:
            # Always record history (even before touchdown / during runway-separation hold).
""",
    ),
    (
        "run_simulation history loop: agent_states_hist_tick lookup",
        """                stand_cooldown_index = _build_stand_pushback_clearance_index(agents)
            st_h = control_state.agent_states.get(ag.id)
            _gh = (
""",
        """                stand_cooldown_index = _build_stand_pushback_clearance_index(agents)
            st_h = agent_states_hist_tick(ag.id)
            _gh = (
""",
    ),
    (
        "run_simulation history loop: edge_hist_tick (no per-iter binder)",
        """            _rw0: Optional[str] = None
            _edge_hist = control_state.edge_resources.get
            if _eid0:
                _er0 = _edge_hist(_eid0)
""",
        """            _rw0: Optional[str] = None
            if _eid0:
                _er0 = edge_hist_tick(_eid0)
""",
    ),
    (
        "reserve_path: hoist edge runway intersection binders",
        """    edge_resources = control_state.edge_resources
    runway_resources = control_state.runway_resources
    intersection_resources = control_state.intersection_resources

    def _billed_at(idx_i: int) -> int:
""",
        """    edge_resources = control_state.edge_resources
    runway_resources = control_state.runway_resources
    intersection_resources = control_state.intersection_resources
    edge_get_rp = edge_resources.get
    rw_get_rp = runway_resources.get
    ir_get_rp = intersection_resources.get

    def _billed_at(idx_i: int) -> int:
""",
    ),
    (
        "reserve_path: loops use hoisted getters",
        """    for idx, eid in enumerate(lookahead):
        er = edge_resources.get(eid)
        if er is None:
            continue
        if er.runway_id:
            if idx >= depth_cap:
                continue
            if aid not in er.reserved_by:
                er.reserved_by.append(aid)
            rr = runway_resources.get(str(er.runway_id))
            if rr is not None and aid not in rr.reserved_by:
                rr.reserved_by.append(aid)
            continue
        if not _edge_uses_full_depth_reservation(agent, idx, control_state):
            continue
        billed = _billed_at(idx)
        if billed > depth_cap:
            continue
        if aid not in er.reserved_by:
            er.reserved_by.append(aid)
    for k in range(len(lookahead) - 1):
        er_k = edge_resources.get(lookahead[k])
        if er_k is None or not er_k.intersection_out:
            continue
        if er_k.runway_id:
            if k >= depth_cap:
                continue
        else:
            if not _edge_uses_full_depth_reservation(agent, k, control_state):
                continue
            bk = _billed_at(k)
            if bk > depth_cap:
                continue
        iid = er_k.intersection_out
        ir = intersection_resources.get(iid)
        if ir is not None and aid not in ir.reserved_by:
            ir.reserved_by.append(aid)
        st.reserved_intersections.append(iid)
""",
        """    for idx, eid in enumerate(lookahead):
        er = edge_get_rp(eid)
        if er is None:
            continue
        if er.runway_id:
            if idx >= depth_cap:
                continue
            if aid not in er.reserved_by:
                er.reserved_by.append(aid)
            rr = rw_get_rp(str(er.runway_id))
            if rr is not None and aid not in rr.reserved_by:
                rr.reserved_by.append(aid)
            continue
        if not _edge_uses_full_depth_reservation(agent, idx, control_state):
            continue
        billed = _billed_at(idx)
        if billed > depth_cap:
            continue
        if aid not in er.reserved_by:
            er.reserved_by.append(aid)
    for k in range(len(lookahead) - 1):
        er_k = edge_get_rp(lookahead[k])
        if er_k is None or not er_k.intersection_out:
            continue
        if er_k.runway_id:
            if k >= depth_cap:
                continue
        else:
            if not _edge_uses_full_depth_reservation(agent, k, control_state):
                continue
            bk = _billed_at(k)
            if bk > depth_cap:
                continue
        iid = er_k.intersection_out
        ir = ir_get_rp(iid)
        if ir is not None and aid not in ir.reserved_by:
            ir.reserved_by.append(aid)
        st.reserved_intersections.append(iid)
""",
    ),
    (
        "_yield_temp_occupied_incident_edges_for_pathfinding: stand_resources binder",
        """    aid = str(exclude_flight_id)
    for tid, e_set in inc.items():
        sr = control_state.stand_resources.get(str(tid))
""",
        """    aid = str(exclude_flight_id)
    stand_yield_temp = control_state.stand_resources.get
    for tid, e_set in inc.items():
        sr = stand_yield_temp(str(tid))
""",
    ),
    (
        "can_reserve_path lookahead: hoist resource getters",
        """    aid_key = str(aid)

    for idx, eid in enumerate(lookahead):
        billed_here: Optional[int] = None
        er = edge_resources.get(eid)
        if er is None:
            return False, f"unknown_edge:{eid}"
        for ts_id in edge_incident_temp_stands.get(str(eid), ()):
            sr_t = stand_resources.get(ts_id)
""",
        """    aid_key = str(aid)
    edge_get_crp = edge_resources.get
    stand_get_crp = stand_resources.get
    rw_get_crp = runway_resources.get
    ir_get_crp = intersection_resources.get

    for idx, eid in enumerate(lookahead):
        billed_here: Optional[int] = None
        er = edge_get_crp(eid)
        if er is None:
            return False, f"unknown_edge:{eid}"
        for ts_id in edge_incident_temp_stands.get(str(eid), ()):
            sr_t = stand_get_crp(ts_id)
""",
    ),
    (
        "can_reserve_path: stand sid lookup via binder",
        """            sid = str(agent.apron_stand_id or "").strip()
            if sid:
                sr = stand_resources.get(sid)
""",
        """            sid = str(agent.apron_stand_id or "").strip()
            if sid:
                sr = stand_get_crp(sid)
""",
    ),
    (
        "can_reserve_path: runway dep lookup rr_dep",
        """            and pt0 in ("runway", "runway_taxiway")
        ):
            rr_dep = runway_resources.get(dep_rwy)
""",
        """            and pt0 in ("runway", "runway_taxiway")
        ):
            rr_dep = rw_get_crp(dep_rwy)
""",
    ),
    (
        "can_reserve_path: runway dep rr_dep_b",
        """            if rem_m is not None and rem_m <= float(DEP_RUNWAY_HOLD_BUFFER_M):
                rr_dep_b = runway_resources.get(dep_rwy)
""",
        """            if rem_m is not None and rem_m <= float(DEP_RUNWAY_HOLD_BUFFER_M):
                rr_dep_b = rw_get_crp(dep_rwy)
""",
    ),
    (
        "can_reserve_path: runway rr on edge",
        """        if er.runway_id:
            rwid = str(er.runway_id)
            rr = runway_resources.get(rwid)
""",
        """        if er.runway_id:
            rwid = str(er.runway_id)
            rr = rw_get_crp(rwid)
""",
    ),
    (
        "can_reserve_path: intersection ir",
        """            ir_id = er.intersection_out
            if ir_id:
                ir = intersection_resources.get(ir_id)
""",
        """            ir_id = er.intersection_out
            if ir_id:
                ir = ir_get_crp(ir_id)
""",
    ),
    (
        "refresh_resource_occupancy: stand_resources.get binder in tick loop",
        """    intersection_resources = control_state.intersection_resources
    agent_states_get = control_state.agent_states.get
    for ag in agents:
        td = _arr_touchdown_motion_abs_sec(
            ag, agents, rw_lag, control_state=control_state
        )
        if td is not None and t_abs + 1e-9 < float(td):
            continue
        st_ag = agent_states_get(ag.id)
        if st_ag is not None and _agent_deadlock_ghost_at_time(st_ag, t_abs):
            continue
        if not ag.edge_ids:
            temp_wait = _agent_occupies_temp_stand_slot(ag, t_abs, control_state)
            if temp_wait:
                sr_w = stand_resources.get(temp_wait)
""",
        """    intersection_resources = control_state.intersection_resources
    agent_states_get = control_state.agent_states.get
    stand_get_occ = stand_resources.get
    for ag in agents:
        td = _arr_touchdown_motion_abs_sec(
            ag, agents, rw_lag, control_state=control_state
        )
        if td is not None and t_abs + 1e-9 < float(td):
            continue
        st_ag = agent_states_get(ag.id)
        if st_ag is not None and _agent_deadlock_ghost_at_time(st_ag, t_abs):
            continue
        if not ag.edge_ids:
            temp_wait = _agent_occupies_temp_stand_slot(ag, t_abs, control_state)
            if temp_wait:
                sr_w = stand_get_occ(temp_wait)
""",
    ),
    (
        "refresh_resource_occupancy: stand_get_occ sid_slot and temp_slot",
        """        sid_slot = _agent_occupies_apron_stand_slot(ag, t_abs, control_state)
        if sid_slot:
            sr = stand_resources.get(sid_slot)
            if sr is not None and ag.id not in sr.occupied_by:
                sr.occupied_by.append(ag.id)
        temp_slot = _agent_occupies_temp_stand_slot(ag, t_abs, control_state)
        if temp_slot:
            sr_t = stand_resources.get(temp_slot)
""",
        """        sid_slot = _agent_occupies_apron_stand_slot(ag, t_abs, control_state)
        if sid_slot:
            sr = stand_get_occ(sid_slot)
            if sr is not None and ag.id not in sr.occupied_by:
                sr.occupied_by.append(ag.id)
        temp_slot = _agent_occupies_temp_stand_slot(ag, t_abs, control_state)
        if temp_slot:
            sr_t = stand_get_occ(temp_slot)
""",
    ),
    (
        "refresh_resource_occupancy: bind dicts early + getter binders",
        """) -> None:
    for e in control_state.edge_resources.values():
        e.occupied_by.clear()
    for ir in control_state.intersection_resources.values():
        ir.occupied_by.clear()
    for rr in control_state.runway_resources.values():
        rr.occupied_by.clear()
    for sr in control_state.stand_resources.values():
        sr.occupied_by.clear()
    g = control_state.path_graph
    ppm = max(float(pixels_per_meter), 1e-9)
    rad_px = NODE_OCCUPANCY_RADIUS_M * ppm
    t_abs = float(sim_time_abs)
    rw_lag = float(runway_release_lag_sec)
    edge_resources = control_state.edge_resources
    stand_resources = control_state.stand_resources
    runway_resources = control_state.runway_resources
    intersection_resources = control_state.intersection_resources
    agent_states_get = control_state.agent_states.get
    stand_get_occ = stand_resources.get
    for ag in agents:
        td = _arr_touchdown_motion_abs_sec(
            ag, agents, rw_lag, control_state=control_state
        )
        if td is not None and t_abs + 1e-9 < float(td):
            continue
        st_ag = agent_states_get(ag.id)
        if st_ag is not None and _agent_deadlock_ghost_at_time(st_ag, t_abs):
            continue
        if not ag.edge_ids:
            temp_wait = _agent_occupies_temp_stand_slot(ag, t_abs, control_state)
            if temp_wait:
                sr_w = stand_get_occ(temp_wait)
                if sr_w is not None and ag.id not in sr_w.occupied_by:
                    sr_w.occupied_by.append(ag.id)
            continue
        eid0 = str(ag.edge_ids[0])
        er = edge_resources.get(eid0)
        if er and ag.id not in er.occupied_by:
            er.occupied_by.append(ag.id)
        if er and er.runway_id:
            rr = runway_resources.get(str(er.runway_id))
            if rr and ag.id not in rr.occupied_by:
                rr.occupied_by.append(ag.id)
        dep_rw = str(ag.dep_runway_id or "").strip()
        ph0 = str(ag.edge_phases[0]) if ag.edge_phases else ""
        pt0 = (
            str(ag.segment_path_types[0] or "")
            if ag.segment_path_types and len(ag.segment_path_types) == len(ag.edge_ids)
            else ""
        )
        if (
            dep_rw
            and ph0 in (PHASE_HOLDING_LINEUP, PHASE_LINEUP_DEPARTURE)
            and pt0 in ("runway", "runway_taxiway")
        ):
            rr_d = runway_resources.get(dep_rw)
            if rr_d and ag.id not in rr_d.occupied_by:
                rr_d.occupied_by.append(ag.id)
        sid_slot = _agent_occupies_apron_stand_slot(ag, t_abs, control_state)
        if sid_slot:
            sr = stand_get_occ(sid_slot)
            if sr is not None and ag.id not in sr.occupied_by:
                sr.occupied_by.append(ag.id)
        temp_slot = _agent_occupies_temp_stand_slot(ag, t_abs, control_state)
        if temp_slot:
            sr_t = stand_get_occ(temp_slot)
            if sr_t is not None and ag.id not in sr_t.occupied_by:
                sr_t.occupied_by.append(ag.id)
        if not g or not er or not ag.segment_endpoints:
            continue
        pxy = (float(ag.col), float(ag.row))
        for nid in (er.intersection_in, er.intersection_out):
            if not nid:
                continue
            idx_s = nid[1:] if nid.startswith("N") else ""
            if not idx_s.isdigit():
                continue
            ni = int(idx_s)
            if ni < 0 or ni >= len(g.nodes):
                continue
            if path_dist(pxy, g.nodes[ni]) <= rad_px:
                ir = intersection_resources.get(nid)
                if ir and ag.id not in ir.occupied_by:
                    ir.occupied_by.append(ag.id)
""",
        """) -> None:
    edge_resources = control_state.edge_resources
    intersection_resources = control_state.intersection_resources
    runway_resources = control_state.runway_resources
    stand_resources = control_state.stand_resources
    for e in edge_resources.values():
        e.occupied_by.clear()
    for ir in intersection_resources.values():
        ir.occupied_by.clear()
    for rr in runway_resources.values():
        rr.occupied_by.clear()
    for sr in stand_resources.values():
        sr.occupied_by.clear()
    g = control_state.path_graph
    ppm = max(float(pixels_per_meter), 1e-9)
    rad_px = NODE_OCCUPANCY_RADIUS_M * ppm
    t_abs = float(sim_time_abs)
    rw_lag = float(runway_release_lag_sec)
    agent_states_get = control_state.agent_states.get
    stand_get_occ = stand_resources.get
    edge_get_occ = edge_resources.get
    rw_get_occ = runway_resources.get
    ir_get_occ = intersection_resources.get
    for ag in agents:
        td = _arr_touchdown_motion_abs_sec(
            ag, agents, rw_lag, control_state=control_state
        )
        if td is not None and t_abs + 1e-9 < float(td):
            continue
        st_ag = agent_states_get(ag.id)
        if st_ag is not None and _agent_deadlock_ghost_at_time(st_ag, t_abs):
            continue
        if not ag.edge_ids:
            temp_wait = _agent_occupies_temp_stand_slot(ag, t_abs, control_state)
            if temp_wait:
                sr_w = stand_get_occ(temp_wait)
                if sr_w is not None and ag.id not in sr_w.occupied_by:
                    sr_w.occupied_by.append(ag.id)
            continue
        eid0 = str(ag.edge_ids[0])
        er = edge_get_occ(eid0)
        if er and ag.id not in er.occupied_by:
            er.occupied_by.append(ag.id)
        if er and er.runway_id:
            rr = rw_get_occ(str(er.runway_id))
            if rr and ag.id not in rr.occupied_by:
                rr.occupied_by.append(ag.id)
        dep_rw = str(ag.dep_runway_id or "").strip()
        ph0 = str(ag.edge_phases[0]) if ag.edge_phases else ""
        pt0 = (
            str(ag.segment_path_types[0] or "")
            if ag.segment_path_types and len(ag.segment_path_types) == len(ag.edge_ids)
            else ""
        )
        if (
            dep_rw
            and ph0 in (PHASE_HOLDING_LINEUP, PHASE_LINEUP_DEPARTURE)
            and pt0 in ("runway", "runway_taxiway")
        ):
            rr_d = rw_get_occ(dep_rw)
            if rr_d and ag.id not in rr_d.occupied_by:
                rr_d.occupied_by.append(ag.id)
        sid_slot = _agent_occupies_apron_stand_slot(ag, t_abs, control_state)
        if sid_slot:
            sr = stand_get_occ(sid_slot)
            if sr is not None and ag.id not in sr.occupied_by:
                sr.occupied_by.append(ag.id)
        temp_slot = _agent_occupies_temp_stand_slot(ag, t_abs, control_state)
        if temp_slot:
            sr_t = stand_get_occ(temp_slot)
            if sr_t is not None and ag.id not in sr_t.occupied_by:
                sr_t.occupied_by.append(ag.id)
        if not g or not er or not ag.segment_endpoints:
            continue
        pxy = (float(ag.col), float(ag.row))
        for nid in (er.intersection_in, er.intersection_out):
            if not nid:
                continue
            idx_s = nid[1:] if nid.startswith("N") else ""
            if not idx_s.isdigit():
                continue
            ni = int(idx_s)
            if ni < 0 or ni >= len(g.nodes):
                continue
            if path_dist(pxy, g.nodes[ni]) <= rad_px:
                ir = ir_get_occ(nid)
                if ir and ag.id not in ir.occupied_by:
                    ir.occupied_by.append(ag.id)
""",
    ),
    (
        "_temp_stand_has_other_claimant_or_occupant: stand_resources binder",
        """    sr = control_state.stand_resources.get(tid)
    if sr is not None:
        if any(str(x) != aid for x in sr.occupied_by):
""",
        """    stand_claim_temp = control_state.stand_resources.get
    sr = stand_claim_temp(tid)
    if sr is not None:
        if any(str(x) != aid for x in sr.occupied_by):
""",
    ),
    (
        "_pick_temp_stand_for_arrival_detour: stand_resources binder in loop",
        """    if dest_xy is None:
        return None
    aid = str(fobj.get("id") or "")
    ranked: List[Tuple[float, str]] = []
    for tst in raw_ts:
        if not isinstance(tst, dict):
            continue
        tid = str(tst.get("id") or "").strip()
        if not tid:
            continue
        if not _stand_accepts_flight_aircraft(tst, fobj, information):
            continue
        sr = control_state.stand_resources.get(tid)
""",
        """    if dest_xy is None:
        return None
    aid = str(fobj.get("id") or "")
    stand_pick_temp = control_state.stand_resources.get
    ranked: List[Tuple[float, str]] = []
    for tst in raw_ts:
        if not isinstance(tst, dict):
            continue
        tid = str(tst.get("id") or "").strip()
        if not tid:
            continue
        if not _stand_accepts_flight_aircraft(tst, fobj, information):
            continue
        sr = stand_pick_temp(tid)
""",
    ),
    (
        "ensure_agent_control_states: local agent_states dict",
        """def ensure_agent_control_states(control_state: SimulationControlState, agents: Iterable[Flight]) -> None:
    for ag in agents:
        fid = str(ag.id)
        if fid not in control_state.agent_states:
            control_state.agent_states[fid] = AgentControlState(flight_id=fid)
""",
        """def ensure_agent_control_states(control_state: SimulationControlState, agents: Iterable[Flight]) -> None:
    ast = control_state.agent_states
    for ag in agents:
        fid = str(ag.id)
        if fid not in ast:
            ast[fid] = AgentControlState(flight_id=fid)
""",
    ),
    (
        "_single_full_reservation_pass: agent_states.values binder",
        """    _clear_all_reservations(control_state)
    t_dec = float(sim_time)
    for st in control_state.agent_states.values():
        gu = st.deadlock_ghost_until_abs_sec
""",
        """    _clear_all_reservations(control_state)
    t_dec = float(sim_time)
    agent_states_vals = control_state.agent_states.values
    for st in agent_states_vals():
        gu = st.deadlock_ghost_until_abs_sec
""",
    ),
    (
        "_clear_all_reservations: local resource dicts",
        """def _clear_all_reservations(control_state: SimulationControlState) -> None:
    for e in control_state.edge_resources.values():
        e.reserved_by.clear()
    for ir in control_state.intersection_resources.values():
        ir.reserved_by.clear()
    for rr in control_state.runway_resources.values():
        rr.reserved_by.clear()
""",
        """def _clear_all_reservations(control_state: SimulationControlState) -> None:
    er_c = control_state.edge_resources
    ir_c = control_state.intersection_resources
    rr_c = control_state.runway_resources
    for e in er_c.values():
        e.reserved_by.clear()
    for ir in ir_c.values():
        ir.reserved_by.clear()
    for rr in rr_c.values():
        rr.reserved_by.clear()
""",
    ),
    (
        "_expire_forced_open_resources: local resource dicts",
        """def _expire_forced_open_resources(control_state: SimulationControlState, sim_time: float) -> None:
    t = float(sim_time)
    for e in control_state.edge_resources.values():
        u = e.forced_open_until_sec
        if e.forced_open and u is not None and t > float(u):
            e.forced_open = False
            e.forced_open_until_sec = None
    for ir in control_state.intersection_resources.values():
        u = ir.forced_open_until_sec
        if ir.forced_open and u is not None and t > float(u):
            ir.forced_open = False
            ir.forced_open_until_sec = None
    for rr in control_state.runway_resources.values():
        u = rr.forced_open_until_sec
        if rr.forced_open and u is not None and t > float(u):
            rr.forced_open = False
            rr.forced_open_until_sec = None
""",
        """def _expire_forced_open_resources(control_state: SimulationControlState, sim_time: float) -> None:
    t = float(sim_time)
    er_e = control_state.edge_resources
    ir_e = control_state.intersection_resources
    rr_e = control_state.runway_resources
    for e in er_e.values():
        u = e.forced_open_until_sec
        if e.forced_open and u is not None and t > float(u):
            e.forced_open = False
            e.forced_open_until_sec = None
    for ir in ir_e.values():
        u = ir.forced_open_until_sec
        if ir.forced_open and u is not None and t > float(u):
            ir.forced_open = False
            ir.forced_open_until_sec = None
    for rr in rr_e.values():
        u = rr.forced_open_until_sec
        if rr.forced_open and u is not None and t > float(u):
            rr.forced_open = False
            rr.forced_open_until_sec = None
""",
    ),
    (
        "_arr_touchdown_motion_abs_sec: local touchdown_motion_by_id map",
        """    if control_state is not None and control_state.touchdown_motion_by_id is not None:
        return control_state.touchdown_motion_by_id.get(str(agent.id))
""",
        """    if control_state is not None and control_state.touchdown_motion_by_id is not None:
        t_mid = control_state.touchdown_motion_by_id
        return t_mid.get(str(agent.id))
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
