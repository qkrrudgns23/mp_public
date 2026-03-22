# -*- coding: utf-8 -*-
"""
scenario JSONtake it Plotly Just create a graph HTML included in the report rev2.
- input: JSON (dict, list of dicts, or JSON file path)
- treatment: JSON → Summary Extract → Plotly Batch creation of charts → HTML Report assembly
-Output: HTML Save string or file

Example usage:
  from utils.generate_scenario_report_rev2 import load_scenarios_from_json, build_report_html
  data = load_scenarios_from_json("path/to/scenarios.json")  # or dict / list
  html = build_report_html(data)
  # save: open("report.html","w",encoding="utf-8").write(html)
"""

import html as _html
import json
import math
from pathlib import Path
from typing import Any, List, Union

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent
SCENARIOS_DIR = PROJECT_ROOT / "Scenarios"

# Unified color for each scenario (Same scenario across all reports = same color)
REPORT_SCENARIO_PALETTE = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#dc2626", "#0891b2", "#ca8a04", "#c026d3"]

# location name JSONUse only when you can't read from (my code T1/T2 Remove hardcoding)
_FALLBACK_LOCATION_NAMES = ["Location_1", "Location_2"]


def _hex_to_rgba(hex_color, alpha=0.25):
    """Convert 6-digit hex (e.g. #2563eb) to rgba string for Plotly (fillcolor does not accept #RRGGBBAA)."""
    if not hex_color or not isinstance(hex_color, str) or not hex_color.startswith("#") or len(hex_color) < 7:
        return "rgba(0,0,0,0.2)"
    h = hex_color.lstrip("#")[:6]
    if len(h) != 6:
        return "rgba(0,0,0,0.2)"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _fmt_delay_hours(v, decimals=2):
    """Delay hours Format: Remove .00 if integer (7 not 7.00), If there are few decimalsuntil."""
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.{decimals}f}".rstrip("0").rstrip(".") if decimals else f"{int(v):,}"


def _float(s, key, default=None):
    v = s.get(key) if isinstance(s, dict) else None
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _default_summary():
    return {
        "id": "", "aggregate_base": "", "date_range": {},
        "terminal_names": list(_FALLBACK_LOCATION_NAMES), "focus_terminal": _FALLBACK_LOCATION_NAMES[0], "focus_dep_arr": "Dep",
        "value_97": None, "value_95": None, "by_location": {},
        "total_rank": None, "final_score": None, "units_list": [], "default_units_list": [],
        "summary_labels": {}, "total_sum": None, "capacity": None,
        "excess_over_capacity_sum": None, "excess_as_percent_of_total": None, "total_delay_hours_capacity_constraint": None,
        "design_criteria_value": None, "required_facilities": None, "capacity_per_unit": None,
        "formula": "", "statistics": {}, "daily_hourly_pivot": {"index": [], "columns": [], "data": []},
        "stacked_pivot": {"index": [], "columns": [], "data": []}, "stacked_pivot_group_col": "",
        "summary_by_group": {"index": [], "columns": [], "data": []},
        "graph_data": {},
    }


def extract_summary(s):
    if s is None or not isinstance(s, dict):
        return _default_summary()
    meta = s.get("meta") or {}
    row = s.get("scenario_result_table_row") or {}
    ts = s.get("time_series") or {}
    sv = s.get("sorted_values") or {}
    # terminal(location) Name: Scenario JSONRead first at
    term_names = []
    if isinstance(meta, dict):
        for key in ("terminal_names", "terminals", "locations"):
            val = meta.get(key)
            if isinstance(val, (list, tuple)) and val:
                term_names = [str(x).strip() for x in val if x is not None and str(x).strip()]
                break
        if not term_names:
            rl = meta.get("report_labels") or meta.get("summary_labels") or {}
            if isinstance(rl, dict):
                if isinstance(rl.get("terminal_names"), (list, tuple)) and rl["terminal_names"]:
                    term_names = [str(x).strip() for x in rl["terminal_names"] if x is not None and str(x).strip()]
                elif rl.get("first_terminal") or rl.get("second_terminal"):
                    term_names = [str(rl.get("first_terminal") or "").strip(), str(rl.get("second_terminal") or "").strip()]
                    term_names = [x for x in term_names if x]
    if not term_names:
        term_assignments = (meta.get("terminal_assignments") or {}) if isinstance(meta, dict) else {}
        term_names = sorted(term_assignments.keys()) if term_assignments else []
    if not term_names and isinstance(row, dict) and row:
        seen = set()
        for k in row:
            if "_" in k and (k.endswith("_97.5%") or "_Total" in k or k.endswith("_Dep_97.5%") or k.endswith("_Arr_97.5%")):
                t = k.split("_")[0]
                if t and t not in ("Dep", "Arr", "Rank", "Dist", "Final", "Total"):
                    seen.add(t)
        term_names = sorted(seen) if seen else []
    if not term_names:
        term_names = list(_FALLBACK_LOCATION_NAMES)
    focus_terminal = (meta.get("terminal") or term_names[0]).strip()
    focus_dep_arr = (meta.get("dep_arr") or "Dep").strip()
    term_assign = (meta.get("terminal_assignments") or {}) if isinstance(meta, dict) else {}
    default_assign = meta.get("default_assignments") or meta.get("fixed_assignments") or {}
    # in this line (focus_terminal, focus_dep_arr)Use only values ​​corresponding to
    key_97 = f"{focus_terminal}_{'Dep' if focus_dep_arr.upper() == 'DEP' else 'Arr'}_97.5%"
    key_95 = f"{focus_terminal}_{'Dep' if focus_dep_arr.upper() == 'DEP' else 'Arr'}_95%"
    value_97 = row.get(key_97) if isinstance(row, dict) else None
    value_95 = row.get(key_95) if isinstance(row, dict) else None
    units_raw = row.get(focus_terminal) if isinstance(row.get(focus_terminal), list) else term_assign.get(focus_terminal) or []
    units_list = [str(u).strip() for u in (units_raw or []) if u is not None and str(u).strip()]
    d_raw = default_assign.get(focus_terminal) if isinstance(default_assign, dict) else []
    default_units_list = [str(u).strip() for u in (d_raw or []) if u is not None and str(u).strip()]
    summary_labels = (meta.get("summary_labels") or meta.get("report_labels") or {}) if isinstance(meta, dict) else {}
    pivot = (ts.get("daily_hourly_pivot") or {}) if isinstance(ts, dict) else {}
    pivot_data = pivot.get("data", []) if isinstance(pivot, dict) else []
    pivot_index = pivot.get("index", []) if isinstance(pivot, dict) else []
    pivot_columns = pivot.get("columns", []) if isinstance(pivot, dict) else []
    stk = (ts.get("stacked_pivot") or {}) if isinstance(ts, dict) else {}
    stk_data = stk.get("data", []) if isinstance(stk, dict) else []
    stk_index = stk.get("index", []) if isinstance(stk, dict) else []
    stk_columns = stk.get("columns", []) if isinstance(stk, dict) else []
    sbg_raw = (ts.get("summary_by_group") or {}) if isinstance(ts, dict) else {}
    # Supported Format: (1) tabs: [{label, data: {index,columns,data}}], (2) legacy: {index, columns, data}
    if isinstance(sbg_raw, dict) and sbg_raw.get("tabs"):
        sbg = sbg_raw  # Passed as tab format
    else:
        sbg = {"index": sbg_raw.get("index", []) if isinstance(sbg_raw, dict) else [],
               "columns": sbg_raw.get("columns", []) if isinstance(sbg_raw, dict) else [],
               "data": sbg_raw.get("data", []) if isinstance(sbg_raw, dict) else []}
    # for all locations Dep/Arrstar 97.5%, 95%, capacity, total_sum, excess_%
    by_location = {}
    for loc in term_names:
        ul = row.get(loc) if isinstance(row, dict) and isinstance(row.get(loc), list) else term_assign.get(loc) or []
        ulist = [str(u).strip() for u in (ul or []) if u is not None and str(u).strip()]
        dlist = [str(u).strip() for u in ((default_assign.get(loc) or []) if isinstance(default_assign, dict) else []) if u is not None and str(u).strip()]
        cap_val = _float(ts, "capacity", None) if isinstance(ts, dict) else None
        total_val = ts.get("total_sum") if isinstance(ts, dict) else None
        exc_val = _float(ts, "excess_as_percent_of_total", None) if isinstance(ts, dict) else None
        is_this_combo_dep = focus_terminal == loc and focus_dep_arr.upper() == "DEP"
        is_this_combo_arr = focus_terminal == loc and focus_dep_arr.upper() == "ARR"
        is_this_combo_both = focus_terminal == loc and focus_dep_arr.upper() == "BOTH"
        delay_val = _float(ts, "total_delay_hours_capacity_constraint", None) if isinstance(ts, dict) else None
        avg_seat_val = _float(ts, "avg_seat", None) if isinstance(ts, dict) else None
        st_dict = (sv.get("statistics") or {}) if isinstance(sv, dict) else {}
        both_97 = (_float(st_dict, "97.5%", None) if is_this_combo_both else None) or row.get(f"{loc}_Both_97.5%") if isinstance(row, dict) else None
        both_95 = (_float(st_dict, "95%", None) if is_this_combo_both else None) or row.get(f"{loc}_Both_95%") if isinstance(row, dict) else None
        by_location[loc] = {
            "Dep": {
                "97.5%": row.get(f"{loc}_Dep_97.5%") if isinstance(row, dict) else None,
                "95%": row.get(f"{loc}_Dep_95%") if isinstance(row, dict) else None,
                "capacity": cap_val if is_this_combo_dep else None,
                "total_sum": total_val if is_this_combo_dep else None,
                "excess_%": exc_val if is_this_combo_dep else None,
                "delay_hours": delay_val if is_this_combo_dep else None,
                "avg_seat": avg_seat_val if is_this_combo_dep else None,
                "units_list": ulist, "default_units_list": dlist,
            },
            "Arr": {
                "97.5%": row.get(f"{loc}_Arr_97.5%") if isinstance(row, dict) else None,
                "95%": row.get(f"{loc}_Arr_95%") if isinstance(row, dict) else None,
                "capacity": cap_val if is_this_combo_arr else None,
                "total_sum": total_val if is_this_combo_arr else None,
                "excess_%": exc_val if is_this_combo_arr else None,
                "delay_hours": delay_val if is_this_combo_arr else None,
                "avg_seat": avg_seat_val if is_this_combo_arr else None,
                "units_list": ulist, "default_units_list": dlist,
            },
            "Both": {
                "97.5%": both_97,
                "95%": both_95,
                "capacity": cap_val if is_this_combo_both else None,
                "total_sum": total_val if is_this_combo_both else None,
                "excess_%": exc_val if is_this_combo_both else None,
                "delay_hours": delay_val if is_this_combo_both else None,
                "avg_seat": avg_seat_val if is_this_combo_both else None,
                "units_list": ulist, "default_units_list": dlist,
            },
        }
    return {
        "id": meta.get("report_display_name") or meta.get("scenario_id", ""),
        "aggregate_base": meta.get("aggregate_base", ""), "date_range": meta.get("date_range", {}),
        "terminal_names": term_names,
        "focus_terminal": focus_terminal, "focus_dep_arr": focus_dep_arr, "dep_arr": focus_dep_arr,
        "value_97": value_97, "value_95": value_95,
        "by_location": by_location,
        "total_rank": row.get("TotalRank") if isinstance(row, dict) else None,
        "final_score": row.get("Final_Score") if isinstance(row, dict) else None,
        "units_list": units_list, "default_units_list": default_units_list,
        "summary_labels": summary_labels,
        "total_sum": ts.get("total_sum") if isinstance(ts, dict) else None,
        "capacity": ts.get("capacity") if isinstance(ts, dict) else None,
        "excess_over_capacity_sum": ts.get("excess_over_capacity_sum") if isinstance(ts, dict) else None,
        "excess_as_percent_of_total": ts.get("excess_as_percent_of_total") if isinstance(ts, dict) else None,
        "total_delay_hours_capacity_constraint": _float(ts, "total_delay_hours_capacity_constraint", None) if isinstance(ts, dict) else None,
        "design_criteria_value": sv.get("design_criteria_value") if isinstance(sv, dict) else None,
        "required_facilities": sv.get("required_facilities") if isinstance(sv, dict) else None,
        "capacity_per_unit": sv.get("capacity_per_unit") if isinstance(sv, dict) else None,
        "formula": (sv.get("formula") or "") if isinstance(sv, dict) else "",
        "statistics": (sv.get("statistics") or {}) if isinstance(sv, dict) else {},
        "daily_hourly_pivot": {"index": pivot_index, "columns": pivot_columns, "data": pivot_data},
        "stacked_pivot": {"index": stk_index, "columns": stk_columns, "data": stk_data},
        "stacked_pivot_group_col": (ts.get("stacked_pivot_group_col") or "") if isinstance(ts, dict) else "",
        "summary_by_group": sbg,
        "graph_data": (sv.get("graph_data") or {}) if isinstance(sv, dict) else {},
    }


def _get_dep_arr(rep, loc, dep_arr, key="97.5%"):
    """Get value from merged rep's by_location[loc][dep_arr][key]. rep is one merged scenario summary."""
    bl = rep.get("by_location") or {}
    loc_d = bl.get(loc) or {}
    da_d = loc_d.get(dep_arr) or {}
    return da_d.get(key)


def _merge_rep_from_group(group, sid):
    """Merge a group of summaries (same scenario_id) into one rep: id, terminal_names, by_location (all loc×Dep/Arr)."""
    if not group:
        return {"id": sid, "terminal_names": list(_FALLBACK_LOCATION_NAMES), "by_location": {}}
    first = group[0]
    tn = list(first.get("terminal_names") or _FALLBACK_LOCATION_NAMES)
    # by_location[loc][da] = { "97.5%", "95%", "capacity", "total_sum", "excess_%", "delay_hours", "avg_seat", "units_list", "default_units_list" }
    def _empty_da():
        return {"97.5%": None, "95%": None, "capacity": None, "total_sum": None, "excess_%": None, "delay_hours": None, "avg_seat": None, "units_list": [], "default_units_list": []}
    merged_bl = {}
    for loc in tn:
        merged_bl[loc] = {"Dep": _empty_da(), "Arr": _empty_da(), "Both": _empty_da()}
    for s in group:
        loc = (s.get("focus_terminal") or "").strip()
        da = (s.get("dep_arr") or s.get("focus_dep_arr") or "Dep").strip()
        if not loc or da not in ("Dep", "Arr", "Both"):
            continue
        if loc not in merged_bl:
            merged_bl[loc] = {"Dep": _empty_da(), "Arr": _empty_da(), "Both": _empty_da()}
        if da not in merged_bl[loc]:
            merged_bl[loc][da] = _empty_da()
        st_dict = s.get("statistics") or {}
        merged_bl[loc][da]["97.5%"] = s.get("value_97") if s.get("value_97") is not None else _float(st_dict, "97.5%", None)
        merged_bl[loc][da]["95%"] = s.get("value_95") if s.get("value_95") is not None else _float(st_dict, "95%", None)
        merged_bl[loc][da]["capacity"] = _float(s, "capacity", None)
        merged_bl[loc][da]["total_sum"] = s.get("total_sum")
        merged_bl[loc][da]["excess_%"] = _float(s, "excess_as_percent_of_total", None)
        merged_bl[loc][da]["delay_hours"] = _float(s, "total_delay_hours_capacity_constraint", None)
        bl_s = s.get("by_location") or {}
        loc_s = bl_s.get(loc) or {}
        da_s = loc_s.get(da) or {}
        merged_bl[loc][da]["avg_seat"] = da_s.get("avg_seat")
        merged_bl[loc][da]["units_list"] = da_s.get("units_list") or s.get("units_list") or []
        merged_bl[loc][da]["default_units_list"] = da_s.get("default_units_list") or s.get("default_units_list") or []
    # Fill 97.5%/95% from first summary's by_location (row) when merged is still None
    bl0 = first.get("by_location") or {}
    for loc in tn:
        for da in ("Dep", "Arr", "Both"):
            if da not in merged_bl.get(loc, {}):
                continue
            if merged_bl[loc][da]["97.5%"] is None and bl0.get(loc) and (bl0[loc].get(da) or {}).get("97.5%") is not None:
                merged_bl[loc][da]["97.5%"] = bl0[loc][da]["97.5%"]
            if merged_bl[loc][da]["95%"] is None and bl0.get(loc) and (bl0[loc].get(da) or {}).get("95%") is not None:
                merged_bl[loc][da]["95%"] = bl0[loc][da]["95%"]
    return {"id": sid, "terminal_names": tn, "by_location": merged_bl}


def _plotly_bar_html(scenario_ids, values, title, yaxis_title, colors=None):
    if not _HAS_PLOTLY or not scenario_ids or not values:
        return ""
    colors = colors or REPORT_SCENARIO_PALETTE
    x_labels = [str(s).replace("Scenario_", "S") for s in scenario_ids]
    fig = go.Figure(data=[go.Bar(x=x_labels, y=values, marker_color=[colors[i % len(colors)] for i in range(len(values))],
        text=[f"{v:,.0f}" if isinstance(v, (int, float)) and abs(v) >= 10 else f"{v:.2f}" for v in values],
        textposition="outside", textfont=dict(size=12))])
    fig.update_layout(title=dict(text=title, font=dict(size=16)), xaxis_title="Scenario", yaxis_title=yaxis_title,
        height=380, margin=dict(t=60, b=50, l=60, r=40), paper_bgcolor="rgba(30,41,59,0.5)", plot_bgcolor="rgba(30,41,59,0.3)",
        font=dict(color="#e2e8f0", size=12), xaxis=dict(gridcolor="rgba(51,65,85,0.5)", tickfont=dict(size=11)),
        yaxis=dict(gridcolor="rgba(51,65,85,0.5)"), showlegend=False)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": True, "responsive": True})


def _combo_pattern(ci):
    """Separate bars by combination order. Dep/Arr/Both × several location Expansion of patterns for distinction."""
    patterns = [
        {},  # 0: solid
        {"marker_pattern": dict(shape="/", solidity=0.35, fgopacity=0.9, size=12)},   # 1: slash
        {"marker_pattern": dict(shape=".", solidity=0.6, fgopacity=0.9, size=6)},     # 2: dots
        {"marker_pattern": dict(shape="\\", solidity=0.35, fgopacity=0.9, size=12)},  # 3: backslash
        {"marker_pattern": dict(shape="x", solidity=0.5, fgopacity=0.85, size=10)},   # 4: x
        {"marker_pattern": dict(shape="-", solidity=0.5, fgopacity=0.9, size=8)},     # 5: horizontal
        {"marker_pattern": dict(shape="|", solidity=0.5, fgopacity=0.9, size=8)},     # 6: vertical
        {"marker_pattern": dict(shape="+", solidity=0.5, fgopacity=0.85, size=10)},   # 7: plus
    ]
    return patterns[ci % len(patterns)]


def _plotly_excess_grouped_bar_html(scenario_ids, combos, get_excess_fn, colors=None):
    """terminal×% excess capacity by departure/arrival combination. Same color for each scenario, combination is diagonal/dot/Separated by transparency."""
    if not _HAS_PLOTLY or not scenario_ids or not combos:
        return ""
    colors = colors or REPORT_SCENARIO_PALETTE
    x_labels = [str(s).replace("Scenario_", "S") for s in scenario_ids]
    scenario_colors = [colors[i % len(colors)] for i in range(len(scenario_ids))]
    fig = go.Figure()
    for ci, (term, dep_arr) in enumerate(combos):
        vals = [(get_excess_fn(sid, term, dep_arr) or 0) for sid in scenario_ids]
        name = f"{term} {dep_arr}"
        extra = _combo_pattern(ci)
        fig.add_trace(go.Bar(name=name, x=x_labels, y=vals, marker_color=scenario_colors,
            text=[f"{v:.2f}%" for v in vals], textposition="outside", textfont=dict(size=10), **extra))
    fig.update_layout(title=dict(text="", font=dict(size=16)),
        xaxis_title="Scenario", yaxis_title="Excess as % of total", barmode="group",
        height=400, margin=dict(t=60, b=50, l=60, r=40), paper_bgcolor="rgba(30,41,59,0.5)", plot_bgcolor="rgba(30,41,59,0.3)",
        font=dict(color="#e2e8f0", size=12), xaxis=dict(gridcolor="rgba(51,65,85,0.5)"), yaxis=dict(gridcolor="rgba(51,65,85,0.5)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": True, "responsive": True})


def _plotly_delay_hours_grouped_bar_html(scenario_ids, combos, get_delay_fn, colors=None, title=None, yaxis_title=None, text_fmt=None):
    """terminal×By departure/arrival combination Delay Hours. Same color for each scenario, combination is diagonal/dot/Separated by transparency."""
    if not _HAS_PLOTLY or not scenario_ids or not combos:
        return ""
    colors = colors or REPORT_SCENARIO_PALETTE
    x_labels = [str(s).replace("Scenario_", "S") for s in scenario_ids]
    scenario_colors = [colors[i % len(colors)] for i in range(len(scenario_ids))]
    _title = title if title is not None else ""
    _yaxis = yaxis_title or "Delay (hours)"
    _text_fmt = text_fmt or (lambda v: _fmt_delay_hours(v, 1))
    fig = go.Figure()
    for ci, (term, dep_arr) in enumerate(combos):
        vals = [(get_delay_fn(sid, term, dep_arr) or 0) for sid in scenario_ids]
        name = f"{term} {dep_arr}"
        extra = _combo_pattern(ci)
        fig.add_trace(go.Bar(name=name, x=x_labels, y=vals, marker_color=scenario_colors,
            text=[_text_fmt(v) for v in vals], textposition="outside", textfont=dict(size=10), **extra))
    fig.update_layout(title=dict(text=_title, font=dict(size=16)),
        xaxis_title="Scenario", yaxis_title=_yaxis, barmode="group",
        height=400, margin=dict(t=60, b=50, l=60, r=40), paper_bgcolor="rgba(30,41,59,0.5)", plot_bgcolor="rgba(30,41,59,0.3)",
        font=dict(color="#e2e8f0", size=12), xaxis=dict(gridcolor="rgba(51,65,85,0.5)"), yaxis=dict(gridcolor="rgba(51,65,85,0.5)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": True, "responsive": True})


def _plotly_dep_arr_single_grouped_bar_html(scenario_ids, rep_summaries, report_labels, colors=None, percentile="97.5%", capacity_per_series=None, aggregate_base=None, combos=None):
    """depart(Dep) + arrive(Arr) Peaks to Group Bars. combosIf present, only that combination is displayed(Both Excluded if not selected). capacity_per_series: Dep/Arr/(Both) per loc."""
    if not _HAS_PLOTLY or not scenario_ids or not rep_summaries:
        return ""
    colors = colors or REPORT_SCENARIO_PALETTE
    terminal_names = report_labels.get("terminal_names") or []
    if not terminal_names:
        return ""
    y_unit = _yaxis_unit(aggregate_base)
    x_labels = [str(s).replace("Scenario_", "S") for s in scenario_ids]
    n = len(scenario_ids)
    scenario_colors = [colors[i % len(colors)] for i in range(n)]
    key = "95%" if percentile == "95%" else "97.5%"
    label_suffix = "95%" if percentile == "95%" else "97.5%"
    # combosIf there is actual exportUse only combinations (Both When not selected Both exception). If there is no Dep/Arronly basic
    if combos:
        series_order = list(combos)
    else:
        series_order = [(loc, da) for da in ("Dep", "Arr") for loc in terminal_names]
    if not series_order:
        return ""
    fig = go.Figure()
    for si, (loc, da) in enumerate(series_order):
        vals = [float(_get_dep_arr(rep, loc, da, key) or 0) for rep in rep_summaries]
        extra = _combo_pattern(si)
        fig.add_trace(go.Bar(name=f"{loc} {da} {label_suffix}", x=x_labels, y=vals, marker_color=scenario_colors,
            text=[f"{v:,.0f}" for v in vals], textposition="outside", textfont=dict(size=10), **extra))

    # Capacity: each bar(series)Line position matching with the same hierarchy as.
    # capacity_per_series order: Dep(loc0..), Arr(loc0..), Both(loc0..) | series_order: Dep(all locs), Arr(all locs), Both(all locs)
    shapes = []
    n_series = len(series_order)
    n_locs = len(terminal_names)
    if capacity_per_series and len(capacity_per_series) >= n_series and len(capacity_per_series[0]) == n:
        bar_width = 0.8 / max(n_series, 1)
        for si, (loc, da) in enumerate(series_order):
            loc_idx = terminal_names.index(loc) if loc in terminal_names else si % n_locs
            da_idx = ("Dep", "Arr", "Both").index(da) if da in ("Dep", "Arr", "Both") else si // n_locs
            cap_index = da_idx * n_locs + loc_idx
            if cap_index >= len(capacity_per_series):
                continue
            cap_list = capacity_per_series[cap_index]
            x0_off = -0.4 + si * bar_width
            x1_off = x0_off + bar_width
            for i in range(n):
                v = cap_list[i]
                if v is not None and isinstance(v, (int, float)) and v == v:
                    shapes.append(dict(
                        type="line",
                        x0=i + x0_off, x1=i + x1_off, y0=float(v), y1=float(v),
                        xref="x", yref="y",
                        line=dict(color="rgba(239,68,68,0.9)", width=1.5, dash="dash")
                    ))

    fig.update_layout(
        title=dict(text="", font=dict(size=16)),
        xaxis_title="Scenario", yaxis_title=y_unit, barmode="group",
        bargap=0.2, bargroupgap=0,
        height=420, margin=dict(t=60, b=50, l=60, r=40), paper_bgcolor="rgba(30,41,59,0.5)", plot_bgcolor="rgba(30,41,59,0.3)",
        font=dict(color="#e2e8f0", size=12), xaxis=dict(gridcolor="rgba(51,65,85,0.5)"), yaxis=dict(gridcolor="rgba(51,65,85,0.5)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#cbd5e1")),
        shapes=shapes
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": True, "responsive": True})


def _build_comparison_bar_charts_plotly(id_list, rep_summaries, report_labels, colors):
    if not id_list or not rep_summaries:
        return {"excess": ""}
    def _one_excess(rep):
        v = rep.get("excess_as_percent_of_total")
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
        bl = rep.get("by_location") or {}
        vals = []
        for loc_d in bl.values():
            for da_d in (loc_d.get("Dep") or {}, loc_d.get("Arr") or {}):
                e = da_d.get("excess_%")
                if e is not None:
                    try:
                        vals.append(float(e))
                    except (TypeError, ValueError):
                        pass
        return max(vals) if vals else 0
    excess_pcts = [_one_excess(s) for s in rep_summaries]
    excess_html = _plotly_bar_html(id_list, excess_pcts, "Excess as % of total", "Excess as % of total", colors) if excess_pcts else ""
    return {"excess": excess_html}


def _build_capacity_per_series(rep_summaries, terminal_names, agg, include_both=False):
    """Build capacity_per_series for charts: list of [Dep per loc, Arr per loc, (optional) Both per loc].
    Each element is a list of capacity values, one per scenario (same order as rep_summaries).
    When agg is 'movement' and value is None, use 10 as default."""
    if not rep_summaries or not terminal_names:
        return None
    n = len(rep_summaries)
    is_movement = (agg or "").strip().lower() == "movement"
    cap_lists = []
    for loc in terminal_names:
        dep_caps = []
        for i in range(n):
            cap_val = _get_dep_arr(rep_summaries[i] if i < len(rep_summaries) else {}, loc, "Dep", "capacity")
            if cap_val is None and is_movement:
                cap_val = 10
            dep_caps.append(cap_val)
        cap_lists.append(dep_caps)
    for loc in terminal_names:
        arr_caps = []
        for i in range(n):
            cap_val = _get_dep_arr(rep_summaries[i] if i < len(rep_summaries) else {}, loc, "Arr", "capacity")
            if cap_val is None and is_movement:
                cap_val = 10
            arr_caps.append(cap_val)
        cap_lists.append(arr_caps)
    if include_both:
        for loc in terminal_names:
            both_caps = []
            for i in range(n):
                rep = rep_summaries[i] if i < len(rep_summaries) else {}
                cap_val = _get_dep_arr(rep, loc, "Both", "capacity")
                if cap_val is None:
                    dep_cap = _get_dep_arr(rep, loc, "Dep", "capacity")
                    arr_cap = _get_dep_arr(rep, loc, "Arr", "capacity")
                    if is_movement and dep_cap is None and arr_cap is None:
                        cap_val = 10
                    else:
                        cap_val = (dep_cap or arr_cap) if (dep_cap is None or arr_cap is None) else ((dep_cap + arr_cap) / 2)
                both_caps.append(cap_val)
            cap_lists.append(both_caps)
    return cap_lists if any(cap_lists) else None


def _format_agg_for_axis(aggregate_base):
    """aggregate_base value Yaxis/An easy-to-read string for the title. yes: total_seat_count -> 'seat', total_pax -> 'pax'."""
    if not aggregate_base or not isinstance(aggregate_base, str):
        return "value"
    agg = aggregate_base.strip().lower()
    if agg == "total_seat_count":
        return "seat"
    if agg == "total_pax":
        return "pax"
    if agg == "movement":
        return "movement"
    return agg.replace("_", " ").strip() or "value"


def _yaxis_unit(aggregate_base):
    """chart Yaxis/title unit. movement -> movement/hour, Others seat -> seat/hour."""
    if not aggregate_base or not isinstance(aggregate_base, str):
        return "seat/hour"
    agg = aggregate_base.strip().lower()
    if agg == "movement":
        return "movement/hour"
    return "seat/hour"


def _total_metric_label(aggregate_base):
    """Scenario summary For table header: Countunification."""
    return "Count"


def _plotly_timeseries_both_by_terminal_html(scenarios, unique_ids, colors, agg_filter=None):
    """Both Exclusive: Scenario Details Cumulative profileused in time_series(time_index, values)picture as is. agg_filter corresponding aggonly. Returns [(title, html), ...]."""
    if not _HAS_PLOTLY or not scenarios or not unique_ids:
        return []
    from collections import defaultdict
    by_terminal = defaultdict(list)
    for s in scenarios:
        if not isinstance(s, dict):
            continue
        meta = s.get("meta") or {}
        if (meta.get("dep_arr") or "").strip().upper() != "BOTH":
            continue
        agg = (meta.get("aggregate_base") or "").strip() or None
        if agg_filter is not None and agg_filter != agg:
            continue
        ts = s.get("time_series")
        if not ts or not isinstance(ts, dict):
            continue
        time_index = ts.get("time_index")
        values = ts.get("values")
        if not time_index or values is None:
            continue
        sid = str(meta.get("scenario_id", "") or "").strip() or "Scenario"
        term = str(meta.get("terminal", "") or "").strip() or "Location"
        by_terminal[term].append((sid, time_index, values, agg))
    if not by_terminal:
        return []
    colors = colors or REPORT_SCENARIO_PALETTE
    id_to_idx = {str(u).strip(): i for i, u in enumerate(unique_ids)}
    id_to_idx.update({str(u).replace("Scenario_", "S"): i for i, u in enumerate(unique_ids)})
    out = []
    for term in sorted(by_terminal.keys()):
        group = by_terminal[term]
        agg_label = None
        for _sid, _ti, _v, _agg in group:
            if _agg:
                agg_label = _format_agg_for_axis(_agg)
                break
        if not agg_label:
            agg_label = "value"
        fig = go.Figure()
        for sid, time_index, values, _ in group:
            idx = id_to_idx.get(sid, id_to_idx.get(sid.replace("Scenario_", "S"), 0))
            c = colors[idx % len(colors)]
            x = list(time_index) if isinstance(time_index, (list, tuple)) else time_index
            y = list(values) if isinstance(values, (list, tuple)) else (list(values) if hasattr(values, "__iter__") and not isinstance(values, (str, bytes)) else [])
            if len(y) != len(x):
                continue
            y = [float(v) if v is not None else None for v in y]
            fig.add_trace(go.Scatter(
                x=x, y=y, name=sid.replace("Scenario_", "S"),
                line=dict(color=c, width=2), mode="lines",
            ))
        if len(fig.data) == 0:
            continue
        y_unit = _yaxis_unit(group[0][3]) if group and group[0][3] else "value"
        fig.update_layout(
            title=dict(text=f"Time series (Both) · {term} · {agg_label}", font=dict(size=16)),
            xaxis_title="Time",
            yaxis_title=y_unit,
            height=480,
            margin=dict(t=60, b=50, l=60, r=40),
            paper_bgcolor="rgba(30,41,59,0.5)",
            plot_bgcolor="rgba(30,41,59,0.3)",
            font=dict(color="#e2e8f0", size=12),
            xaxis=dict(gridcolor="rgba(51,65,85,0.5)"),
            yaxis=dict(gridcolor="rgba(51,65,85,0.5)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#cbd5e1")),
        )
        out.append((f"{term} · Both ({agg_label})", fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": True, "responsive": True})))
    return out


def _plotly_cumulative_by_terminal_html(scenarios, unique_ids, colors, agg_filter=None, series_type=None):
    """By terminal cumulative. agg_filter: corresponding aggonly; series_type: 'Dep'|'Arr'|'Both' just one of. NoneEverything behind it. Returns [(title, html), ...]."""
    if not _HAS_PLOTLY or not scenarios or not unique_ids:
        return []
    from collections import defaultdict
    by_terminal = defaultdict(list)
    for s in scenarios:
        if not isinstance(s, dict):
            continue
        meta = s.get("meta") or {}
        agg = (meta.get("aggregate_base") or "").strip() or None
        if agg_filter is not None and agg_filter != agg:
            continue
        # cumulative: Relocation_Masterat cumsum 1After counting times JSONThe value contained in. Reportis used as is without calculating twice.
        cum = s.get("cumulative")
        if not cum or not isinstance(cum, dict):
            continue
        time_index = cum.get("time_index")
        dep = cum.get("cumulative_departure")
        arr = cum.get("cumulative_arrival")
        net = cum.get("net_cumulative")
        if not time_index or (not dep and not arr and not net):
            continue
        dep_arr = (meta.get("dep_arr") or "").strip() or "Dep"
        sid = str(meta.get("scenario_id", "") or "").strip() or "Scenario"
        term = str(meta.get("terminal", "") or "").strip() or "Location"
        by_terminal[term].append((sid, time_index, dep, arr, net, agg, dep_arr))
    if not by_terminal:
        return []
    colors = colors or REPORT_SCENARIO_PALETTE
    id_to_idx = {str(u).strip(): i for i, u in enumerate(unique_ids)}
    id_to_idx.update({str(u).replace("Scenario_", "S"): i for i, u in enumerate(unique_ids)})
    show_dep = series_type is None or series_type == "Dep"
    show_arr = series_type is None or series_type == "Arr"
    show_both = series_type is None or series_type == "Both"
    out = []
    for term in sorted(by_terminal.keys()):
        group = by_terminal[term]
        agg_label = None
        for _sid, _ti, _d, _a, _n, _agg, _da in group:
            if _agg:
                agg_label = _format_agg_for_axis(_agg)
                break
        if not agg_label:
            agg_label = "value"
        y_title = f"Cumulative ({agg_label})"
        fig = go.Figure()
        # Bothto draw the same scenario IDof Depand Arrmatching
        dep_by_sid = {}
        arr_by_sid = {}
        net_by_sid = {}
        for sid, time_index, dep, arr, net, _, dep_arr in group:
            if dep_arr.upper() == "DEP" and dep is not None:
                dep_by_sid[sid] = (time_index, dep)
            elif dep_arr.upper() == "ARR" and arr is not None:
                arr_by_sid[sid] = (time_index, arr)
            elif dep_arr.upper() == "BOTH" and net is not None:
                net_by_sid[sid] = (time_index, net)
        # Dep drawing series
        if show_dep and series_type != "Both":
            for sid, (time_index, dep) in dep_by_sid.items():
                idx = id_to_idx.get(sid, 0)
                c = colors[idx % len(colors)]
                try:
                    x = list(time_index) if not isinstance(time_index, list) else time_index
                except (TypeError, ValueError):
                    continue
                y_dep = list(dep) if isinstance(dep, (list, tuple)) else (list(dep) if hasattr(dep, "__iter__") and not isinstance(dep, (str, bytes)) else [])
                if len(y_dep) == len(x):
                    y_dep = [float(v) if v is not None else None for v in y_dep]
                    fig.add_trace(go.Scatter(
                        x=x, y=y_dep, name=f"{sid} Dep",
                        line=dict(color=c, width=2, dash="solid"),
                        mode="lines",
                    ))
        # Arr drawing series
        if show_arr and series_type != "Both":
            for sid, (time_index, arr) in arr_by_sid.items():
                idx = id_to_idx.get(sid, 0)
                c = colors[idx % len(colors)]
                try:
                    x = list(time_index) if not isinstance(time_index, list) else time_index
                except (TypeError, ValueError):
                    continue
                y_arr = list(arr) if isinstance(arr, (list, tuple)) else (list(arr) if hasattr(arr, "__iter__") and not isinstance(arr, (str, bytes)) else [])
                if len(y_arr) == len(x):
                    y_arr = [float(v) if v is not None else None for v in y_arr]
                    fig.add_trace(go.Scatter(
                        x=x, y=y_arr, name=f"{sid} Arr",
                        line=dict(color=c, width=2, dash="dash"),
                        mode="lines",
                    ))
        # Both drawing series: Relocation MasterSame as - only 1 line per scenario. Ndog scenario = Nimprovement
        if show_both:
            # series_type=="Both"When: net_by_siduse only (Dep+Arr No matching). unique_ids Only 1 in each order
            if series_type == "Both":
                def _sid_match(a, b):
                    sa, sb = str(a).strip(), str(b).strip()
                    return sa == sb or sa.replace("Scenario_", "S") == sb.replace("Scenario_", "S")
                for sid in unique_ids:
                    match_key = None
                    for k in net_by_sid:
                        if _sid_match(k, sid):
                            match_key = k
                            break
                    if match_key is None:
                        continue
                    time_index, net = net_by_sid[match_key]
                    idx = id_to_idx.get(match_key, id_to_idx.get(str(sid).strip(), 0))
                    c = colors[idx % len(colors)]
                    try:
                        x = list(time_index) if not isinstance(time_index, list) else time_index
                    except (TypeError, ValueError):
                        continue
                    y_net = list(net) if isinstance(net, (list, tuple)) else (list(net) if hasattr(net, "__iter__") and not isinstance(net, (str, bytes)) else [])
                    if len(y_net) == len(x):
                        y_net = [float(v) if v is not None else None for v in y_net]
                        vals = [v for v in y_net if v is not None]
                        offset = -min(vals) if vals and min(vals) < 0 else 0.0
                        y_adj = [v + offset if v is not None else None for v in y_net]
                        label = str(match_key).replace("Scenario_", "S")
                        fig.add_trace(go.Scatter(
                            x=x, y=y_adj, name=label,
                            line=dict(color=c, width=2),
                            mode="lines",
                        ))
            else:
                for sid, (time_index, net) in net_by_sid.items():
                    idx = id_to_idx.get(sid, 0)
                    c = colors[idx % len(colors)]
                    try:
                        x = list(time_index) if not isinstance(time_index, list) else time_index
                    except (TypeError, ValueError):
                        continue
                    y_net = list(net) if isinstance(net, (list, tuple)) else (list(net) if hasattr(net, "__iter__") and not isinstance(net, (str, bytes)) else [])
                    if len(y_net) == len(x):
                        y_net = [float(v) if v is not None else None for v in y_net]
                        vals = [v for v in y_net if v is not None]
                        offset = -min(vals) if vals and min(vals) < 0 else 0.0
                        y_adj = [v + offset if v is not None else None for v in y_net]
                        fig.add_trace(go.Scatter(
                            x=x, y=y_adj, name=f"{sid} Net",
                            line=dict(color=c, width=2, dash="dot"),
                            mode="lines",
                        ))
        if len(fig.data) == 0:
            continue
        type_label = series_type if series_type else "Dep / Arr / Both"
        fig.update_layout(
            title=dict(text=f"Cumulative · {term} · {type_label} · {agg_label}", font=dict(size=16)),
            xaxis_title="Time",
            yaxis_title=y_title,
            height=480,
            margin=dict(t=60, b=50, l=60, r=40),
            paper_bgcolor="rgba(30,41,59,0.5)",
            plot_bgcolor="rgba(30,41,59,0.3)",
            font=dict(color="#e2e8f0", size=12),
            xaxis=dict(gridcolor="rgba(51,65,85,0.5)"),
            yaxis=dict(gridcolor="rgba(51,65,85,0.5)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#cbd5e1")),
            hovermode="x unified",
        )
        out.append((f"{term} · {type_label} ({agg_label})", fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": True, "responsive": True})))
    return out


def _plotly_radial_dep_arr_97_html(statistics_table, report_labels, unique_ids, rep_summaries, colors, capacity_per_series=None, aggregate_base=None):
    """Radar Chart: Departures by Location/arrive 97.5%. Ndog location support. Capacityis the red dotted line. aggregate_baseIf present, the table column name is {loc}_Dep/Arr_97.5%_{aggregate_base} interpreted in format."""
    if not _HAS_PLOTLY:
        return ""
    terminal_names = report_labels.get("terminal_names") or []
    if not terminal_names:
        return ""
    # aggFor star tables, the column names are {loc}_Dep_97.5%_{aggregate_base} may be of the form
    first_row = (statistics_table or [{}])[0] if statistics_table and len(statistics_table) > 0 else {}
    first_keys = set(first_row.keys()) if isinstance(first_row, dict) else set()
    def _col_raw(loc, dep_arr):
        base = f"{loc}_{dep_arr}_97.5%"
        if aggregate_base and (f"{base}_{aggregate_base}" in first_keys):
            return f"{base}_{aggregate_base}"
        return base
    cols_raw = [_col_raw(loc, "Dep") for loc in terminal_names] + [_col_raw(loc, "Arr") for loc in terminal_names]
    theta_labels = [f"{loc} Dep 97.5%" for loc in terminal_names] + [f"{loc} Arr 97.5%" for loc in terminal_names]
    colors = colors or REPORT_SCENARIO_PALETTE

    id_set = set()
    if unique_ids:
        for u in unique_ids:
            s = str(u).strip()
            if s:
                id_set.add(s)
                id_set.add(s.replace("Scenario_", "S"))

    # statistics_tablecorresponding to aggAll scenarios of(selected + others) collection — seat/movement etc. agg commonness
    rows_for_radial = []
    if statistics_table and isinstance(statistics_table, list) and len(statistics_table) > 0:
        for row in statistics_table:
            if not isinstance(row, dict):
                continue
            sid = row.get("Scenario_ID") or row.get("scenario_id") or row.get("index")
            if sid is None:
                continue
            sid_str = str(sid).strip().replace("Scenario_", "S")
            vals = []
            for c in cols_raw:
                v = row.get(c)
                try:
                    vals.append(float(v) if v is not None and v != "" else 0)
                except (TypeError, ValueError):
                    vals.append(0)
            is_selected = bool(id_set and (sid_str in id_set or sid_str.replace("Scenario_", "S") in id_set))
            rows_for_radial.append({"scenario_id": sid_str, "values": vals, "is_selected": is_selected})
        if unique_ids and rows_for_radial:
            order_map = {str(u).replace("Scenario_", "S"): i for i, u in enumerate(unique_ids)}
            def _sort_key(r):
                if r.get("is_selected"):
                    return (0, order_map.get(r["scenario_id"], 999))
                return (1, 0)
            rows_for_radial.sort(key=_sort_key)
            # Radar: Show only selected scenarios (other exception)
            rows_for_radial = [r for r in rows_for_radial if r.get("is_selected")]
    if not rows_for_radial and unique_ids and rep_summaries:
        series_order = [(loc, "Dep") for loc in terminal_names] + [(loc, "Arr") for loc in terminal_names]
        for i, sid in enumerate(unique_ids):
            rep = rep_summaries[i] if i < len(rep_summaries) else {}
            vals = [float(_get_dep_arr(rep, loc, da, "97.5%") or 0) for loc, da in series_order]
            rows_for_radial.append({"scenario_id": str(sid).replace("Scenario_", "S"), "values": vals})

    if not rows_for_radial:
        return ""

    order_map = {str(u).replace("Scenario_", "S"): i for i, u in enumerate(unique_ids)} if unique_ids else {}
    all_vals = [v for r in rows_for_radial for v in r["values"]]
    v_min = min(all_vals) if all_vals else 0
    v_max = max(all_vals) if all_vals else 1
    if v_max <= v_min:
        v_max = v_min + 1
    denom = v_max - v_min if (v_max - v_min) != 0 else 1

    fig = go.Figure()
    theta_close = theta_labels + [theta_labels[0]]
    for i, r in enumerate(rows_for_radial):
        vals = r["values"]
        r_close = vals + [vals[0]]
        r_norm = [((float(x) if x is not None else 0) - v_min) / denom for x in r_close]
        color_idx = order_map.get(r["scenario_id"], i)
        scenario_color = colors[color_idx % len(colors)]
        fig.add_trace(go.Scatterpolar(
            r=r_norm,
            theta=theta_close,
            mode="lines+markers",
            name=r["scenario_id"],
            legendgroup=r["scenario_id"],
            showlegend=True,
            line=dict(color=scenario_color, width=2),
            marker=dict(size=8),
        ))
    cap_vals = None
    n_ax = len(theta_labels)
    if capacity_per_series and len(capacity_per_series) >= n_ax and len(capacity_per_series[0]) > 0:
        cap_vals = []
        for i in range(n_ax):
            c = capacity_per_series[i][0]
            if c is not None and isinstance(c, (int, float)) and c == c:
                cap_vals.append(float(c))
            else:
                cap_vals = None
                break
        if cap_vals is not None:
            cap_close = cap_vals + [cap_vals[0]]
            cap_norm = [((float(x) if x is not None else 0) - v_min) / denom for x in cap_close]
            fig.add_trace(go.Scatterpolar(
                r=cap_norm,
                theta=theta_close,
                mode="lines",
                name="Capacity",
                line=dict(color="rgba(239,68,68,0.9)", width=1.5, dash="dash"),
            ))
    fig.update_layout(
        title=dict(text="Radar comparison (by location Dep/Arr 97.5%) · Statistics table", font=dict(size=16)),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10), gridcolor="rgba(51,65,85,0.5)"),
        angularaxis=dict(gridcolor="rgba(51,65,85,0.5)", linecolor="rgba(51,65,85,0.7)"),
            bgcolor="rgba(30,41,59,0.3)",
        ),
        height=480,
        margin=dict(t=60, b=50, l=60, r=60),
        paper_bgcolor="rgba(30,41,59,0.5)",
        font=dict(color="#e2e8f0", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": True, "responsive": True})


# Light gray, more transparent for "other" scenarios so selected ones stand out
_SCATTER_OTHER_COLOR = "rgba(148, 163, 184, 0.22)"


def _plotly_dep_arr_scatter_by_terminal_html(statistics_table, report_labels, unique_ids, rep_summaries, colors, capacity_per_series=None, aggregate_base=None):
    """Per-location scatter: 97.5% Dep (x) vs 97.5% Arr (y). Ndog location support. aggregate_baseDepending on the axis unit(seat/hour vs movement/hour) apply. Returns [(title, html), ...]."""
    if not _HAS_PLOTLY:
        return []
    terminal_names = report_labels.get("terminal_names") or []
    if not terminal_names:
        return []
    colors = colors or REPORT_SCENARIO_PALETTE
    id_set = set()
    id_to_idx = {}
    if unique_ids:
        for i, u in enumerate(unique_ids):
            s = str(u).strip()
            if s:
                id_set.add(s)
                id_set.add(s.replace("Scenario_", "S"))
                id_to_idx[s] = i
                id_to_idx[s.replace("Scenario_", "S")] = i

    out = []
    # capacity_per_series: [loc0 Dep, loc1 Dep, ...], [loc0 Arr, loc1 Arr, ...]
    n_loc = len(terminal_names)
    for ti, term in enumerate(terminal_names):
        dep_col = f"{term}_Dep_97.5%"
        arr_col = f"{term}_Arr_97.5%"
        if aggregate_base and statistics_table and len(statistics_table) > 0 and isinstance(statistics_table[0], dict):
            suffixed_dep = f"{term}_Dep_97.5%_{aggregate_base}"
            suffixed_arr = f"{term}_Arr_97.5%_{aggregate_base}"
            if suffixed_dep in statistics_table[0] and suffixed_arr in statistics_table[0]:
                dep_col, arr_col = suffixed_dep, suffixed_arr
        points = []
        selected_points_from_reps = []
        # selected scenario(unique_ids)Is rep_summariesFirst sought from — custom display name(S35_Y2026 etc.) When used statistics_tableof Scenario_IDIn disagreement with OtherAvoid problems classified as
        if unique_ids and rep_summaries:
            for i, sid in enumerate(unique_ids):
                rep = rep_summaries[i] if i < len(rep_summaries) else {}
                x_f = _get_dep_arr(rep, term, "Dep", "97.5%")
                y_f = _get_dep_arr(rep, term, "Arr", "97.5%")
                if x_f is not None and y_f is not None:
                    try:
                        x_f, y_f = float(x_f), float(y_f)
                    except (TypeError, ValueError):
                        continue
                    sid_str = str(sid).strip()
                    selected_points_from_reps.append({"scenario_id": sid_str, "x": x_f, "y": y_f, "idx": i})
        # statistics_tableat othersGet Dragon Points (Excluding selected scenarios)
        if statistics_table and isinstance(statistics_table, list):
            for row in statistics_table:
                if not isinstance(row, dict):
                    continue
                sid = row.get("Scenario_ID") or row.get("scenario_id") or row.get("index")
                if sid is None:
                    continue
                sid_str = str(sid).strip().replace("Scenario_", "S")
                if sid_str in id_set:
                    continue  # rep_summariesExcludes selection scenarios that have already been processed in
                x_val = row.get(dep_col)
                y_val = row.get(arr_col)
                try:
                    x_f = float(x_val) if x_val is not None and x_val != "" else None
                    y_f = float(y_val) if y_val is not None and y_val != "" else None
                except (TypeError, ValueError):
                    x_f, y_f = None, None
                if x_f is not None and y_f is not None:
                    points.append({"scenario_id": sid_str, "x": x_f, "y": y_f})
        # rep_summariesIf there is a point selected in , use it. If not, use the existing point. statistics_table based classification
        if selected_points_from_reps:
            selected_points = [(p, p["scenario_id"], p["idx"]) for p in selected_points_from_reps]
            gray_x = [p["x"] for p in points]
            gray_y = [p["y"] for p in points]
            gray_text = [p["scenario_id"] for p in points]
            points = selected_points_from_reps + points  # zone/For calculating axis range
        else:
            # rep_summaries fallback When there is no: statistics_table the whole pointsClassified after use
            if not points and statistics_table and isinstance(statistics_table, list):
                for row in statistics_table:
                    if not isinstance(row, dict):
                        continue
                    sid = row.get("Scenario_ID") or row.get("scenario_id") or row.get("index")
                    if sid is None:
                        continue
                    sid_str = str(sid).strip().replace("Scenario_", "S")
                    x_val = row.get(dep_col)
                    y_val = row.get(arr_col)
                    try:
                        x_f = float(x_val) if x_val is not None and x_val != "" else None
                        y_f = float(y_val) if y_val is not None and y_val != "" else None
                    except (TypeError, ValueError):
                        x_f, y_f = None, None
                    if x_f is not None and y_f is not None:
                        points.append({"scenario_id": sid_str, "x": x_f, "y": y_f})
            if not points:
                continue
            gray_x, gray_y, gray_text = [], [], []
            selected_points = []
            for p in points:
                sid_str = p["scenario_id"]
                idx = id_to_idx.get(sid_str)
                if idx is not None:
                    selected_points.append((p, sid_str, idx))
                else:
                    gray_x.append(p["x"])
                    gray_y.append(p["y"])
                    gray_text.append(sid_str)
        # Build background (zones + capacity lines) first so scatter is drawn on top and stays visible
        shapes = []
        if capacity_per_series and len(capacity_per_series) >= n_loc * 2 and len(capacity_per_series[0]) > 0:
            dep_idx = ti
            arr_idx = n_loc + ti
            dep_cap = capacity_per_series[dep_idx][0]
            arr_cap = capacity_per_series[arr_idx][0]
            try:
                dep_cap = float(dep_cap) if dep_cap is not None else None
                arr_cap = float(arr_cap) if arr_cap is not None else None
            except (TypeError, ValueError):
                dep_cap, arr_cap = None, None
            if dep_cap is not None and arr_cap is not None:
                all_x = [p["x"] for p in points]
                all_y = [p["y"] for p in points]
                x_max = max(all_x) if all_x else dep_cap * 2
                y_max = max(all_y) if all_y else arr_cap * 2
                # by magnification x/y Boundary calculation (0 = 0, 1 = cap, Others = cap*ratio; 'max' = x_max/y_max)
                def _x(v):
                    return 0 if v == 0 else (x_max if v == "max" else dep_cap * v)
                def _y(v):
                    return 0 if v == 0 else (y_max if v == "max" else arr_cap * v)
                # Zone rectangles: (x0_ratio, x1_ratio, y0_ratio, y1_ratio, fillcolor)
                zone_rects = [
                    # Red: outside 0.3~1.7
                    (0, 0.3, 0, "max", "rgba(239, 68, 68, 0.22)"),
                    (1.7, "max", 0, "max", "rgba(239, 68, 68, 0.22)"),
                    (0.3, 1.7, 0, 0.3, "rgba(239, 68, 68, 0.22)"),
                    (0.3, 1.7, 1.7, "max", "rgba(239, 68, 68, 0.22)"),
                    # Orange: L-shape around green (0.3~1.7 band)
                    (0.3, 0.5, 0.3, 1.7, "rgba(251, 146, 60, 0.28)"),
                    (1.5, 1.7, 0.3, 1.7, "rgba(251, 146, 60, 0.28)"),
                    (0.5, 1.5, 0.3, 0.5, "rgba(251, 146, 60, 0.28)"),
                    (0.5, 1.5, 1.5, 1.7, "rgba(251, 146, 60, 0.28)"),
                    # Green (0.5~1.5), Bright green (0.7~1.3)
                    (0.5, 1.5, 0.5, 1.5, "rgba(34, 197, 94, 0.28)"),
                    (0.7, 1.3, 0.7, 1.3, "rgba(74, 222, 128, 0.4)"),
                ]
                for x0r, x1r, y0r, y1r, fill in zone_rects:
                    shapes.append(dict(
                        type="rect",
                        x0=_x(x0r), x1=_x(x1r), y0=_y(y0r), y1=_y(y1r),
                        xref="x", yref="y",
                        line=dict(width=0), fillcolor=fill,
                    ))
                # Capacity lines on top of zones
                cap_line = dict(color="rgba(239,68,68,0.9)", width=1.5, dash="solid")
                shapes.append(dict(type="line", x0=dep_cap, x1=dep_cap, y0=0, y1=arr_cap, xref="x", yref="y", line=cap_line))
                shapes.append(dict(type="line", x0=0, x1=dep_cap, y0=arr_cap, y1=arr_cap, xref="x", yref="y", line=cap_line))
        axis_unit = _yaxis_unit(aggregate_base)
        # Axis range calculation (0starting from)
        all_x_vals = [p["x"] for p in points]
        all_y_vals = [p["y"] for p in points]
        x_max_val = max(all_x_vals) if all_x_vals else 100
        y_max_val = max(all_y_vals) if all_y_vals else 100
        # add some space
        x_range_max = x_max_val * 1.1 if x_max_val > 0 else 100
        y_range_max = y_max_val * 1.1 if y_max_val > 0 else 100
        
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=f"Dep 97.5% vs Arr 97.5% · {term}", font=dict(size=16)),
            xaxis_title=f"Dep 97.5% ({axis_unit})",
            yaxis_title=f"Arr 97.5% ({axis_unit})",
            height=480, margin=dict(t=60, b=50, l=60, r=40),
            paper_bgcolor="rgba(30,41,59,0.5)", plot_bgcolor="rgba(30,41,59,0.3)",
            font=dict(color="#e2e8f0", size=12),
            xaxis=dict(gridcolor="rgba(51,65,85,0.5)", range=[0, x_range_max]),
            yaxis=dict(gridcolor="rgba(51,65,85,0.5)", range=[0, y_range_max]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#cbd5e1")),
            showlegend=True,
            shapes=shapes,
        )
        # Draw gray points first, then selected scenario points last so they render on top and stay sharp
        # othersAlways draw if you have it
        if gray_x and gray_y:
            fig.add_trace(go.Scatter(
                x=gray_x, y=gray_y, mode="markers", name="Other",
                marker=dict(size=12, color=_SCATTER_OTHER_COLOR, line=dict(width=1, color="rgba(255,255,255,0.2)")),
                text=gray_text,
                hovertemplate="%{text}<br>Dep 97.5%: %{x:,.0f}<br>Arr 97.5%: %{y:,.0f}<extra></extra>",
            ))
        for j, (p, sid_str, idx) in enumerate(selected_points):
            color = colors[idx % len(colors)]
            fig.add_trace(go.Scatter(
                x=[p["x"]], y=[p["y"]], mode="markers+text",
                name=sid_str,
                legendgroup=sid_str,
                showlegend=True,
                marker=dict(size=16, color=color, line=dict(width=2, color="rgba(255,255,255,0.7)")),
                text=[sid_str], textposition="top center", textfont=dict(size=10),
            ))
        out.append((f"Dep vs Arr 97.5% · {term}", fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": True, "responsive": True})))
    return out


def _zone_score_from_ratio(ratio):
    """Score by capacity ratio: 0.7~1.3 → 100, 0.5~1.5 → 75, 0.3~1.7 → 50, else 25."""
    if ratio is None or (isinstance(ratio, float) and math.isnan(ratio)):
        return 25
    try:
        r = float(ratio)
    except (TypeError, ValueError):
        return 25
    if 0.7 <= r <= 1.3:
        return 100
    if 0.5 <= r <= 1.5:
        return 75
    if 0.3 <= r <= 1.7:
        return 50
    return 25


def _scenario_dep_arr_score_table(unique_ids, rep_summaries, report_labels, capacity_per_series):
    """Build per-scenario score table by location × Dep/Arr (100/75/50/25 by zone). Returns list of dicts with scenario_id, per (loc,da) score keys, total."""
    terminal_names = report_labels.get("terminal_names") or []
    if not terminal_names:
        return []
    n_loc = len(terminal_names)
    # capacity_per_series: [loc0 Dep, loc1 Dep, ...], [loc0 Arr, ...]
    caps_dep = [None] * n_loc
    caps_arr = [None] * n_loc
    if capacity_per_series and len(capacity_per_series) >= n_loc * 2:
        for i in range(n_loc):
            if capacity_per_series[i] and len(capacity_per_series[i]) > 0:
                try:
                    caps_dep[i] = float(capacity_per_series[i][0])
                except (TypeError, ValueError):
                    pass
            j = n_loc + i
            if capacity_per_series[j] and len(capacity_per_series[j]) > 0:
                try:
                    caps_arr[i] = float(capacity_per_series[j][0])
                except (TypeError, ValueError):
                    pass
    rows = []
    for i, sid in enumerate(unique_ids):
        rep = rep_summaries[i] if i < len(rep_summaries) else {}
        row_dict = {"scenario_id": str(sid)}
        total = 0
        for li, loc in enumerate(terminal_names):
            dep_val = _get_dep_arr(rep, loc, "Dep", "97.5%")
            arr_val = _get_dep_arr(rep, loc, "Arr", "97.5%")
            try:
                dep_val = float(dep_val) if dep_val is not None else None
                arr_val = float(arr_val) if arr_val is not None else None
            except (TypeError, ValueError):
                dep_val, arr_val = None, None
            r_dep = (dep_val / caps_dep[li]) if (dep_val is not None and caps_dep[li] and caps_dep[li] != 0) else None
            r_arr = (arr_val / caps_arr[li]) if (arr_val is not None and caps_arr[li] and caps_arr[li] != 0) else None
            s_dep = _zone_score_from_ratio(r_dep)
            s_arr = _zone_score_from_ratio(r_arr)
            row_dict[f"{loc}_Dep_score"] = s_dep
            row_dict[f"{loc}_Arr_score"] = s_arr
            total += (s_dep or 0) + (s_arr or 0)
        row_dict["total"] = total
        rows.append(row_dict)
    return rows


def _sorted_values_percentile_x_positions(n, x_use):
    """Sorted valuesin descending order(rank 0=max)When max, 97.5%, 95%, 90% location. Direction: Left=max, right=90%."""
    if n <= 0 or not x_use:
        return []
    try:
        x_float = [float(x_use[i]) for i in range(min(n, len(x_use)))]
    except (TypeError, ValueError):
        x_float = list(range(n))
    i_max = 0
    i_97_5 = max(0, min(n - 1, int(0.025 * n)))
    i_95 = max(0, min(n - 1, int(0.05 * n)))
    i_90 = max(0, min(n - 1, int(0.10 * n)))
    return [
        (x_float[i_max], "max"),
        (x_float[i_97_5], "97.5%"),
        (x_float[i_95], "95%"),
        (x_float[i_90], "90%"),
    ]


def _plotly_sorted_values_line_html(summaries_with_graph_data, colors, id_list=None):
    """By terminal·depart/Each by arrival Sorted Values Create chart. vertical line=gray dotted, horizontal lines=Capacity red line. return: [(title, html), ...]"""
    if not _HAS_PLOTLY or not summaries_with_graph_data:
        return []
    from collections import defaultdict
    colors = colors or REPORT_SCENARIO_PALETTE
    by_combo = defaultdict(list)
    for sm in summaries_with_graph_data:
        gd = sm.get("graph_data") or {}
        if gd.get("x_index") and gd.get("y_sorted_values"):
            term = (sm.get("focus_terminal") or "").strip()
            da = (sm.get("dep_arr") or "Dep").strip()
            agg = (sm.get("aggregate_base") or "").strip() or ""
            # aggsame terminal including also key·Dep/ArrEven if it's different aggis separated
            by_combo[(term, da, agg)].append(sm)
    # order: T1 Dep, T2 Dep, T1 Arr, T2 Arr (aggis already filtered so don't include it in the sort)
    combos = sorted(by_combo.keys(), key=lambda x: (0 if (x[1].upper() == "DEP") else 1, x[0], x[2]))
    out = []
    for (term, dep_arr, agg) in combos:
        group = by_combo[(term, dep_arr, agg)]
        if not group:
            continue
        fig = go.Figure()
        common_n = None
        common_x_use = None
        for si, sm in enumerate(group):
            gd = sm.get("graph_data") or {}
            x_index = gd.get("x_index")
            y_sorted = gd.get("y_sorted_values")
            if not x_index or not y_sorted or len(x_index) != len(y_sorted):
                continue
            n = len(x_index)
            if common_n is None or n < common_n:
                common_n = n
                common_x_use = list(x_index[:n]) if not isinstance(x_index[0], (int, float)) else x_index[:n]
                try:
                    common_x_use = [float(x) for x in common_x_use]
                except (TypeError, ValueError):
                    common_x_use = list(range(n))
            name = (sm.get("id") or "").replace("Scenario_", "S")
            sid = sm.get("id")
            if id_list and sid is not None and sid in id_list:
                color = colors[id_list.index(sid) % len(colors)]
            else:
                color = colors[si % len(colors)]
            fig.add_trace(go.Scatter(
                x=x_index, y=[float(v) for v in y_sorted], mode="lines", name=name, line=dict(width=2, color=color),
                hovertemplate="Rank %{x}<br>Value %{y:,.0f}<extra></extra>"))
        if len(fig.data) == 0 or common_n is None or common_x_use is None:
            continue
        positions = _sorted_values_percentile_x_positions(common_n, common_x_use)
        i_max = 0
        i_97_5 = max(0, min(common_n - 1, int(0.025 * common_n)))
        i_95 = max(0, min(common_n - 1, int(0.05 * common_n)))
        i_90 = max(0, min(common_n - 1, int(0.10 * common_n)))
        indices = [i_max, i_97_5, i_95, i_90]
        for (x_val, label), idx in zip(positions, indices):
            ys = []
            names = []
            for sm in group:
                gd = sm.get("graph_data") or {}
                y_sorted = gd.get("y_sorted_values")
                if not y_sorted or len(y_sorted) <= idx:
                    continue
                try:
                    ys.append(float(y_sorted[idx]))
                    names.append((sm.get("id") or "").replace("Scenario_", "S"))
                except (TypeError, ValueError):
                    pass
            if ys:
                fig.add_trace(go.Scatter(
                    x=[x_val] * len(ys), y=ys, mode="markers",
                    marker=dict(size=10, color="rgba(239,68,68,0.3)", symbol="circle", line=dict(width=1, color="rgba(239,68,68,0.9)")),
                    text=[f"{label} · {nm}: {v:,.0f}" for nm, v in zip(names, ys)], hovertemplate="%{text}<extra></extra>",
                    showlegend=False))
        shapes = []
        annotations = []
        cap = None
        for sm in group:
            c = _float(sm, "capacity", None)
            if c is not None and (isinstance(c, (int, float)) and c == c):
                cap = float(c)
                break
        if cap is not None:
            shapes.append(dict(type="line", x0=0, x1=1, y0=cap, y1=cap, xref="paper", yref="y", line=dict(dash="solid", color="rgba(239,68,68,0.9)", width=1.5)))
        # aggis already taken from the loop variable
        agg_label = _format_agg_for_axis(agg) if agg else "value"
        yaxis_title = f"Demand ({agg_label}/hour)"
        title = f"Sorted Values · {term} {dep_arr}"
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)), xaxis_title="Rank (index)", yaxis_title=yaxis_title,
            height=420, margin=dict(t=60, b=50, l=60, r=80), paper_bgcolor="rgba(30,41,59,0.5)", plot_bgcolor="rgba(30,41,59,0.3)",
            font=dict(color="#e2e8f0", size=12), xaxis=dict(gridcolor="rgba(51,65,85,0.5)"), yaxis=dict(gridcolor="rgba(51,65,85,0.5)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified",
            shapes=shapes, annotations=annotations, autosize=True)
        chart_html = fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": True, "responsive": True})
        out.append((agg_label, dep_arr, title, chart_html))
    return out


def load_scenarios_from_json(origin: Union[str, Path, dict, list]) -> List[dict]:
    """
    JSONScenario dict Load into list.
    - origin: JSON file path(str/Path), single scenario dict, or scenario dict list
    """
    if origin is None:
        return []

    # file path
    if isinstance(origin, (str, Path)):
        path = Path(origin)
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, TypeError):
            return []
        return _normalize_to_scenario_list(data)

    # already dict or list
    return _normalize_to_scenario_list(origin)


def _normalize_to_scenario_list(data: Any) -> List[dict]:
    """single dict / list of dict / Others → scenario dict list."""
    if data is None:
        return []
    if isinstance(data, list):
        return [s for s in data if isinstance(s, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _minimal_html_no_data():
    """Minimal HTML when no data is available."""
    return """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Scenario Report</title></head>
<body style="font-family: sans-serif; padding: 2rem; background: #0f172a; color: #e2e8f0;">
  <div style="max-width: 600px; margin: 0 auto;">
    <h1>Scenario Report</h1>
    <p>No JSON input or no scenario data. Please provide JSON and try again.</p>
  </div>
</body>
</html>"""


def _html_common_indicators_chart_wrap(kind, chart_html):
    """Common Indicators Single chart within section wrap. kind: '97.5' | '95' | 'excess'. output remains the same."""
    if kind == "97.5":
        return """
      <div class="chart-wrap">
        <div class="chart-title">Dep / Arr (97.5%)</div>
        <div class="plotly-embed">""" + chart_html + """</div>
        <p class="chart-desc">Same color per scenario. Dep/Arr/Both × locations: distinct patterns (solid, slash, dots, backslash, x, etc.). Red dashed line: Capacity.</p>
      </div>
"""
    if kind == "95":
        return """
      <div class="chart-wrap">
        <div class="chart-title">Dep / Arr (95%)</div>
        <div class="plotly-embed">""" + chart_html + """</div>
        <p class="chart-desc">Same color per scenario. Dep/Arr/Both × locations: distinct patterns (solid, slash, dots, backslash, x, etc.). Red dashed line: Capacity.</p>
      </div>
"""
    if kind == "excess":
        return """
      <div class="chart-wrap">
        <div class="chart-title">Excess as % of total · By location × Dep/Arr/Both</div>
        <div class="plotly-embed">""" + chart_html + """</div>
        <p class="chart-desc">Same color per scenario. Dep/Arr/Both × locations: distinct patterns for each combination.</p>
      </div>
"""
    if kind == "delay":
        return """
      <div class="chart-wrap">
        <div class="chart-title">Total Delay Hours (Capacity) · By location × Dep/Arr/Both</div>
        <div class="plotly-embed">""" + chart_html + """</div>
        <p class="chart-desc">Total delay hours from capacity constraint (cascading overflow). Same color per scenario.</p>
      </div>
"""
    if kind == "avg_delay":
        return """
      <div class="chart-wrap">
        <div class="chart-title">Avg Delay (per throughput) · By location × Dep/Arr/Both</div>
        <div class="plotly-embed">""" + chart_html + """</div>
        <p class="chart-desc">Total Delay Hours ÷ Throughput. Average delay per unit. Same color per scenario.</p>
      </div>
"""
    if kind == "avg_seat":
        return """
      <div class="chart-wrap">
        <div class="chart-title">Avg Seat · By location × Dep/Arr/Both</div>
        <div class="plotly-embed">""" + chart_html + """</div>
        <p class="chart-desc">Total seats ÷ Total flights. Independent of aggregate base. Same color per scenario.</p>
      </div>
"""
    return ""


def _html_table_from_split(split_dict, capacity=None, table_class="report-table", max_height_px=500):
    """Build HTML table from pandas-like split format: index (row labels), columns, data (list of rows). If capacity is set, cells >= capacity get red text."""
    idx = split_dict.get("index") or []
    cols = split_dict.get("columns") or []
    data = split_dict.get("data") or []
    if not cols and not idx:
        return ""
    cap = capacity if capacity is not None and isinstance(capacity, (int, float)) else None
    thead = "<thead><tr><th></th>" + "".join('<th class="num">' + _html.escape(str(c)) + "</th>" for c in cols) + "</tr></thead>\n"
    tbody = ""
    for r, row_label in enumerate(idx):
        row_vals = data[r] if r < len(data) else []
        cells = ""
        for c, val in enumerate(row_vals):
            if c >= len(cols):
                break
            num_val = None
            try:
                num_val = float(val) if val is not None else None
            except (TypeError, ValueError):
                pass
            style = ""
            if cap is not None and num_val is not None and num_val >= cap:
                style = ' style="color: #e53935;"'
            disp = "" if val is None else (f"{num_val:.0f}" if num_val is not None and isinstance(val, (int, float)) else str(val))
            cells += f'<td class="num"{style}>{_html.escape(disp)}</td>'
        if len(row_vals) < len(cols):
            cells += "".join('<td class="num"></td>' for _ in range(len(cols) - len(row_vals)))
        tbody += f"<tr><th>{_html.escape(str(row_label))}</th>{cells}</tr>"
    style_attr = f' style="max-height: {max_height_px}px; overflow: auto;"' if max_height_px else ""
    return f'<div class="table-wrap"{style_attr}><table class="{_html.escape(table_class)}">{thead}<tbody>{tbody}</tbody></table></div>'


def _plotly_daily_hourly_bars_html(summary, yaxis_title="Value"):
    """Bar graph by day of the week and time zone (grouped bars). summary has daily_hourly_pivot (index=time_of_day, columns=date (day), data). Returns HTML or empty string."""
    if not _HAS_PLOTLY:
        return ""
    pivot = summary.get("daily_hourly_pivot") or {}
    idx = pivot.get("index") or []
    cols = pivot.get("columns") or []
    data = pivot.get("data") or []
    if not idx or not cols or not data:
        return ""
    capacity = summary.get("capacity")
    try:
        colors = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd"]
        fig = go.Figure()
        for c, col_name in enumerate(cols):
            y = [float((row[c] if c < len(row) else 0) or 0) for row in data]
            fig.add_trace(go.Bar(x=list(idx), y=y, name=str(col_name), marker=dict(color=colors[c % len(colors)])))
        if capacity is not None:
            try:
                cap = float(capacity)
                fig.add_hline(y=cap, line_dash="dash", line_color="red", annotation_text=f"Capacity: {cap}", annotation_position="right")
            except (TypeError, ValueError):
                pass
        fig.update_layout(
            title=dict(text=""),
            xaxis_title="Time of Day",
            yaxis_title=yaxis_title,
            height=480,
            barmode="group",
            margin=dict(t=30, b=80, l=60, r=150),
            xaxis=dict(tickangle=45, type="category", tickfont=dict(color="#ffffff"), title_font=dict(color="#ffffff")),
            yaxis=dict(tickfont=dict(color="#ffffff"), title_font=dict(color="#ffffff")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            legend=dict(font=dict(color="#ffffff"), orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": True, "responsive": True})
    except Exception:
        return ""


def _plotly_stacked_by_group_html(summary, group_col_label="Group", yaxis_title="Value"):
    """By time zone Groupstar Stacked Bar. summary has stacked_pivot (index=Time, columns=group names, data). Returns HTML or empty string."""
    if not _HAS_PLOTLY:
        return ""
    pivot = summary.get("stacked_pivot") or {}
    idx = pivot.get("index") or []
    cols = pivot.get("columns") or []
    data = pivot.get("data") or []
    if not idx or not cols or not data:
        return ""
    capacity = summary.get("capacity")
    try:
        colors = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f"]
        fig = go.Figure()
        for c, col_name in enumerate(cols):
            y = [float((row[c] if c < len(row) else 0) or 0) for row in data]
            fig.add_trace(go.Bar(
                x=[str(i) for i in idx],
                y=y,
                name=str(col_name),
                marker=dict(color=colors[c % len(colors)]),
            ))
        if capacity is not None:
            try:
                cap = float(capacity)
                fig.add_hline(y=cap, line_dash="dash", line_color="red", annotation_text=f"Capacity: {cap}", annotation_position="right")
            except (TypeError, ValueError):
                pass
        group_col = summary.get("stacked_pivot_group_col") or group_col_label
        fig.update_layout(
            title=dict(text=""),
            xaxis_title="Time",
            yaxis_title=yaxis_title,
            height=480,
            barmode="stack",
            margin=dict(t=30, b=80, l=60, r=150),
            xaxis=dict(tickangle=45, type="category", tickfont=dict(color="#ffffff"), title_font=dict(color="#ffffff")),
            yaxis=dict(tickfont=dict(color="#ffffff"), title_font=dict(color="#ffffff")),
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, font=dict(color="#ffffff")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": True, "responsive": True})
    except Exception:
        return ""


def _render_summary_by_group_html(sbg):
    """summary_by_group → HTML. sbg: {tabs: [{label, data}]} or {index, columns, data}."""
    if not sbg or not isinstance(sbg, dict):
        return ""
    tabs = sbg.get("tabs")
    if tabs and isinstance(tabs, list) and len(tabs) >= 2:
        # tab UIraw render
        container_id = "sbg_tabs_" + str(id(sbg) % 100000)
        out = [f'<div class="tabs-container" id="{container_id}">\n', '  <div class="tabs-header">\n']
        for i, t in enumerate(tabs):
            lab = t.get("label") or f"Tab {i+1}"
            active = " active" if i == 0 else ""
            out.append(f'    <button class="tab-btn{active}" data-tab="sbg_panel_{container_id}_{i}" data-tabs-container="{container_id}">' + _html.escape(str(lab)) + "</button>\n")
        out.append("  </div>\n")
        for i, t in enumerate(tabs):
            d = t.get("data") or {}
            display = "block" if i == 0 else "none"
            out.append(f'  <div class="tab-panel" id="sbg_panel_{container_id}_{i}" data-tabs-container="{container_id}" style="display: {display};">\n')
            tbl = _html_table_from_split(d, capacity=None, max_height_px=300) if (d.get("index") or d.get("data")) else ""
            if tbl:
                out.append("    <div class=\"chart-wrap\">" + tbl + "</div>\n")
            out.append("  </div>\n")
        out.append("</div>\n")
        return "".join(out)
    if tabs and len(tabs) == 1:
        d = tabs[0].get("data") or {}
        return _html_table_from_split(d, capacity=None, max_height_px=300) if (d.get("index") or d.get("data")) else ""
    return _html_table_from_split(sbg, capacity=None, max_height_px=300) if (sbg.get("index") or sbg.get("data")) else ""


def _render_time_series_panel(summary, agg_label):
    """Single summary → (1) table HTML, (2) daily hourly bar HTML, (3) stacked bar HTML, (4) summary-by-group table HTML. agg_label for y-axis."""
    y_label = _format_agg_for_axis(summary.get("aggregate_base") or "") or "Value"
    pivot = summary.get("daily_hourly_pivot") or {}
    capacity = summary.get("capacity")
    table_html = _html_table_from_split(pivot, capacity=capacity, max_height_px=450) if (pivot.get("index") or pivot.get("data")) else ""
    bar_html = _plotly_daily_hourly_bars_html(summary, yaxis_title=y_label)
    stacked_html = _plotly_stacked_by_group_html(summary, yaxis_title=y_label)
    sbg = summary.get("summary_by_group") or {}
    summary_table_html = _render_summary_by_group_html(sbg)
    return table_html, bar_html, stacked_html, summary_table_html


def _html_radar_chart_wrap(radial_html):
    """Radar section single chart wrap."""
    return """
        <div class="chart-wrap">
          <div class="plotly-embed">""" + radial_html + """</div>
          <p class="chart-desc">Location Dep/Arr 97.5% peak only. No fill. Red dashed line: Capacity.</p>
        </div>
"""


_SCATTER_ZONE_DESC = (
    'Scatter from statistics table. Compared scenarios use palette colors; other scenarios in light gray (more transparent).<br>'
    'Zones: <span class="zone-swatch" style="background:rgba(74,222,128,0.7);"></span> Both 0.7×~1.3× (bright green, 100 pts); '
    '<span class="zone-swatch" style="background:rgba(34,197,94,0.6);"></span> Both 0.5×~1.5× (green, 75 pts); '
    '<span class="zone-swatch" style="background:rgba(251,146,60,0.7);"></span> Both 0.3×~1.7× (orange, 50 pts); '
    '<span class="zone-swatch" style="background:rgba(239,68,68,0.55);"></span> Otherwise (red, 25 pts).'
)


def _html_scatter_chart_wrap(chart_title, chart_html, use_zone_desc=True):
    """Dep 97.5% vs Arr 97.5% scatter single chart wrap. use_zone_desc=FalseShort Description(For tab panel)."""
    desc = _SCATTER_ZONE_DESC if use_zone_desc else "Scatter from statistics table. Compared scenarios use palette colors; other scenarios in light gray."
    return """
      <div class="chart-wrap">
        <div class="chart-title">""" + _html.escape(chart_title) + """</div>
        <div class="plotly-embed">""" + chart_html + """</div>
        <p class="chart-desc">""" + desc + """</p>
      </div>
"""


def _html_scatter_score_table(score_rows_agg, tns):
    """Scenario score by location and Dep/Arr table HTML."""
    header_cells = "".join(
        f'<th class="num">{_html.escape(str(loc))} Dep</th><th class="num">{_html.escape(str(loc))} Arr</th>' for loc in tns
    ) + '<th class="num">Total</th>'
    rows = ""
    for r in score_rows_agg:
        cells = "".join(
            f'<td class="num">{r.get(f"{loc}_Dep_score", "")}</td><td class="num">{r.get(f"{loc}_Arr_score", "")}</td>'
            for loc in tns
        ) + f'<td class="num"><strong>{r.get("total", "")}</strong></td>'
        sid = r.get("scenario_id", "")
        rows += f"<tr><td>{_html.escape(str(sid))}</td>{cells}</tr>"
    return """
      <div class="chart-wrap" style="margin-top: 1rem;">
        <h3 class="section-subtitle">Scenario score by location and Dep/Arr (100 / 75 / 50 / 25)</h3>
        <table class="report-table">
          <thead><tr><th>Scenario</th>""" + header_cells + """</tr></thead>
          <tbody>""" + rows + """
          </tbody>
        </table>
      </div>
"""


def build_report_html(scenarios: Union[List[dict], dict, str, Path]) -> str:
    """
    scenario JSON(list/dict/channel)take it Plotly using only charts HTML Generate a report.
    """
    if not isinstance(scenarios, list):
            scenarios = load_scenarios_from_json(scenarios)
    scenarios = [s for s in (scenarios or []) if s is not None and isinstance(s, dict)]

    summaries = [extract_summary(s) for s in scenarios]
    summaries = [
        s for s in summaries
        if s.get("id") is not None
        or s.get("total_sum") is not None
        or s.get("required_facilities") is not None
    ]
    if not summaries:
        return _minimal_html_no_data()

    # scenario ID Based on the unique list and by_id group
    def _sid(s):
        return s.get("id") if s.get("id") else "_anonymous_%s" % id(s)

    unique_ids = list(dict.fromkeys([_sid(s) for s in summaries]))
    from collections import defaultdict
    by_id = defaultdict(list)
    for s in summaries:
        by_id[_sid(s)].append(s)
    def _sort_key(s):
        t = s.get("focus_terminal") or ""
        d = (s.get("dep_arr") or s.get("focus_dep_arr") or "Dep").upper()
        return (t, 0 if d == "DEP" else 1)
    for sid in by_id:
        by_id[sid] = sorted(by_id[sid], key=_sort_key)

    # aggNot much scenario group (Multiple of the same scenario aggMay appear in → agg Each tab agg data only)
    by_agg = defaultdict(lambda: defaultdict(list))
    for s in summaries:
        sid = _sid(s)
        agg = (s.get("aggregate_base") or "").strip()
        by_agg[agg][sid].append(s)
    for agg in by_agg:
        for sid in by_agg[agg]:
            by_agg[agg][sid] = sorted(by_agg[agg][sid], key=_sort_key)

    # Merge by scenario rep (One per scenario: id, terminal_names, by_location)
    rep_summaries = [_merge_rep_from_group(by_id.get(sid) or [], sid) for sid in unique_ids]
    n_scenarios = len(unique_ids)
    if not rep_summaries:
        return _minimal_html_no_data()

    # Label — The location name is the merged repof terminal_names integration (my code T1/T2 doesn't exist)
    seen_order = []
    for sm in rep_summaries:
        tns = sm.get("terminal_names") or []
        if isinstance(tns, (list, tuple)) and tns:
            for t in tns:
                t = str(t).strip() if t is not None else ""
                if t and t not in seen_order:
                    seen_order.append(t)
    if not seen_order:
        seen_order = list(_FALLBACK_LOCATION_NAMES)
    tn = seen_order
    report_labels = {
        "first_terminal": tn[0] if tn else _FALLBACK_LOCATION_NAMES[0],
        "second_terminal": (tn[1] if len(tn) >= 2 else tn[0]) if tn else _FALLBACK_LOCATION_NAMES[1],
        "terminal_names": tn,
        "aggregate_base": (summaries[0] or {}).get("aggregate_base") or "",
    }

    colors = [REPORT_SCENARIO_PALETTE[i % len(REPORT_SCENARIO_PALETTE)] for i in range(max(n_scenarios, 1))]
    id_list = unique_ids

    # ——— Plotly Batch creation of charts ———
    plotly_charts = _build_comparison_bar_charts_plotly(id_list, rep_summaries, report_labels, colors)
    summaries_with_graph_data = [s for s in summaries if (s.get("graph_data") or {}).get("x_index") and (s.get("graph_data") or {}).get("y_sorted_values")]
    plotly_sorted_by_combo = _plotly_sorted_values_line_html(summaries_with_graph_data, colors, id_list) if summaries_with_graph_data else []

    # radial chart: statistics_table·capacity_per_seriesbelow(by_id since)Called after creation in
    plotly_radial_dep_arr_97 = ""

    date_range0 = (summaries[0] or {}).get("date_range") or {}
    date_start = (date_range0.get("start") or "")[:10] if isinstance(date_range0, dict) else ""
    date_end = (date_range0.get("end") or "")[:10] if isinstance(date_range0, dict) else ""

    # ——— HTML assembly (Plotlyuse only) ———
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Scenario Comparison Report · {n_scenarios} scenario(s)</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {{ --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8; --border: #334155; }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem; line-height: 1.6; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ font-size: 1.75rem; font-weight: 700; margin-bottom: 0.5rem; }}
    .subtitle {{ color: var(--muted); font-size: 0.95rem; margin-bottom: 2rem; }}
    .section {{ margin-bottom: 2.5rem; }}
    .section-title {{ font-size: 1.25rem; font-weight: 600; margin-bottom: 1rem; color: var(--accent); }}
    .subsection-title {{ font-size: 1.05rem; font-weight: 600; margin: 1.5rem 0 0.75rem 0; color: var(--muted); }}
    .section-subtitle {{ font-size: 1rem; font-weight: 600; margin: 0 0 0.75rem 0; color: var(--text); }}
    .chart-wrap {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; width: 100%; box-sizing: border-box; }}
    .chart-title {{ font-size: 0.95rem; font-weight: 600; margin-bottom: 1rem; }}
    .chart-desc {{ margin-top: 0.75rem; padding: 0.75rem 1rem; background: rgba(15,23,42,0.6); border-radius: 8px; font-size: 0.85rem; color: var(--muted); line-height: 1.5; border-left: 3px solid var(--accent); word-wrap: break-word; overflow-wrap: break-word; }}
.chart-desc .zone-swatch {{ display: inline-block; width: 10px; height: 10px; vertical-align: middle; margin-right: 5px; border-radius: 2px; }}
    .plotly-embed .plotly-graph-div, .plotly-embed > div {{ width: 100% !important; max-width: 100% !important; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem; }}
    .card h3 {{ margin: 0 0 0.75rem 0; font-size: 1rem; color: var(--accent); }}
    .card .value {{ font-size: 1.5rem; font-weight: 700; }}
    .card .value-desc {{ font-size: 0.75rem; font-weight: 500; color: var(--muted); margin-left: 0.35rem; }}
    .card .label {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.25rem; }}
    .card .unit-assignments {{ margin-top: 1rem; padding-top: 0.75rem; border-top: 1px solid var(--border); }}
    .card .unit-term-label {{ font-size: 0.8rem; font-weight: 600; color: var(--accent); margin-bottom: 0.35rem; }}
    .card .unit-chips {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.75rem; }}
    .card .unit-chip {{ display: inline-block; padding: 0.35rem 0.6rem; font-size: 0.75rem; border-radius: 8px; background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.4); color: var(--text); }}
    .card .unit-chip.default {{ background: rgba(148, 163, 184, 0.2); border-color: rgba(148, 163, 184, 0.5); }}
    .card table.report-table.card-metrics-table {{ font-size: 0.8rem; margin: 0.5rem 0; }}
    .card table.report-table.card-metrics-table th, .card table.report-table.card-metrics-table td {{ padding: 0.4rem 0.5rem; }}
    .plotly-embed {{ min-height: 320px; width: 100%; }}
    .report-overview {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem; margin-bottom: 2rem; font-size: 0.95rem; line-height: 1.7; }}
    .report-overview strong {{ color: var(--accent); }}
    table.report-table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    table.report-table th, table.report-table td {{ padding: 0.6rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }}
    table.report-table th {{ color: var(--muted); font-weight: 600; }}
    table.report-table td.num {{ text-align: right; }}
    table.report-table-summary {{ font-size: 1rem; }}
    table.report-table-summary th {{ font-size: 1rem; padding: 0.75rem 1rem; }}
    table.report-table-summary td {{ padding: 0.75rem 1rem; }}
    .summary-section .chart-wrap {{ border-left: 4px solid var(--accent); }}
    .sorted-tabs-row {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; align-items: center; }}
    .sorted-tabs-row input {{ display: none; }}
    .sorted-tab-label {{ padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid var(--border); background: var(--card); color: var(--muted); cursor: pointer; font-size: 0.9rem; }}
    .sorted-tabs-row input:checked + .sorted-tab-label {{ background: var(--accent); color: var(--bg); border-color: var(--accent); }}
    .sorted-tab-panel {{ display: none; margin-top: 0.5rem; }}
    .sorted-tab-panel.active {{ display: block; }}
    .ts-scenario-panels-wrap {{ display: grid; grid-template-columns: 1fr; margin-top: 0.5rem; }}
    .ts-scenario-panels-wrap > .ts-scenario-panel {{ grid-row: 1; grid-column: 1; min-height: 400px; }}
    .ts-scenario-panels-wrap > .ts-scenario-panel.ts-sc-hidden {{ visibility: hidden; pointer-events: none; }}
    .tabs-container {{ margin-bottom: 1rem; }}
    .tabs-header {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; align-items: center; }}
    .tab-btn {{ padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid var(--border); background: var(--card); color: var(--muted); cursor: pointer; font-size: 0.9rem; }}
    .tab-btn:hover {{ border-color: var(--accent); color: var(--text); }}
    .tab-btn.active {{ background: var(--accent); color: var(--bg); border-color: var(--accent); }}
    .tab-panel {{ margin-top: 0.5rem; }}
    footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border); color: var(--muted); font-size: 0.85rem; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Scenario Comparison Report</h1>
    <p class="subtitle">{n_scenarios} scenario(s) · Analysis period: {date_start} ~ {date_end}</p>
"""

    # Comparison Overview: Aggregate base count·List of types, terminals, scenarios
    aggregate_bases = sorted(set(s.get("aggregate_base") or "" for s in summaries if s.get("aggregate_base")))
    n_agg = len(aggregate_bases)
    agg_text = ", ".join(aggregate_bases) if aggregate_bases else "—"
    scenario_list_text = ", ".join(unique_ids)
    html += f"""
    <div class="report-overview">
      <h2 class="section-title">Comparison Overview</h2>
      <p><strong>Aggregate base</strong>: {n_agg} — {agg_text}. (Base metric for aggregation.)</p>
      <p><strong>Analysis period</strong>: {date_start} ~ {date_end}</p>
      <p><strong>Locations</strong>: {", ".join(report_labels.get("terminal_names", []) or _FALLBACK_LOCATION_NAMES)}</p>
      <p><strong>Scenarios compared</strong>: {scenario_list_text} — Dep/Arr metrics are shown separately in the charts and tables below.</p>
      <p><strong>Scenario summary</strong>: For each scenario, total (seats or movement by aggregate), required facilities, excess as % of total, and delay (h) are shown per location × Dep/Arr combination (e.g. T2 Dep, T2 Arr, T4 Dep, T4 Arr).</p>
    </div>
"""

    # terminal×List of departure and arrival combinations (For displaying capacity overage ratio, etc.) — summaries standard. Common Indicators: Dep, Arr, Both include
    _combo_key = lambda s: (s.get("focus_terminal") or "", (s.get("dep_arr") or s.get("focus_dep_arr") or "Dep").strip())
    _all_combos = set(_combo_key(s) for s in summaries)
    _dep_arr_order = {"Dep": 0, "Arr": 1, "Both": 2}
    combos = sorted(_all_combos, key=lambda x: (_dep_arr_order.get(str(x[1]).strip(), 3), x[0]))
    def _excess_for(sid, term, dep_arr):
        for s in by_id.get(sid) or []:
            if (s.get("focus_terminal") or "") == term and (s.get("dep_arr") or s.get("focus_dep_arr") or "Dep").strip() == dep_arr:
                return _float(s, "excess_as_percent_of_total", 0)
        return None
    def _delay_for(sid, term, dep_arr):
        rep = next((r for r in rep_summaries if (r.get("id") or "") == sid), None)
        if rep:
            return _get_dep_arr(rep, term, dep_arr, "delay_hours")
        for s in by_id.get(sid) or []:
            if (s.get("focus_terminal") or "") == term and (s.get("dep_arr") or s.get("focus_dep_arr") or "Dep").strip() == dep_arr:
                return _float(s, "total_delay_hours_capacity_constraint", 0)
        return None

    def _avg_delay_for(sid, term, dep_arr):
        rep = next((r for r in rep_summaries if (r.get("id") or "") == sid), None)
        if rep:
            d = _get_dep_arr(rep, term, dep_arr, "delay_hours")
            t = _get_dep_arr(rep, term, dep_arr, "total_sum")
            if t and float(t) > 0 and d is not None:
                return float(d) / float(t)
        for s in by_id.get(sid) or []:
            if (s.get("focus_terminal") or "") == term and (s.get("dep_arr") or s.get("focus_dep_arr") or "Dep").strip() == dep_arr:
                d = _float(s, "total_delay_hours_capacity_constraint", 0)
                t = s.get("total_sum")
                if t and float(t) > 0:
                    return float(d) / float(t)
        return None

    def _avg_seat_for(sid, term, dep_arr):
        rep = next((r for r in rep_summaries if (r.get("id") or "") == sid), None)
        if rep:
            v = _get_dep_arr(rep, term, dep_arr, "avg_seat")
            if v is not None:
                return float(v)
        for s in by_id.get(sid) or []:
            if (s.get("focus_terminal") or "") == term and (s.get("dep_arr") or s.get("focus_dep_arr") or "Dep").strip() == dep_arr:
                v = _float(s, "avg_seat", None)
                if v is not None:
                    return float(v)
        return None

    # common indicators: Excess as % of total, Total Delay Hours, Avg Delay, Avg Seat
    plotly_excess_grouped = _plotly_excess_grouped_bar_html(unique_ids, combos, _excess_for, colors) if combos else ""
    plotly_delay_grouped = _plotly_delay_hours_grouped_bar_html(unique_ids, combos, _delay_for, colors) if combos else ""
    _avg_fmt = lambda v: f"{float(v):.3f}" if v is not None and abs(float(v)) < 10 else (f"{float(v):,.1f}" if v is not None else "0")
    plotly_avg_delay_grouped = _plotly_delay_hours_grouped_bar_html(
        unique_ids, combos, _avg_delay_for, colors,
        title="",
        yaxis_title="Avg Delay (hours)",
        text_fmt=_avg_fmt,
    ) if combos else ""
    _avg_seat_fmt = lambda v: f"{float(v):,.1f}" if v is not None else "0"
    plotly_avg_seat_grouped = _plotly_delay_hours_grouped_bar_html(
        unique_ids, combos, _avg_seat_for, colors,
        title="",
        yaxis_title="Avg Seat (seats/flight)",
        text_fmt=_avg_seat_fmt,
    ) if combos else ""
    terminal_names = report_labels.get("terminal_names") or []
    agg_main = (summaries[0] or {}).get("aggregate_base") or "" if summaries else ""
    include_both_global = any(str(c[1]).strip().upper() == "BOTH" for c in combos)
    capacity_per_series = _build_capacity_per_series(rep_summaries, terminal_names, agg_main, include_both=include_both_global)
    statistics_table = None
    statistics_tables_by_agg = {}
    for s in scenarios:
        if not isinstance(s, dict):
            continue
        t = (s or {}).get("statistics_table")
        meta = (s or {}).get("meta") or {}
        agg = (meta.get("aggregate_base") or "").strip() or ""
        sid_meta = meta.get("scenario_id") or ""
        if t and isinstance(t, list) and len(t) > 0:
            if statistics_table is None:
                statistics_table = t
            # aggThe star scenario results table corresponds to aggof export Use only once (Includes entire scenario rows, avoids duplication)
            if agg not in statistics_tables_by_agg:
                statistics_tables_by_agg[agg] = [dict(row) for row in t if isinstance(row, dict)]
    plotly_radial_dep_arr_97 = _plotly_radial_dep_arr_97_html(
        statistics_table, report_labels, unique_ids, rep_summaries, colors, capacity_per_series=capacity_per_series
    )
    plotly_dep_arr_scatter_by_terminal = _plotly_dep_arr_scatter_by_terminal_html(
        statistics_table, report_labels, unique_ids, rep_summaries, colors, capacity_per_series=capacity_per_series
    )
    dep_arr_score_table = _scenario_dep_arr_score_table(
        unique_ids, rep_summaries, report_labels, capacity_per_series
    )
    # aggFor star tab: yes aggFilter only scenarios with (by_agg use)
    def _unique_ids_and_reps_for_agg(agg):
        sids = [sid for sid in unique_ids if (by_agg.get(agg) or {}).get(sid)]
        reps = [_merge_rep_from_group((by_agg.get(agg) or {}).get(sid) or [], sid) for sid in sids]
        return sids, reps

    sorted_values_by_agg = []
    if aggregate_bases and summaries_with_graph_data:
        agg_keys_sv = [a for a in aggregate_bases if (a or "").strip()] or [""]
        for agg in agg_keys_sv:
            summaries_agg = [s for s in summaries_with_graph_data if (s.get("aggregate_base") or "").strip() == agg]
            if not summaries_agg:
                continue
            uids_agg, _ = _unique_ids_and_reps_for_agg(agg)
            if not uids_agg:
                continue
            indices_agg = [unique_ids.index(sid) for sid in uids_agg]
            colors_agg = [colors[i % len(colors)] for i in indices_agg]
            items_agg = _plotly_sorted_values_line_html(summaries_agg, colors_agg, uids_agg)
            if items_agg:
                sorted_values_by_agg.append((agg, _format_agg_for_axis(agg) if agg else "value", items_agg))

    # Time Series (Table by day of the week, bar by day of the week, Groupstar Stacked Bar, Summary by Group) — aggBy star, by scenario, term·dep_arrstar panel
    time_series_by_agg = []
    agg_keys_ts = [a for a in aggregate_bases if (a or "").strip()] if aggregate_bases else [""]
    _dep_arr_order = {"Dep": 0, "Arr": 1, "Both": 2}

    for agg in agg_keys_ts:
        summaries_ts = [s for s in summaries if (s.get("aggregate_base") or "").strip() == agg]
        summaries_ts = [s for s in summaries_ts if (s.get("daily_hourly_pivot") or {}).get("data")]
        if not summaries_ts:
            continue
        agg_label = _format_agg_for_axis(agg) if agg else "value"
        from collections import defaultdict
        by_scenario = defaultdict(list)
        for s in summaries_ts:
            sid = s.get("id") or s.get("focus_terminal") or id(s)
            sid = str(sid).strip() or "Scenario"
            term = (s.get("focus_terminal") or "").strip() or "Location"
            dep_arr = (s.get("dep_arr") or s.get("focus_dep_arr") or "Dep").strip()
            table_html, bar_html, stacked_html, summary_table_html = _render_time_series_panel(s, agg_label)
            if table_html or bar_html or stacked_html or summary_table_html:
                by_scenario[sid].append((term, dep_arr, table_html, bar_html, stacked_html, summary_table_html))
        if by_scenario:
            def _dedup_panels(panels):
                panels = sorted(panels, key=lambda p: (p[0], _dep_arr_order.get(p[1], 3)))
                seen = set()
                out = []
                for p in panels:
                    k = (p[0], p[1])
                    if k not in seen:
                        seen.add(k)
                        out.append(p)
                return out
            scenario_list = []
            seen_sids = set()
            for uid in unique_ids:
                match = None
                for k in by_scenario:
                    if k == uid or k.replace("Scenario_", "S") == str(uid).replace("Scenario_", "S"):
                        match = k
                        break
                if match and match not in seen_sids:
                    seen_sids.add(match)
                    panels_dedup = _dedup_panels(by_scenario[match])
                    label = str(match).replace("Scenario_", "S")
                    scenario_list.append((match, label, panels_dedup))
            for k in by_scenario:
                if k not in seen_sids:
                    seen_sids.add(k)
                    panels_dedup = _dedup_panels(by_scenario[k])
                    scenario_list.append((k, str(k).replace("Scenario_", "S"), panels_dedup))
            if scenario_list:
                time_series_by_agg.append((agg, agg_label, scenario_list))

    plotly_dep_arr_97_5 = _plotly_dep_arr_single_grouped_bar_html(unique_ids, rep_summaries, report_labels, colors, percentile="97.5%", capacity_per_series=capacity_per_series, combos=combos)
    plotly_dep_arr_95 = _plotly_dep_arr_single_grouped_bar_html(unique_ids, rep_summaries, report_labels, colors, percentile="95%", capacity_per_series=capacity_per_series, combos=combos)

    # Cumulative comparison: JSON Both of the scenario net_cumulative (Relocation_Master Cumulative profile Net cumulativeSame as)
    both_scenarios = [s for s in scenarios if isinstance(s, dict) and (s.get("meta") or {}).get("dep_arr", "").strip().upper() == "BOTH" and (s.get("time_series") or s.get("cumulative"))]
    cumulative_aggs = sorted(set((s.get("meta") or {}).get("aggregate_base") or "" for s in both_scenarios))
    cumulative_aggs = [a for a in cumulative_aggs if (a or "").strip()] or [""]
    cumulative_tabs = []
    for agg in cumulative_aggs:
        agg_label = _format_agg_for_axis(agg) if (agg or "").strip() else "value"
        charts = _plotly_cumulative_by_terminal_html(scenarios, unique_ids, colors, agg_filter=agg or None, series_type="Both")
        if charts:
            cumulative_tabs.append((f"{agg_label} · Both (Net cumulative)", charts))

    # Common Indicators: aggA lot of charts (For tabs)
    common_indicators_by_agg = []
    agg_tab_keys = [a for a in aggregate_bases if (a or "").strip()] if aggregate_bases else []
    if not agg_tab_keys and (plotly_dep_arr_97_5 or plotly_dep_arr_95 or plotly_excess_grouped or plotly_delay_grouped):
        agg_tab_keys = [""]
    for agg in agg_tab_keys:
        unique_ids_agg, rep_summaries_agg = _unique_ids_and_reps_for_agg(agg)
        if not unique_ids_agg:
            continue
        indices_agg = [unique_ids.index(sid) for sid in unique_ids_agg]
        _combos_agg_set = set(_combo_key(s) for s in summaries if (s.get("aggregate_base") or "").strip() == agg)
        combos_agg = sorted(_combos_agg_set, key=lambda x: (_dep_arr_order.get(str(x[1]).strip(), 3), x[0]))
        # Bothgo combosonly when in capacityto Both include
        include_both_agg = any(str(c[1]).strip().upper() == "BOTH" for c in combos_agg)
        capacity_per_series_agg = _build_capacity_per_series(rep_summaries_agg, terminal_names, agg, include_both=include_both_agg)
        colors_agg = [colors[i % len(colors)] for i in indices_agg]
        def _excess_for_agg(agg_key):
            def fn(sid, term, dep_arr):
                for s in (by_agg.get(agg_key) or {}).get(sid) or []:
                    if (s.get("focus_terminal") or "") == term and (s.get("dep_arr") or s.get("focus_dep_arr") or "Dep").strip() == dep_arr:
                        return _float(s, "excess_as_percent_of_total", 0)
                return None
            return fn
        def _delay_for_agg(agg_key):
            def fn(sid, term, dep_arr):
                rep = next((r for r in rep_summaries_agg if (r.get("id") or "") == sid), None)
                if rep:
                    return _get_dep_arr(rep, term, dep_arr, "delay_hours")
                for s in (by_agg.get(agg_key) or {}).get(sid) or []:
                    if (s.get("focus_terminal") or "") == term and (s.get("dep_arr") or s.get("focus_dep_arr") or "Dep").strip() == dep_arr:
                        return _float(s, "total_delay_hours_capacity_constraint", 0)
                return None
            return fn
        def _avg_delay_for_agg(agg_key):
            def fn(sid, term, dep_arr):
                rep = next((r for r in rep_summaries_agg if (r.get("id") or "") == sid), None)
                if rep:
                    d = _get_dep_arr(rep, term, dep_arr, "delay_hours")
                    t = _get_dep_arr(rep, term, dep_arr, "total_sum")
                    if t and float(t) > 0 and d is not None:
                        return float(d) / float(t)
                for s in (by_agg.get(agg_key) or {}).get(sid) or []:
                    if (s.get("focus_terminal") or "") == term and (s.get("dep_arr") or s.get("focus_dep_arr") or "Dep").strip() == dep_arr:
                        d = _float(s, "total_delay_hours_capacity_constraint", 0)
                        t = s.get("total_sum")
                        if t and float(t) > 0:
                            return float(d) / float(t)
                return None
            return fn
        html_97_5 = _plotly_dep_arr_single_grouped_bar_html(unique_ids_agg, rep_summaries_agg, report_labels, colors_agg, percentile="97.5%", capacity_per_series=capacity_per_series_agg, aggregate_base=agg, combos=combos_agg)
        html_95 = _plotly_dep_arr_single_grouped_bar_html(unique_ids_agg, rep_summaries_agg, report_labels, colors_agg, percentile="95%", capacity_per_series=capacity_per_series_agg, aggregate_base=agg, combos=combos_agg)
        html_excess = _plotly_excess_grouped_bar_html(unique_ids_agg, combos_agg, _excess_for_agg(agg), colors_agg) if combos_agg else ""
        excess_chart = _build_comparison_bar_charts_plotly(unique_ids_agg, rep_summaries_agg, report_labels, colors_agg)
        html_excess = html_excess or (excess_chart.get("excess") or "")
        html_delay = _plotly_delay_hours_grouped_bar_html(unique_ids_agg, combos_agg, _delay_for_agg(agg), colors_agg) if combos_agg else ""
        _avg_fmt_agg = lambda v: f"{float(v):.3f}" if v is not None and abs(float(v)) < 10 else (f"{float(v):,.1f}" if v is not None else "0")
        html_avg_delay = _plotly_delay_hours_grouped_bar_html(
            unique_ids_agg, combos_agg, _avg_delay_for_agg(agg), colors_agg,
            title="",
            yaxis_title="Avg Delay (hours)",
            text_fmt=_avg_fmt_agg,
        ) if combos_agg else ""
        def _avg_seat_for_agg(agg_key):
            def fn(sid, term, dep_arr):
                rep = next((r for r in rep_summaries_agg if (r.get("id") or "") == sid), None)
                if rep:
                    v = _get_dep_arr(rep, term, dep_arr, "avg_seat")
                    if v is not None:
                        return float(v)
                for s in (by_agg.get(agg_key) or {}).get(sid) or []:
                    if (s.get("focus_terminal") or "") == term and (s.get("dep_arr") or s.get("focus_dep_arr") or "Dep").strip() == dep_arr:
                        v = _float(s, "avg_seat", None)
                        if v is not None:
                            return float(v)
                return None
            return fn
        _avg_seat_fmt_agg = lambda v: f"{float(v):,.1f}" if v is not None else "0"
        html_avg_seat = _plotly_delay_hours_grouped_bar_html(
            unique_ids_agg, combos_agg, _avg_seat_for_agg(agg), colors_agg,
            title="",
            yaxis_title="Avg Seat (seats/flight)",
            text_fmt=_avg_seat_fmt_agg,
        ) if combos_agg else ""
        common_indicators_by_agg.append((_format_agg_for_axis(agg) if agg else "value", html_97_5, html_95, html_excess, html_delay, html_avg_delay, html_avg_seat))

    # Radar comparison: aggA lot of charts (For tabs)
    radar_by_agg = []
    radar_agg_keys = [a for a in aggregate_bases if (a or "").strip()] if aggregate_bases else []
    if not radar_agg_keys and plotly_radial_dep_arr_97:
        radar_agg_keys = [""]
    for agg in radar_agg_keys:
        unique_ids_agg, rep_summaries_agg = _unique_ids_and_reps_for_agg(agg)
        stat_table_agg = statistics_tables_by_agg.get(agg)
        has_selected = bool(unique_ids_agg)
        has_table = bool(stat_table_agg and len(stat_table_agg) > 0)
        if not has_selected and not has_table:
            continue
        indices_agg = [unique_ids.index(sid) for sid in unique_ids_agg] if unique_ids_agg else []
        capacity_per_series_agg = _build_capacity_per_series(rep_summaries_agg, terminal_names, agg, include_both=True)
        if not stat_table_agg and statistics_table:
            id_set_agg = set(unique_ids_agg) | {str(s).replace("Scenario_", "S") for s in unique_ids_agg}
            def _sid_in_set(row):
                sid = str(row.get("Scenario_ID") or row.get("scenario_id") or "").strip()
                sid_alt = sid.replace("Scenario_", "S")
                return sid in id_set_agg or sid_alt in id_set_agg
            stat_table_agg = [r for r in statistics_table if isinstance(r, dict) and _sid_in_set(r)]
        colors_agg = [colors[i % len(colors)] for i in indices_agg]
        # aggstar statistics table use — seat Includes applicable agg full scenario(selected + others) mark
        table_for_radial = stat_table_agg if stat_table_agg else None
        radial_html = _plotly_radial_dep_arr_97_html(
            table_for_radial, report_labels, unique_ids_agg, rep_summaries_agg, colors_agg, capacity_per_series=capacity_per_series_agg, aggregate_base=agg
        )
        if radial_html:
            radar_by_agg.append((_format_agg_for_axis(agg) if agg else "value", radial_html))

    # Dep 97.5% vs Arr 97.5% scatter: aggFor star tab
    scatter_by_agg = []
    scatter_agg_keys = [a for a in aggregate_bases if (a or "").strip()] if aggregate_bases else []
    if not scatter_agg_keys and plotly_dep_arr_scatter_by_terminal:
        scatter_agg_keys = [""]
    for agg in scatter_agg_keys:
        unique_ids_agg, rep_summaries_agg = _unique_ids_and_reps_for_agg(agg)
        stat_table_agg = statistics_tables_by_agg.get(agg)
        # corresponding aggEven if there is no scenario selected in statistics_tableIf there is data in scatter painting (entire otherdisplayed as)
        has_selected = bool(unique_ids_agg)
        has_table = bool(stat_table_agg and len(stat_table_agg) > 0)
        if not has_selected and not has_table:
            continue
        indices_agg = [unique_ids.index(sid) for sid in unique_ids_agg] if unique_ids_agg else []
        capacity_per_series_agg = _build_capacity_per_series(rep_summaries_agg, terminal_names, agg, include_both=True)
        # corresponding aggof statistics_table full use → scatterScenario selected from(palette) + remain(other, gray) show all
        table_for_scatter = stat_table_agg if stat_table_agg else None
        colors_agg = [colors[i % len(colors)] for i in indices_agg]
        scatter_list = _plotly_dep_arr_scatter_by_terminal_html(
            table_for_scatter, report_labels, unique_ids_agg, rep_summaries_agg, colors_agg,
            capacity_per_series=capacity_per_series_agg, aggregate_base=agg
        )
        score_rows_agg = _scenario_dep_arr_score_table(
            unique_ids_agg, rep_summaries_agg, report_labels, capacity_per_series_agg
        ) if capacity_per_series_agg else []
        if scatter_list or score_rows_agg:
            scatter_by_agg.append((_format_agg_for_axis(agg) if agg else "value", scatter_list, score_rows_agg))

    tns = report_labels.get("terminal_names") or []

    def _render_summary_cards(ids_list, reps_list, colors_list, agg=None):
        total_label = _total_metric_label(agg)
        out = []
        def _chip(s, is_default=False):
            cls = "unit-chip default" if is_default else "unit-chip"
            return '<span class="' + cls + '">' + _html.escape(str(s)) + '</span>'
        def _unit_assign_block(sid, rep, color, locs, chip_fn):
            bl = rep.get("by_location") or {}
            parts = [f'<div style="margin-bottom:0.75rem;"><span class="unit-term-label" style="color:{color};">{_html.escape(str(sid))}</span>']
            for loc in locs:
                loc_d = bl.get(loc) or {}
                dep_d = loc_d.get("Dep") or {}
                ulist = dep_d.get("units_list") or []
                chips = "".join(chip_fn(u) for u in ulist) if ulist else chip_fn("(none)")
                parts.append(f'<div style="margin-top:0.35rem;"><span style="color:var(--muted);font-size:0.8rem;">{_html.escape(str(loc))}</span> <span class="unit-chips">{chips}</span></div>')
            parts.append("</div>")
            return "".join(parts)

        if len(ids_list) >= 2:
            # common table: Location, Dep/Arr Group columns by scenario, once on the left
            header_cells = "<th>Location</th><th>Dep/Arr</th>"
            for i, sid in enumerate(ids_list):
                c = colors_list[i % len(colors_list)]
                header_cells += f'<th colspan="6" class="num" style="color:{c};">{_html.escape(str(sid))}</th>'
            rows_html = ""
            for loc in tns:
                for da in ("Dep", "Arr", "Both"):
                    row_cells = f"<td>{_html.escape(str(loc))}</td><td>{da}</td>"
                    for i in range(len(ids_list)):
                        rep = reps_list[i] if i < len(reps_list) else {}
                        bl = rep.get("by_location") or {}
                        loc_d = bl.get(loc) or {}
                        d = loc_d.get(da) or {}
                        both_d = loc_d.get("Both") or {}
                        total_seat = d.get("total_sum") or 0
                        exc = d.get("excess_%") or 0
                        try:
                            total_seat = float(total_seat) if total_seat is not None else 0
                            exc = float(exc) if exc is not None else 0
                        except (TypeError, ValueError):
                            total_seat, exc = 0, 0
                        peak_97 = d.get("97.5%")
                        try:
                            peak_97 = float(peak_97) if peak_97 is not None else 0
                        except (TypeError, ValueError):
                            peak_97 = 0
                        # Directional: Both Only when there is. Dep=Dep/Both, Arr=Arr/Both, Both=1
                        both_97_val = both_d.get("97.5%")
                        try:
                            both_97_val = float(both_97_val) if both_97_val is not None else 0
                        except (TypeError, ValueError):
                            both_97_val = 0
                        if da == "Both":
                            directional_str = "1"
                        elif both_97_val > 0:
                            directional_str = f"{peak_97 / both_97_val:.3f}"
                        else:
                            directional_str = "—"
                        cap_val = d.get("capacity")
                        try:
                            cap_val = float(cap_val) if cap_val is not None else None
                        except (TypeError, ValueError):
                            cap_val = None
                        cap_str = f"{cap_val:,.0f}" if cap_val is not None else "—"
                        delay_val = d.get("delay_hours")
                        try:
                            delay_val = float(delay_val) if delay_val is not None else None
                        except (TypeError, ValueError):
                            delay_val = None
                        delay_str = _fmt_delay_hours(delay_val)
                        row_cells += f'<td class="num">{(total_seat):,.0f}</td><td class="num">{exc:.2f}%</td><td class="num">{peak_97:,.0f}</td><td class="num">{directional_str}</td><td class="num">{cap_str}</td><td class="num">{delay_str}</td>'
                    rows_html += f"<tr>{row_cells}</tr>\n            "
            subheader = "<tr><th></th><th></th>"
            for _ in ids_list:
                subheader += f'<th class="num">{_html.escape(total_label)}</th><th class="num">Excess as % of total</th><th class="num">97.5%</th><th class="num">Directional</th><th class="num">Capacity</th><th class="num">Delay (h)</th>'
            subheader += "</tr>"
            out.append(f"""
        <div class="card" style="border-left: 4px solid var(--accent);">
          <table class="report-table card-metrics-table">
            <thead><tr>{header_cells}</tr>{subheader}</thead>
            <tbody>{rows_html}
            </tbody>
          </table>
          <div class="unit-assignments" style="margin-top:1rem;">
            <p class="subsection-title">Unit assignments by scenario</p>
""" + "".join(
                _unit_assign_block(ids_list[i], reps_list[i] if i < len(reps_list) else {}, colors_list[i % len(colors_list)], tns, _chip)
                for i in range(len(ids_list))
            ) + """
          </div>
        </div>""")
            return "\n".join(out)

        # 1 scenario: existing card format (Location, Dep/Arr Includes 1 table)
        for i, sid in enumerate(ids_list):
            c = colors_list[i % len(colors_list)]
            rep = reps_list[i] if i < len(reps_list) else {}
            bl = rep.get("by_location") or {}
            default_block = ""
            unit_blocks = []
            rows_html = ""
            for loc in tns:
                loc_d = bl.get(loc) or {}
                dep_d = loc_d.get("Dep") or {}
                arr_d = loc_d.get("Arr") or {}
                ulist = dep_d.get("units_list") or []
                dlist = dep_d.get("default_units_list") or []
                loc_chips = "".join(_chip(u) for u in ulist) if ulist else _chip("(none)")
                d_chips = "".join(_chip(u, True) for u in dlist) if dlist else ""
                if d_chips:
                    default_block += f"""
            <div class="unit-term-label" style="color:var(--muted);">{loc} fixed (default)</div>
            <div class="unit-chips">{d_chips}</div>"""
                unit_blocks.append(f"""
            <div class="unit-term-label">{loc} assignment</div>
            <div class="unit-chips">{loc_chips}</div>""")
                both_d = loc_d.get("Both") or {}
                both_97_val = both_d.get("97.5%")
                try:
                    both_97_val = float(both_97_val) if both_97_val is not None else 0
                except (TypeError, ValueError):
                    both_97_val = 0
                for da in ("Dep", "Arr", "Both"):
                    d = loc_d.get(da) or {}
                    total_seat = d.get("total_sum") or 0
                    exc = d.get("excess_%") or 0
                    try:
                        total_seat = float(total_seat) if total_seat is not None else 0
                        exc = float(exc) if exc is not None else 0
                    except (TypeError, ValueError):
                        total_seat, exc = 0, 0
                    peak_97 = d.get("97.5%")
                    try:
                        peak_97 = float(peak_97) if peak_97 is not None else 0
                    except (TypeError, ValueError):
                        peak_97 = 0
                    # Directional: Both Only when there is
                    if da == "Both":
                        directional_str = "1"
                    elif both_97_val > 0:
                        directional_str = f"{peak_97 / both_97_val:.3f}"
                    else:
                        directional_str = "—"
                    cap_val = d.get("capacity")
                    try:
                        cap_val = float(cap_val) if cap_val is not None else None
                    except (TypeError, ValueError):
                        cap_val = None
                    cap_str = f"{cap_val:,.0f}" if cap_val is not None else "—"
                    delay_val = d.get("delay_hours")
                    try:
                        delay_val = float(delay_val) if delay_val is not None else None
                    except (TypeError, ValueError):
                        delay_val = None
                    delay_str = _fmt_delay_hours(delay_val)
                    rows_html += f"""
            <tr>
              <td>{loc}</td>
              <td>{da}</td>
              <td class="num">{(total_seat):,.0f}</td>
              <td class="num">{exc:.2f}%</td>
              <td class="num">{peak_97:,.0f}</td>
              <td class="num">{directional_str}</td>
              <td class="num">{cap_str}</td>
              <td class="num">{delay_str}</td>
            </tr>"""
            if default_block:
                default_block = default_block + "\n            "
            out.append(f"""
        <div class="card" style="border-left: 4px solid {c};">
          <h3 style="color: {c};">{sid}</h3>
          <table class="report-table card-metrics-table">
            <thead><tr><th>Location</th><th>Dep/Arr</th><th>{_html.escape(total_label)}</th><th>Excess as % of total</th><th>97.5%</th><th>Directional</th><th>Capacity</th><th>Delay (h)</th></tr></thead>
            <tbody>{rows_html}
            </tbody>
          </table>
          <div class="unit-assignments">
            {default_block}
            """ + "".join(unit_blocks) + """
          </div>
        </div>""")
        return "\n".join(out)

    summary_agg_keys = [a for a in aggregate_bases if (a or "").strip()] if aggregate_bases else []
    if not summary_agg_keys:
        summary_agg_keys = [""]
    use_top_agg_tabs = len(summary_agg_keys) >= 2

    def _render_single_agg_body(agg_idx, agg):
        """single aggFull report text about(sections) HTML. top agg For tabs."""
        out = []
        agg_fmt = _format_agg_for_axis(agg) if (agg or "").strip() else "value"
        uids_agg, reps_agg = _unique_ids_and_reps_for_agg(agg)
        indices_agg = [unique_ids.index(sid) for sid in uids_agg] if uids_agg else []
        colors_agg = [colors[i % len(colors)] for i in indices_agg]
        tns = report_labels.get("terminal_names") or []
        # 1. Scenario Summary
        out.append("""
    <section class="section">
      <h2 class="section-title">Scenario Summary</h2>
      <div class="cards">
""")
        out.append(_render_summary_cards(uids_agg, reps_agg, colors_agg, agg=agg))
        out.append("""      </div>
    </section>
""")
        # 2. Common Indicators (Sections only have content)
        common_97_5 = common_95 = common_excess = common_delay = common_avg_delay = common_avg_seat = None
        for _item in common_indicators_by_agg:
            if _item[0] == agg_fmt:
                common_97_5, common_95, common_excess, common_delay = _item[1], _item[2], _item[3], _item[4]
                common_avg_delay = _item[5] if len(_item) > 5 else None
                common_avg_seat = _item[6] if len(_item) > 6 else None
                break
        if common_97_5 or common_95 or common_excess or common_delay or common_avg_delay or common_avg_seat:
            out.append("""
    <section class="section">
      <h2 class="section-title">Common Indicators (bar charts)</h2>
""")
            html_97_5, html_95, html_excess, html_delay, html_avg_delay, html_avg_seat = common_97_5, common_95, common_excess, common_delay, common_avg_delay, common_avg_seat
            if html_97_5:
                out.append(_html_common_indicators_chart_wrap("97.5", html_97_5))
            if html_95:
                out.append(_html_common_indicators_chart_wrap("95", html_95))
            if html_excess:
                out.append(_html_common_indicators_chart_wrap("excess", html_excess))
            if html_delay:
                out.append(_html_common_indicators_chart_wrap("delay", html_delay))
            if html_avg_delay:
                out.append(_html_common_indicators_chart_wrap("avg_delay", html_avg_delay))
            if html_avg_seat:
                out.append(_html_common_indicators_chart_wrap("avg_seat", html_avg_seat))
            out.append("    </section>\n")
        # 3. Time Series (Table by day of the week, bar by day of the week, Group Stacked Bar, Summary by Group) — Scenario-specific tabs, within each scenario term·dep_arr tab
        ts_scenario_list = None
        for _agg, _label, _scenario_list in time_series_by_agg:
            if _label == agg_fmt or _agg == agg:
                ts_scenario_list = _scenario_list
                break
        if ts_scenario_list:
            out.append("""
    <section class="section">
      <h2 class="section-title">Time Series</h2>
      <p class="subsection-title">Select scenario, then location and Dep/Arr.</p>
      <div class="sorted-tabs-row" id="time_series_scenario_tabs_""" + str(agg_idx) + """">
""")
            for si, (sid, label, panels) in enumerate(ts_scenario_list):
                tid = f"ts_scenario_tab_agg{agg_idx}_{si}"
                checked = " checked" if si == 0 else ""
                out.append(f'        <input type="radio" name="time_series_scenario_agg{agg_idx}" id="{tid}" value="{si}"{checked}><label class="sorted-tab-label" for="{tid}">{_html.escape(label)}</label>\n')
            out.append("      </div>\n")
            out.append('      <div class="ts-scenario-panels-wrap" id="ts_scenario_wrap_agg' + str(agg_idx) + '">\n')
            for si, (sid, label, panels) in enumerate(ts_scenario_list):
                sc_panel_id = f"ts_scenario_panel_agg{agg_idx}_{si}"
                sc_hidden = " ts-sc-hidden" if si != 0 else ""
                out.append(f'      <div class="ts-scenario-panel{sc_hidden}" id="{sc_panel_id}" data-ts-scenario-index="{si}" data-ts-agg="{agg_idx}">\n')
                out.append(f'      <div class="sorted-tabs-row" id="time_series_term_tabs_agg{agg_idx}_sc{si}">\n')
                for ti, (term, dep_arr, table_html, bar_html, stacked_html, summary_table_html) in enumerate(panels):
                    tid = f"ts_term_tab_agg{agg_idx}_sc{si}_{ti}"
                    checked = " checked" if ti == 0 else ""
                    tab_caption = f"{term} · {dep_arr}"
                    out.append(f'        <input type="radio" name="time_series_term_agg{agg_idx}_sc{si}" id="{tid}" value="{ti}"{checked}><label class="sorted-tab-label" for="{tid}">{_html.escape(tab_caption)}</label>\n')
                out.append("      </div>\n")
                for ti, (term, dep_arr, table_html, bar_html, stacked_html, summary_table_html) in enumerate(panels):
                    panel_id = f"ts_term_panel_agg{agg_idx}_sc{si}_{ti}"
                    t_active = " active" if ti == 0 else ""
                    out.append(f'      <div class="sorted-tab-panel{t_active}" id="{panel_id}" data-ts-term-index="{ti}" data-ts-scenario-index="{si}" data-ts-agg="{agg_idx}">\n')
                    if table_html:
                        out.append('        <h3 class="section-subtitle">Daily Time Slot Data Table</h3>\n        <div class="chart-wrap">' + table_html + "</div>\n")
                    if bar_html:
                        out.append('        <h3 class="section-subtitle">Daily Time Slot Bar Chart</h3>\n        <div class="chart-wrap"><div class="plotly-embed">' + bar_html + "</div></div>\n")
                    if stacked_html:
                        out.append('        <h3 class="section-subtitle">Time Slot by Group Stacked Bar Chart</h3>\n        <div class="chart-wrap"><div class="plotly-embed">' + stacked_html + "</div></div>\n")
                    if summary_table_html:
                        out.append('        <h3 class="section-subtitle">Summary by Group</h3>\n        <div class="chart-wrap">' + summary_table_html + "</div>\n")
                    out.append("      </div>\n")
                out.append("      </div>\n")
            out.append("      </div>\n")
            out.append("""
      <script>
(function(){
  var aggIdx = """ + str(agg_idx) + """;
  var scRadios = document.querySelectorAll('input[name="time_series_scenario_agg' + aggIdx + '"]');
  var scPanels = document.querySelectorAll('div[id^="ts_scenario_panel_agg' + aggIdx + '_"]');
  function resizePlotlyIn(el) {
    if (typeof Plotly !== 'undefined' && el) {
      var divs = el.querySelectorAll('.plotly-graph-div, .js-plotly-plot');
      divs.forEach(function(d){
        try {
          if (Plotly.Plots && Plotly.Plots.resize) Plotly.Plots.resize(d);
          else if (Plotly.Plot && Plotly.Plot.resize) Plotly.Plot.resize(d);
          if (Plotly.relayout) Plotly.relayout(d, {});
        } catch(e){}
      });
    }
  }
  function showScenario(i) {
    scPanels.forEach(function(p) {
      var match = p.getAttribute('data-ts-scenario-index') === String(i);
      p.classList.toggle('ts-sc-hidden', !match);
      if (match) {
        requestAnimationFrame(function(){
          setTimeout(function(){ resizePlotlyIn(p); }, 150);
        });
      }
    });
  }
  scRadios.forEach(function(r) {
    r.addEventListener('change', function() { showScenario(this.value); });
  });
  for (var si = 0; si < scRadios.length; si++) {
    (function(si){
      var radios = document.querySelectorAll('input[name="time_series_term_agg' + aggIdx + '_sc' + si + '"]');
      var panels = document.querySelectorAll('.sorted-tab-panel[data-ts-scenario-index="' + si + '"][data-ts-term-index]');
      function showTerm(i) {
      panels.forEach(function(p) {
        var match = p.getAttribute('data-ts-term-index') === String(i);
        p.classList.toggle('active', match);
        if (match && typeof Plotly !== 'undefined') {
          requestAnimationFrame(function(){
            setTimeout(function(){
              p.querySelectorAll('.plotly-graph-div, .js-plotly-plot').forEach(function(d){
                try {
                  if (Plotly.Plots && Plotly.Plots.resize) Plotly.Plots.resize(d);
                  else if (Plotly.Plot && Plotly.Plot.resize) Plotly.Plot.resize(d);
                  if (Plotly.relayout) Plotly.relayout(d, {});
                } catch(e){}
              });
            }, 150);
          });
        }
      });
      }
      radios.forEach(function(r) {
        r.addEventListener('change', function() { showTerm(this.value); });
      });
    })(si);
  }
  showScenario(0);
})();
      </script>
    </section>
""")
        # 4. Sorted Values (single-agg: Dep/Arr/Both tabs only)
        items_agg = None
        for _agg, _label, _items in sorted_values_by_agg:
            if _label == agg_fmt or _agg == agg:
                items_agg = _items
                break
        if items_agg is not None and len(items_agg) > 0:
            from collections import defaultdict
            by_tab = defaultdict(list)
            for item in items_agg:
                if len(item) == 4:
                    a_l, dep_arr, chart_title, chart_html = item
                    by_tab[(a_l, dep_arr)].append((chart_title, chart_html))
                else:
                    chart_title, chart_html = item[0], item[1]
                    by_tab[("value", "Dep")].append((chart_title, chart_html))
            tab_keys = sorted(by_tab.keys(), key=lambda k: (k[0], 0 if (k[1] or "Dep").upper() == "DEP" else 1, k[1] or "Dep"))
            tab_keys = [(a, d or "Dep") for a, d in tab_keys]
            out.append("""
    <section class="section">
      <h2 class="section-title">Sorted Values (by rank) · By location and Dep/Arr</h2>
      <p class="subsection-title">Select tab by Dep/Arr/Both.</p>
      <div class="sorted-tabs-row" id="sorted_tabs_container_""" + str(agg_idx) + """">
""")
            for i, (a_l, dep_arr) in enumerate(tab_keys):
                tid = f"sorted_tab_agg{agg_idx}_{i}"
                checked = " checked" if i == 0 else ""
                tab_caption = f"{a_l} · {dep_arr}"
                out.append(f'        <input type="radio" name="sorted_tabs_agg{agg_idx}" id="{tid}" value="{i}"{checked}><label class="sorted-tab-label" for="{tid}">{_html.escape(tab_caption)}</label>\n')
            out.append("      </div>\n")
            for i, (a_l, dep_arr) in enumerate(tab_keys):
                panel_id = f"sorted_panel_agg{agg_idx}_{i}"
                active = " active" if i == 0 else ""
                out.append(f'      <div class="sorted-tab-panel{active}" id="{panel_id}" data-tab-index="{i}" data-sorted-agg="{agg_idx}">\n')
                for chart_title, chart_html in by_tab[(a_l, dep_arr)]:
                    if not (chart_html and chart_html.strip()):
                        continue
                    out.append("""
        <div class="chart-wrap">
          <div class="plotly-embed">""" + chart_html + """</div>
          <p class="chart-desc">Horizontal red line: Capacity. Light red markers: scenario values at max / 97.5% / 95% / 90%.</p>
        </div>
""")
                out.append("      </div>\n")
            out.append("""
      <script>
(function(){
  var idx = """ + str(agg_idx) + """;
  var c = document.getElementById('sorted_tabs_container_' + idx);
  if (!c) return;
  var radios = c.querySelectorAll('input[name="sorted_tabs_agg' + idx + '"]');
  var panels = document.querySelectorAll('.sorted-tab-panel[data-sorted-agg="' + idx + '"]');
  function showTab(i) {
    panels.forEach(function(p) {
      p.classList.toggle('active', p.getAttribute('data-tab-index') === String(i));
    });
  }
  radios.forEach(function(r) {
    r.addEventListener('change', function() { showTab(this.value); });
  });
  showTab(0);
})();
      </script>
    </section>
""")
        # 5. Radar
        radial_html = None
        for _al, _rh in radar_by_agg:
            if _al == agg_fmt:
                radial_html = _rh
                break
        if radial_html:
            out.append("""
    <section class="section">
      <h2 class="section-title">Radar comparison (by location Dep/Arr 97.5%)</h2>
""" + _html_radar_chart_wrap(radial_html) + """
    </section>
""")
        # 6. Cumulative (if this agg has Both)
        cumulative_charts = None
        for _tab_label, _charts in cumulative_tabs:
            if _tab_label.startswith(agg_fmt) or (agg or "") in _tab_label:
                cumulative_charts = _charts
                break
        if cumulative_charts:
            out.append("""
    <section class="section">
      <h2 class="section-title">Cumulative comparison (by location)</h2>
      <p class="subsection-title">Both only. Net cumulative (Arr - Dep), same as Relocation Master &rarr; Cumulative profile.</p>
""")
            for chart_title, chart_html in cumulative_charts:
                out.append("""
      <div class="chart-wrap">
        <div class="chart-title">""" + chart_title + """</div>
        <div class="plotly-embed">""" + chart_html + """</div>
        <p class="chart-desc">Net cumulative (Arr - Dep). Same as Cumulative profile in Relocation Master. Same color per scenario.</p>
      </div>
""")
            out.append("    </section>\n")
        # 7. Scatter (Section only when there is content·Chart output, avoid blank frames)
        scatter_list, score_rows_agg = None, None
        for _al, _sl, _sr in scatter_by_agg:
            if _al == agg_fmt:
                scatter_list, score_rows_agg = _sl, _sr
                break
        if scatter_list is not None and len(scatter_list) > 0:
            out.append("""
    <section class="section">
      <h2 class="section-title">Dep 97.5% vs Arr 97.5% (by location)</h2>
""")
            for chart_title, chart_html in scatter_list:
                if not (chart_html and chart_html.strip()):
                    continue
                out.append(_html_scatter_chart_wrap(chart_title, chart_html, use_zone_desc=False))
            if score_rows_agg:
                out.append(_html_scatter_score_table(score_rows_agg, tns))
            out.append("    </section>\n")
        return "".join(out)

    if use_top_agg_tabs:
        html += """
    <div class="tabs-container" id="report_agg_tabs_container">
      <div class="tabs-header">
"""
        for idx, agg in enumerate(summary_agg_keys):
            agg_label = _format_agg_for_axis(agg) if (agg or "").strip() else "value"
            active = " active" if idx == 0 else ""
            html += f'        <button class="tab-btn{active}" data-tab="report_agg_panel_{idx}" data-tabs-container="report_agg_tabs_container">' + _html.escape(agg_label) + "</button>\n"
        html += """      </div>
"""
        for idx, agg in enumerate(summary_agg_keys):
            display = "block" if idx == 0 else "none"
            html += f'      <div class="tab-panel" id="report_agg_panel_{idx}" data-tabs-container="report_agg_tabs_container" style="display: {display};">\n'
            html += _render_single_agg_body(idx, agg)
            html += "      </div>\n"
        html += "    </div>\n"
    else:
        if len(summary_agg_keys) >= 2:
            html += """
    <section class="section">
      <h2 class="section-title">Scenario Summary</h2>
      <div class="tabs-container" id="summary-tabs-container">
        <div class="tabs-header">
"""
            for idx, agg in enumerate(summary_agg_keys):
                agg_label = _format_agg_for_axis(agg) if (agg or "").strip() else "value"
                active = " active" if idx == 0 else ""
                html += f'        <button class="tab-btn{active}" data-tab="summary-panel-{idx}" data-tabs-container="summary-tabs-container">' + _html.escape(agg_label) + "</button>\n"
            html += """        </div>
"""
            for idx, agg in enumerate(summary_agg_keys):
                unique_ids_agg, rep_summaries_agg = _unique_ids_and_reps_for_agg(agg)
                indices_agg = [unique_ids.index(sid) for sid in unique_ids_agg]
                colors_agg = [colors[i % len(colors)] for i in indices_agg]
                display = "block" if idx == 0 else "none"
                html += f'        <div class="tab-panel" id="summary-panel-{idx}" data-tabs-container="summary-tabs-container" style="display: {display};">\n'
                html += '      <div class="cards">\n'
                html += _render_summary_cards(unique_ids_agg, rep_summaries_agg, colors_agg, agg=agg)
                html += """      </div>
        </div>
"""
            html += """      </div>
    </section>
"""
        elif summary_agg_keys:
            html += """
    <section class="section">
      <h2 class="section-title">Scenario Summary</h2>
      <div class="cards">
"""
            first_agg = summary_agg_keys[0] if summary_agg_keys else None
            html += _render_summary_cards(unique_ids, rep_summaries, colors, agg=first_agg)
            html += """
      </div>
    </section>
"""

        # ——— Common Indicators (Non-tab layout) ———
        html += """
        <section class="section">
          <h2 class="section-title">Common Indicators (bar charts)</h2>
    """
        if common_indicators_by_agg:
            if len(common_indicators_by_agg) >= 2:
                html += '      <p class="subsection-title">Select tab by aggregate base. Dep/Arr are shown together in each chart.</p>\n'
                html += '      <div class="tabs-container" id="common_indicators_tabs_container">\n'
                html += '        <div class="tabs-header">\n'
                for i, item in enumerate(common_indicators_by_agg):
                    agg_label = item[0]
                    active = " active" if i == 0 else ""
                    html += f'        <button class="tab-btn{active}" data-tab="common_panel_{i}" data-tabs-container="common_indicators_tabs_container">' + _html.escape(agg_label) + "</button>\n"
                html += "        </div>\n"
                for i, item in enumerate(common_indicators_by_agg):
                    html_97_5, html_95, html_excess = item[1], item[2], item[3]
                    html_delay = item[4] if len(item) > 4 else None
                    html_avg_delay = item[5] if len(item) > 5 else None
                    html_avg_seat = item[6] if len(item) > 6 else None
                    panel_id = f"common_panel_{i}"
                    display = "block" if i == 0 else "none"
                    html += f'      <div class="tab-panel" id="{panel_id}" data-tabs-container="common_indicators_tabs_container" style="display: {display};">\n'
                    if html_97_5:
                        html += """
            <div class="chart-wrap">
              <div class="chart-title">Dep / Arr (97.5%)</div>
              <div class="plotly-embed">""" + html_97_5 + """</div>
              <p class="chart-desc">Same color per scenario. Dep/Arr/Both × locations: distinct patterns (solid, slash, dots, backslash, x, etc.). Red dashed line: Capacity.</p>
            </div>
    """
                    if html_95:
                        html += """
            <div class="chart-wrap">
              <div class="chart-title">Dep / Arr (95%)</div>
              <div class="plotly-embed">""" + html_95 + """</div>
              <p class="chart-desc">Same color per scenario. Dep/Arr/Both × locations: distinct patterns (solid, slash, dots, backslash, x, etc.). Red dashed line: Capacity.</p>
            </div>
    """
                    if html_excess:
                        html += """
            <div class="chart-wrap">
              <div class="chart-title">Excess as % of total · By location × Dep/Arr/Both</div>
              <div class="plotly-embed">""" + html_excess + """</div>
              <p class="chart-desc">Same color per scenario. Dep/Arr/Both × locations: distinct patterns for each combination.</p>
            </div>
    """
                    if html_delay:
                        html += """
            <div class="chart-wrap">
              <div class="chart-title">Total Delay Hours (Capacity) · By location × Dep/Arr/Both</div>
              <div class="plotly-embed">""" + html_delay + """</div>
              <p class="chart-desc">Total delay hours from capacity constraint (cascading overflow). Same color per scenario.</p>
            </div>
    """
                    if html_avg_delay:
                        html += """
            <div class="chart-wrap">
              <div class="chart-title">Avg Delay (per throughput) · By location × Dep/Arr/Both</div>
              <div class="plotly-embed">""" + html_avg_delay + """</div>
              <p class="chart-desc">Total Delay Hours ÷ Throughput. Average delay per unit. Same color per scenario.</p>
            </div>
    """
                    if html_avg_seat:
                        html += """
            <div class="chart-wrap">
              <div class="chart-title">Avg Seat · By location × Dep/Arr/Both</div>
              <div class="plotly-embed">""" + html_avg_seat + """</div>
              <p class="chart-desc">Total seats ÷ Total flights. Independent of aggregate base. Same color per scenario.</p>
            </div>
    """
                    html += "      </div>\n"
                html += "      </div>\n"
            else:
                item0 = common_indicators_by_agg[0]
                agg_label, html_97_5, html_95, html_excess = item0[0], item0[1], item0[2], item0[3]
                html_delay = item0[4] if len(item0) > 4 else None
                html_avg_delay = item0[5] if len(item0) > 5 else None
                html_avg_seat = item0[6] if len(item0) > 6 else None
                if html_97_5:
                    html += """
          <div class="chart-wrap">
            <div class="chart-title">Dep / Arr (97.5%)</div>
            <div class="plotly-embed">""" + html_97_5 + """</div>
            <p class="chart-desc">Same color per scenario. Dep/Arr/Both × locations: distinct patterns (solid, slash, dots, backslash, x, etc.). Red dashed line: Capacity.</p>
          </div>
    """
                if html_95:
                    html += """
          <div class="chart-wrap">
            <div class="chart-title">Dep / Arr (95%)</div>
            <div class="plotly-embed">""" + html_95 + """</div>
            <p class="chart-desc">Same color per scenario. Dep/Arr/Both × locations: distinct patterns (solid, slash, dots, backslash, x, etc.). Red dashed line: Capacity.</p>
          </div>
    """
                if html_excess:
                    html += """
          <div class="chart-wrap">
            <div class="chart-title">Excess as % of total · By location × Dep/Arr/Both</div>
            <div class="plotly-embed">""" + html_excess + """</div>
            <p class="chart-desc">Same color per scenario. Dep/Arr/Both × locations: distinct patterns for each combination.</p>
          </div>
    """
                if html_delay:
                    html += """
          <div class="chart-wrap">
            <div class="chart-title">Total Delay Hours (Capacity) · By location × Dep/Arr/Both</div>
            <div class="plotly-embed">""" + html_delay + """</div>
            <p class="chart-desc">Total delay hours from capacity constraint (cascading overflow). Same color per scenario.</p>
          </div>
    """
                if html_avg_delay:
                    html += """
          <div class="chart-wrap">
            <div class="chart-title">Avg Delay (per throughput) · By location × Dep/Arr/Both</div>
            <div class="plotly-embed">""" + html_avg_delay + """</div>
            <p class="chart-desc">Total Delay Hours ÷ Throughput. Average delay per unit. Same color per scenario.</p>
          </div>
    """
                if html_avg_seat:
                    html += """
          <div class="chart-wrap">
            <div class="chart-title">Avg Seat · By location × Dep/Arr/Both</div>
            <div class="plotly-embed">""" + html_avg_seat + """</div>
            <p class="chart-desc">Total seats ÷ Total flights. Independent of aggregate base. Same color per scenario.</p>
          </div>
    """
        else:
            if plotly_dep_arr_97_5:
                html += """
          <div class="chart-wrap">
            <div class="chart-title">Dep / Arr (97.5%)</div>
            <div class="plotly-embed">""" + plotly_dep_arr_97_5 + """</div>
            <p class="chart-desc">Same color per scenario. Dep/Arr/Both × locations: distinct patterns (solid, slash, dots, backslash, x, etc.). Red dashed line: Capacity.</p>
          </div>
    """
            if plotly_dep_arr_95:
                html += """
          <div class="chart-wrap">
            <div class="chart-title">Dep / Arr (95%)</div>
            <div class="plotly-embed">""" + plotly_dep_arr_95 + """</div>
            <p class="chart-desc">Same color per scenario. Dep/Arr/Both × locations: distinct patterns (solid, slash, dots, backslash, x, etc.). Red dashed line: Capacity.</p>
          </div>
    """
            if plotly_excess_grouped or (plotly_charts and plotly_charts.get("excess")):
                html += """
          <div class="chart-wrap">
            <div class="chart-title">Excess as % of total · By location × Dep/Arr/Both</div>
            <div class="plotly-embed">""" + (plotly_excess_grouped or (plotly_charts.get("excess") if plotly_charts else "")) + """</div>
            <p class="chart-desc">Same color per scenario. Dep/Arr/Both × locations: distinct patterns for each combination.</p>
          </div>
    """
            if plotly_delay_grouped:
                html += """
          <div class="chart-wrap">
            <div class="chart-title">Total Delay Hours (Capacity) · By location × Dep/Arr/Both</div>
            <div class="plotly-embed">""" + plotly_delay_grouped + """</div>
            <p class="chart-desc">Total delay hours from capacity constraint (cascading overflow). Same color per scenario.</p>
          </div>
    """
            if plotly_avg_delay_grouped:
                html += """
          <div class="chart-wrap">
            <div class="chart-title">Avg Delay (per throughput) · By location × Dep/Arr/Both</div>
            <div class="plotly-embed">""" + plotly_avg_delay_grouped + """</div>
            <p class="chart-desc">Total Delay Hours ÷ Throughput. Average delay per unit. Same color per scenario.</p>
          </div>
    """
            if plotly_avg_seat_grouped:
                html += """
          <div class="chart-wrap">
            <div class="chart-title">Avg Seat · By location × Dep/Arr/Both</div>
            <div class="plotly-embed">""" + plotly_avg_seat_grouped + """</div>
            <p class="chart-desc">Total seats ÷ Total flights. Independent of aggregate base. Same color per scenario.</p>
          </div>
    """
        html += "        </section>\n"

        # ——— Time Series: aggstar tab, each agg my location·Dep/Arrstar panel ———
        if time_series_by_agg:
            html += """
    <section class="section">
      <h2 class="section-title">Time Series</h2>
      <p class="subsection-title">Daily/time-slot tables and charts. Select tab by aggregate base, then location and Dep/Arr.</p>
"""
            if len(time_series_by_agg) >= 2:
                html += '      <div class="tabs-container" id="time_series_agg_tabs_container">\n'
                html += '        <div class="tabs-header">\n'
                for idx, (agg, agg_label, panels) in enumerate(time_series_by_agg):
                    active = " active" if idx == 0 else ""
                    html += f'        <button class="tab-btn{active}" data-tab="ts_agg_panel_{idx}" data-tabs-container="time_series_agg_tabs_container">' + _html.escape(agg_label) + "</button>\n"
                html += "        </div>\n"
                for idx, (agg, agg_label, scenario_list) in enumerate(time_series_by_agg):
                    display = "block" if idx == 0 else "none"
                    html += f'      <div class="tab-panel" id="ts_agg_panel_{idx}" data-tabs-container="time_series_agg_tabs_container" style="display: {display};">\n'
                    html += '      <p class="subsection-title">Select scenario, then location and Dep/Arr.</p>\n'
                    html += '      <div class="sorted-tabs-row" id="ts_flat_scenario_tabs_' + str(idx) + '">\n'
                    for si, (sid, label, panels) in enumerate(scenario_list):
                        tid = f"ts_flat_sc_tab_{idx}_{si}"
                        checked = " checked" if si == 0 else ""
                        html += f'        <input type="radio" name="ts_flat_scenario_{idx}" id="{tid}" value="{si}"{checked}><label class="sorted-tab-label" for="{tid}">{_html.escape(label)}</label>\n'
                    html += "      </div>\n"
                    html += '      <div class="ts-scenario-panels-wrap" id="ts_flat_sc_wrap_' + str(idx) + '">\n'
                    for si, (sid, label, panels) in enumerate(scenario_list):
                        sc_panel_id = f"ts_flat_sc_panel_{idx}_{si}"
                        sc_hidden = " ts-sc-hidden" if si != 0 else ""
                        html += f'      <div class="ts-scenario-panel{sc_hidden}" id="{sc_panel_id}" data-ts-flat-sc="{si}" data-ts-flat-agg="{idx}">\n'
                        html += '      <div class="sorted-tabs-row" id="ts_flat_term_tabs_' + str(idx) + '_' + str(si) + '">\n'
                        for ti, (term, dep_arr, _ta, _ba, _st, _sb) in enumerate(panels):
                            tid = f"ts_flat_tab_{idx}_{si}_{ti}"
                            checked = " checked" if ti == 0 else ""
                            html += f'        <input type="radio" name="ts_flat_term_{idx}_{si}" id="{tid}" value="{ti}"{checked}><label class="sorted-tab-label" for="{tid}">{_html.escape(term)} · {_html.escape(dep_arr)}</label>\n'
                        html += "      </div>\n"
                        for ti, (term, dep_arr, table_html, bar_html, stacked_html, summary_table_html) in enumerate(panels):
                            panel_id = f"ts_flat_panel_{idx}_{si}_{ti}"
                            active_inner = " active" if ti == 0 else ""
                            html += f'      <div class="sorted-tab-panel{active_inner}" id="{panel_id}" data-tab-index="{ti}" data-ts-flat-sc="{si}" data-ts-flat-agg="{idx}">\n'
                            if table_html:
                                html += '        <h3 class="section-subtitle">Daily Time Slot Data Table</h3>\n        <div class="chart-wrap">' + table_html + "</div>\n"
                            if bar_html:
                                html += '        <h3 class="section-subtitle">Daily Time Slot Bar Chart</h3>\n        <div class="chart-wrap"><div class="plotly-embed">' + bar_html + "</div></div>\n"
                            if stacked_html:
                                html += '        <h3 class="section-subtitle">Time Slot by Group Stacked Bar Chart</h3>\n        <div class="chart-wrap"><div class="plotly-embed">' + stacked_html + "</div></div>\n"
                            if summary_table_html:
                                html += '        <h3 class="section-subtitle">Summary by Group</h3>\n        <div class="chart-wrap">' + summary_table_html + "</div>\n"
                            html += "      </div>\n"
                        html += "      </div>\n"
                    html += "      </div>\n"
                    html += """
      <script>
(function(){
  var idx = """ + str(idx) + """;
  var scRadios = document.querySelectorAll('input[name="ts_flat_scenario_' + idx + '"]');
  var scPanels = document.querySelectorAll('div[id^="ts_flat_sc_panel_' + idx + '_"]');
  function resizePlotlyIn(el) {
    if (typeof Plotly !== 'undefined' && el) {
      var divs = el.querySelectorAll('.plotly-graph-div, .js-plotly-plot');
      divs.forEach(function(d){
        try {
          if (Plotly.Plots && Plotly.Plots.resize) Plotly.Plots.resize(d);
          else if (Plotly.Plot && Plotly.Plot.resize) Plotly.Plot.resize(d);
          if (Plotly.relayout) Plotly.relayout(d, {});
        } catch(e){}
      });
    }
  }
  function showSc(i) {
    scPanels.forEach(function(p) {
      var match = p.getAttribute('data-ts-flat-sc') === String(i);
      p.classList.toggle('ts-sc-hidden', !match);
      if (match) {
        requestAnimationFrame(function(){
          setTimeout(function(){ resizePlotlyIn(p); }, 150);
        });
      }
    });
  }
  scRadios.forEach(function(r) {
    r.addEventListener('change', function() { showSc(this.value); });
  });
  var scenarioCount = """ + str(len(scenario_list)) + """;
  for (var si = 0; si < scenarioCount; si++) {
    (function(si){
      var radios = document.querySelectorAll('input[name="ts_flat_term_' + idx + '_' + si + '"]');
      var panels = document.querySelectorAll('.sorted-tab-panel[data-ts-flat-sc="' + si + '"][data-tab-index]');
      function showTerm(i) {
        panels.forEach(function(p) {
          var match = p.getAttribute('data-tab-index') === String(i);
          p.classList.toggle('active', match);
          if (match && typeof Plotly !== 'undefined') {
            requestAnimationFrame(function(){
              setTimeout(function(){
                p.querySelectorAll('.plotly-graph-div, .js-plotly-plot').forEach(function(d){
                  try {
                    if (Plotly.Plots && Plotly.Plots.resize) Plotly.Plots.resize(d);
                    else if (Plotly.Plot && Plotly.Plot.resize) Plotly.Plot.resize(d);
                    if (Plotly.relayout) Plotly.relayout(d, {});
                  } catch(e){}
                });
              }, 150);
            });
          }
        });
      }
      radios.forEach(function(r) {
        r.addEventListener('change', function() { showTerm(this.value); });
      });
    })(si);
  }
  showSc(0);
})();
      </script>
"""
                    html += "      </div>\n"
                html += "      </div>\n    </section>\n"
            else:
                agg, agg_label, scenario_list = time_series_by_agg[0]
                html += '      <p class="subsection-title">Select scenario, then location and Dep/Arr.</p>\n'
                html += '      <div class="sorted-tabs-row" id="time_series_tabs_single_sc">\n'
                for si, (sid, label, panels) in enumerate(scenario_list):
                    tid = f"ts_tab_single_sc_{si}"
                    checked = " checked" if si == 0 else ""
                    html += f'        <input type="radio" name="time_series_tabs_single_sc" id="{tid}" value="{si}"{checked}><label class="sorted-tab-label" for="{tid}">{_html.escape(label)}</label>\n'
                html += "      </div>\n"
                html += '      <div class="ts-scenario-panels-wrap" id="ts_single_sc_wrap">\n'
                for si, (sid, label, panels) in enumerate(scenario_list):
                    sc_hidden = " ts-sc-hidden" if si != 0 else ""
                    html += f'      <div class="ts-scenario-panel{sc_hidden}" id="ts_single_sc_panel_{si}" data-ts-single-sc-index="{si}">\n'
                    html += f'      <div class="sorted-tabs-row" id="time_series_tabs_single_term_{si}">\n'
                    for ti, (term, dep_arr, _ta, _ba, _st, _sb) in enumerate(panels):
                        tid = f"ts_tab_single_t_{si}_{ti}"
                        checked = " checked" if ti == 0 else ""
                        html += f'        <input type="radio" name="time_series_tabs_single_term_{si}" id="{tid}" value="{ti}"{checked}><label class="sorted-tab-label" for="{tid}">{_html.escape(term)} · {_html.escape(dep_arr)}</label>\n'
                    html += "      </div>\n"
                    for ti, (term, dep_arr, table_html, bar_html, stacked_html, summary_table_html) in enumerate(panels):
                        panel_id = f"ts_panel_single_{si}_{ti}"
                        active = " active" if ti == 0 else ""
                        html += f'      <div class="sorted-tab-panel{active}" id="{panel_id}" data-tab-index="{ti}" data-ts-single-sc="{si}">\n'
                        if table_html:
                            html += '        <h3 class="section-subtitle">Daily Time Slot Data Table</h3>\n        <div class="chart-wrap">' + table_html + "</div>\n"
                        if bar_html:
                            html += '        <h3 class="section-subtitle">Daily Time Slot Bar Chart</h3>\n        <div class="chart-wrap"><div class="plotly-embed">' + bar_html + "</div></div>\n"
                        if stacked_html:
                            html += '        <h3 class="section-subtitle">Time Slot by Group Stacked Bar Chart</h3>\n        <div class="chart-wrap"><div class="plotly-embed">' + stacked_html + "</div></div>\n"
                        if summary_table_html:
                            html += '        <h3 class="section-subtitle">Summary by Group</h3>\n        <div class="chart-wrap">' + summary_table_html + "</div>\n"
                        html += "      </div>\n"
                    html += "      </div>\n"
                html += "      </div>\n"
                html += """
      <script>
(function(){
  var scRadios = document.querySelectorAll('input[name="time_series_tabs_single_sc"]');
  var scPanels = document.querySelectorAll('div[id^="ts_single_sc_panel_"]');
  function resizePlotlyIn(el) {
    if (typeof Plotly !== 'undefined' && el) {
      var divs = el.querySelectorAll('.plotly-graph-div, .js-plotly-plot');
      divs.forEach(function(d){
        try {
          if (Plotly.Plots && Plotly.Plots.resize) Plotly.Plots.resize(d);
          else if (Plotly.Plot && Plotly.Plot.resize) Plotly.Plot.resize(d);
          if (Plotly.relayout) Plotly.relayout(d, {});
        } catch(e){}
      });
    }
  }
  function showSc(i) {
    scPanels.forEach(function(p) {
      var match = p.getAttribute('data-ts-single-sc-index') === String(i);
      p.classList.toggle('ts-sc-hidden', !match);
      if (match) {
        requestAnimationFrame(function(){
          setTimeout(function(){ resizePlotlyIn(p); }, 150);
        });
      }
    });
  }
  scRadios.forEach(function(r) {
    r.addEventListener('change', function() { showSc(this.value); });
  });
  var scenarioCount = """ + str(len(scenario_list)) + """;
  for (var si = 0; si < scenarioCount; si++) {
    (function(si){
      var radios = document.querySelectorAll('input[name="time_series_tabs_single_term_' + si + '"]');
      var panels = document.querySelectorAll('.sorted-tab-panel[data-ts-single-sc="' + si + '"]');
      function showTerm(i) {
        panels.forEach(function(p) {
          var match = p.getAttribute('data-tab-index') === String(i);
          p.classList.toggle('active', match);
          if (match && typeof Plotly !== 'undefined') {
            requestAnimationFrame(function(){
              setTimeout(function(){
                p.querySelectorAll('.plotly-graph-div, .js-plotly-plot').forEach(function(d){
                  try {
                    if (Plotly.Plots && Plotly.Plots.resize) Plotly.Plots.resize(d);
                    else if (Plotly.Plot && Plotly.Plot.resize) Plotly.Plot.resize(d);
                    if (Plotly.relayout) Plotly.relayout(d, {});
                  } catch(e){}
                });
              }, 150);
            });
          }
        });
      }
      radios.forEach(function(r) {
        r.addEventListener('change', function() { showTerm(this.value); });
      });
    })(si);
  }
  showSc(0);
})();
      </script>
    </section>
"""

        # ——— Sorted Values: aggstar tab, each agg my Dep/Arr/Both tab ———
        if len(sorted_values_by_agg) >= 2:
            html += """
    <section class="section">
      <h2 class="section-title">Sorted Values (by rank) · By location and Dep/Arr</h2>
      <p class="subsection-title">Select tab by aggregate base, then Dep/Arr/Both.</p>
      <div class="tabs-container" id="sorted_values_tabs_container">
        <div class="tabs-header">
"""
            for idx, (agg, agg_label, items_agg) in enumerate(sorted_values_by_agg):
                active = " active" if idx == 0 else ""
                html += f'        <button class="tab-btn{active}" data-tab="sorted_values_panel_{idx}" data-tabs-container="sorted_values_tabs_container">' + _html.escape(agg_label) + "</button>\n"
            html += "        </div>\n"
            for idx, (agg, agg_label, items_agg) in enumerate(sorted_values_by_agg):
                display = "block" if idx == 0 else "none"
                html += f'        <div class="tab-panel" id="sorted_values_panel_{idx}" data-tabs-container="sorted_values_tabs_container" style="display: {display};">\n'
            from collections import defaultdict
            by_tab = defaultdict(list)
            for item in items_agg:
                if len(item) == 4:
                    a_l, dep_arr, chart_title, chart_html = item
                    by_tab[(a_l, dep_arr)].append((chart_title, chart_html))
                else:
                    chart_title, chart_html = item[0], item[1]
                    by_tab[("value", "Dep")].append((chart_title, chart_html))
            tab_keys = sorted(by_tab.keys(), key=lambda k: (k[0], 0 if (k[1] or "Dep").upper() == "DEP" else 1, k[1] or "Dep"))
            tab_keys = [(a, d or "Dep") for a, d in tab_keys]
            html += '      <div class="sorted-tabs-row" id="sorted_tabs_container_%d">\n' % idx
            for i, (a_l, dep_arr) in enumerate(tab_keys):
                tid = f"sorted_tab_{idx}_{i}"
                checked = " checked" if i == 0 else ""
                tab_caption = f"{a_l} · {dep_arr}"
                html += f'        <input type="radio" name="sorted_tabs_{idx}" id="{tid}" value="{i}"{checked}><label class="sorted-tab-label" for="{tid}">{_html.escape(tab_caption)}</label>\n'
            html += "      </div>\n"
            for i, (a_l, dep_arr) in enumerate(tab_keys):
                panel_id = f"sorted_panel_{idx}_{i}"
                active_inner = " active" if i == 0 else ""
                html += f'      <div class="sorted-tab-panel{active_inner}" id="{panel_id}" data-tab-index="{i}" data-sorted-outer="{idx}">\n'
                for chart_title, chart_html in by_tab[(a_l, dep_arr)]:
                    html += """
        <div class="chart-wrap">
          <div class="plotly-embed">""" + chart_html + """</div>
          <p class="chart-desc">Horizontal red line: Capacity. Light red markers: scenario values at max / 97.5% / 95% / 90%.</p>
        </div>
"""
                html += "      </div>\n"
            html += """
      <script>
(function(){
  var idx = """ + str(idx) + """;
  var c = document.getElementById('sorted_tabs_container_' + idx);
  if (!c) return;
  var radios = c.querySelectorAll('input[name="sorted_tabs_' + idx + '"]');
  var panels = document.querySelectorAll('.sorted-tab-panel[data-sorted-outer="' + idx + '"]');
  function showTab(i) {
    panels.forEach(function(p) {
      p.classList.toggle('active', p.getAttribute('data-tab-index') === String(i));
    });
  }
  radios.forEach(function(r) {
    r.addEventListener('change', function() { showTab(this.value); });
  });
  showTab(0);
})();
      </script>
"""
            html += "        </div>\n"
            html += "      </div>\n    </section>\n"
        elif len(sorted_values_by_agg) == 1:
            agg, agg_label, items_agg = sorted_values_by_agg[0]
            from collections import defaultdict
            by_tab = defaultdict(list)
            for item in items_agg:
                if len(item) == 4:
                    a_l, dep_arr, chart_title, chart_html = item
                    by_tab[(a_l, dep_arr)].append((chart_title, chart_html))
                else:
                    chart_title, chart_html = item[0], item[1]
                    by_tab[("value", "Dep")].append((chart_title, chart_html))
            tab_keys = sorted(by_tab.keys(), key=lambda k: (k[0], 0 if (k[1] or "Dep").upper() == "DEP" else 1, k[1] or "Dep"))
            tab_keys = [(a, d or "Dep") for a, d in tab_keys]
            html += """
    <section class="section">
      <h2 class="section-title">Sorted Values (by rank) · By location and Dep/Arr</h2>
      <p class="subsection-title">Select tab by Dep/Arr/Both.</p>
      <div class="sorted-tabs-row" id="sorted_tabs_container">
"""
            for i, (a_l, dep_arr) in enumerate(tab_keys):
                tid = f"sorted_tab_{i}"
                checked = " checked" if i == 0 else ""
                tab_caption = f"{a_l} · {dep_arr}"
                html += f'        <input type="radio" name="sorted_tabs" id="{tid}" value="{i}"{checked}><label class="sorted-tab-label" for="{tid}">{_html.escape(tab_caption)}</label>\n'
            html += "      </div>\n"
            for i, (a_l, dep_arr) in enumerate(tab_keys):
                panel_id = f"sorted_panel_{i}"
                active = " active" if i == 0 else ""
                html += f'      <div class="sorted-tab-panel{active}" id="{panel_id}" data-tab-index="{i}">\n'
                for chart_title, chart_html in by_tab[(a_l, dep_arr)]:
                    html += """
            <div class="chart-wrap">
              <div class="plotly-embed">""" + chart_html + """</div>
              <p class="chart-desc">Horizontal red line: Capacity. Light red markers: scenario values at max / 97.5% / 95% / 90%.</p>
            </div>
    """
                html += "      </div>\n"
            html += """
          <script>
    (function(){
      var c = document.getElementById('sorted_tabs_container');
      if (!c) return;
      var radios = c.querySelectorAll('input[name="sorted_tabs"]');
      var panels = document.querySelectorAll('.sorted-tab-panel[data-tab-index]');
      function showTab(idx) {
        panels.forEach(function(p) {
          p.classList.toggle('active', p.getAttribute('data-tab-index') === String(idx));
        });
      }
      radios.forEach(function(r) {
        r.addEventListener('change', function() { showTab(this.value); });
      });
      showTab(0);
    })();
          </script>
        </section>
    """
        elif plotly_sorted_by_combo:
            from collections import defaultdict
            by_tab = defaultdict(list)
            for item in plotly_sorted_by_combo:
                if len(item) == 4:
                    agg_label, dep_arr, chart_title, chart_html = item
                    by_tab[(agg_label, dep_arr)].append((chart_title, chart_html))
                else:
                    chart_title, chart_html = item[0], item[1]
                    by_tab[("value", "Dep")].append((chart_title, chart_html))
            tab_keys = sorted(by_tab.keys(), key=lambda k: (k[0], 0 if (k[1] or "Dep").upper() == "DEP" else 1))
            tab_keys = [(a, d or "Dep") for a, d in tab_keys]
            html += """
        <section class="section">
          <h2 class="section-title">Sorted Values (by rank) · By location and Dep/Arr</h2>
          <p class="subsection-title">Select tab by aggregate and Dep/Arr.</p>
          <div class="sorted-tabs-row" id="sorted_tabs_container">
    """
            for i, (agg_label, dep_arr) in enumerate(tab_keys):
                tid = f"sorted_tab_{i}"
                checked = " checked" if i == 0 else ""
                tab_caption = f"{agg_label} · {dep_arr}"
                html += f'        <input type="radio" name="sorted_tabs" id="{tid}" value="{i}"{checked}><label class="sorted-tab-label" for="{tid}">{_html.escape(tab_caption)}</label>\n'
            html += "      </div>\n"
            for i, (agg_label, dep_arr) in enumerate(tab_keys):
                panel_id = f"sorted_panel_{i}"
                active = " active" if i == 0 else ""
                html += f'      <div class="sorted-tab-panel{active}" id="{panel_id}" data-tab-index="{i}">\n'
                for chart_title, chart_html in by_tab[(agg_label, dep_arr)]:
                    html += """
            <div class="chart-wrap">
              <div class="plotly-embed">""" + chart_html + """</div>
              <p class="chart-desc">Horizontal red line: Capacity. Light red markers: scenario values at max / 97.5% / 95% / 90%.</p>
            </div>
    """
                html += "      </div>\n"
            html += """
          <script>
    (function(){
      var c = document.getElementById('sorted_tabs_container');
      if (!c) return;
      var radios = c.querySelectorAll('input[name="sorted_tabs"]');
      var panels = document.querySelectorAll('.sorted-tab-panel[data-tab-index]');
      function showTab(idx) {
        panels.forEach(function(p) {
          p.classList.toggle('active', p.getAttribute('data-tab-index') === String(idx));
        });
      }
      radios.forEach(function(r) {
        r.addEventListener('change', function() { showTab(this.value); });
      });
      showTab(0);
    })();
          </script>
        </section>
    """
    # Radial comparison: aggstar tab (top movement|seat When using tabs, they are already inside each panel, so they are omitted here.)
    if not use_top_agg_tabs and radar_by_agg:
        html += """
    <section class="section">
      <h2 class="section-title">Radar comparison (by location Dep/Arr 97.5%)</h2>
"""
        if len(radar_by_agg) >= 2:
            html += '      <p class="subsection-title">Select tab by aggregate base.</p>\n'
            html += '      <div class="tabs-container" id="radar_tabs_container">\n'
            html += '        <div class="tabs-header">\n'
            for i, (agg_label, _) in enumerate(radar_by_agg):
                active = " active" if i == 0 else ""
                html += f'        <button class="tab-btn{active}" data-tab="radar_panel_{i}" data-tabs-container="radar_tabs_container">' + _html.escape(agg_label) + "</button>\n"
            html += "        </div>\n"
            for i, (agg_label, radial_html) in enumerate(radar_by_agg):
                panel_id = f"radar_panel_{i}"
                display = "block" if i == 0 else "none"
                html += f'      <div class="tab-panel" id="{panel_id}" data-tabs-container="radar_tabs_container" style="display: {display};">\n'
                html += """
        <div class="chart-wrap">
          <div class="plotly-embed">""" + radial_html + """</div>
          <p class="chart-desc">Location Dep/Arr 97.5% peak only. No fill. Red dashed line: Capacity.</p>
        </div>
      </div>
"""
            html += "      </div>\n"
        else:
            _, radial_html = radar_by_agg[0]
            html += """
      <div class="chart-wrap">
        <div class="plotly-embed">""" + radial_html + """</div>
        <p class="chart-desc">Location Dep/Arr 97.5% peak only. No fill. Red dashed line: Capacity.</p>
      </div>
"""
        html += "    </section>\n"
    elif not use_top_agg_tabs and plotly_radial_dep_arr_97:
        html += """
    <section class="section">
      <h2 class="section-title">Radar comparison (by location Dep/Arr 97.5%)</h2>
      <div class="chart-wrap">
        <div class="plotly-embed">""" + plotly_radial_dep_arr_97 + """</div>
        <p class="chart-desc">Location Dep/Arr 97.5% peak only. No fill. Red dashed line: Capacity.</p>
      </div>
    </section>
"""

    # Cumulative comparison: BothOnly when there is (top agg When using tabs, they are omitted as they are within each panel.)
    if not use_top_agg_tabs and cumulative_tabs:
        html += """
    <section class="section">
      <h2 class="section-title">Cumulative comparison (by location)</h2>
      <p class="subsection-title">Both only. Net cumulative (Arr - Dep), same as Relocation Master &rarr; Cumulative profile. Select tab by aggregate base.</p>
      <div class="sorted-tabs-row" id="cumulative_tabs_container">
"""
        for i, (tab_label, _) in enumerate(cumulative_tabs):
            tid = f"cumulative_tab_{i}"
            checked = " checked" if i == 0 else ""
            html += f'        <input type="radio" name="cumulative_tabs" id="{tid}" value="{i}"{checked}><label class="sorted-tab-label" for="{tid}">{_html.escape(tab_label)}</label>\n'
        html += "      </div>\n"
        for i, (tab_label, charts) in enumerate(cumulative_tabs):
            panel_id = f"cumulative_panel_{i}"
            active = " active" if i == 0 else ""
            html += f'      <div class="sorted-tab-panel{active}" id="{panel_id}" data-tab-index="{i}">\n'
            for chart_title, chart_html in charts:
                html += """
        <div class="chart-wrap">
          <div class="chart-title">""" + chart_title + """</div>
          <div class="plotly-embed">""" + chart_html + """</div>
          <p class="chart-desc">Net cumulative (Arr - Dep). Same as Cumulative profile in Relocation Master. Same color per scenario.</p>
        </div>
"""
            html += "      </div>\n"
        html += """
      <script>
(function(){
  var c = document.getElementById('cumulative_tabs_container');
  if (!c) return;
  var radios = c.querySelectorAll('input[name="cumulative_tabs"]');
  var panels = document.querySelectorAll('.sorted-tab-panel[id^="cumulative_panel_"]');
  function showTab(idx) {
    panels.forEach(function(p) {
      p.classList.toggle('active', p.getAttribute('data-tab-index') === String(idx));
    });
  }
  radios.forEach(function(r) {
    r.addEventListener('change', function() { showTab(this.value); });
  });
  showTab(0);
})();
      </script>
    </section>
"""

    # Dep vs Arr 97.5% scatter per terminal — aggstar tab (top agg When using tabs, they are omitted as they are within each panel.)
    if not use_top_agg_tabs and (scatter_by_agg or plotly_dep_arr_scatter_by_terminal):
        html += """
    <section class="section">
      <h2 class="section-title">Dep 97.5% vs Arr 97.5% (by location)</h2>
"""
        if len(scatter_by_agg) >= 2:
            html += """
      <div class="tabs-container" id="scatter-tabs-container">
        <div class="tabs-header">
"""
            for idx, (agg_label, _scatter_list, _score_rows) in enumerate(scatter_by_agg):
                active = " active" if idx == 0 else ""
                html += f"""
          <button class="tab-btn{active}" data-tab="scatter-panel-{idx}" data-tabs-container="scatter-tabs-container">""" + _html.escape(agg_label) + """</button>
"""
            html += """
        </div>
"""
            for idx, (agg_label, scatter_list, score_rows_agg) in enumerate(scatter_by_agg):
                display = "block" if idx == 0 else "none"
                html += f"""
        <div class="tab-panel" id="scatter-panel-{idx}" data-tabs-container="scatter-tabs-container" style="display: """ + display + """;">
"""
                for chart_title, chart_html in scatter_list:
                    if not (chart_html and chart_html.strip()):
                        continue
                    html += _html_scatter_chart_wrap(chart_title, chart_html)
                if score_rows_agg:
                    tns = report_labels.get("terminal_names") or []
                    html += _html_scatter_score_table(score_rows_agg, tns)
                html += """
        </div>
"""
            html += """
      </div>
"""
        else:
            # single agg or fallback: One block without tabs
            if scatter_by_agg:
                _agg_label, scatter_list, score_rows_agg = scatter_by_agg[0]
                for chart_title, chart_html in scatter_list:
                    if not (chart_html and chart_html.strip()):
                        continue
                    html += _html_scatter_chart_wrap(chart_title, chart_html)
                if score_rows_agg:
                    tns = report_labels.get("terminal_names") or []
                    html += _html_scatter_score_table(score_rows_agg, tns)
            elif plotly_dep_arr_scatter_by_terminal:
                for chart_title, chart_html in plotly_dep_arr_scatter_by_terminal:
                    if not (chart_html and chart_html.strip()):
                        continue
                    html += _html_scatter_chart_wrap(chart_title, chart_html)
                if dep_arr_score_table:
                    tns = report_labels.get("terminal_names") or []
                    html += _html_scatter_score_table(dep_arr_score_table, tns)
        html += "    </section>\n"

    html += """
    <footer>
      Report generated: generate_scenario_report_rev2 — JSON to Plotly to HTML
    </footer>
  </div>
  <script>
(function(){
  document.querySelectorAll('.tabs-container').forEach(function(container){
    var id = container.id;
    if (!id) return;
    var btns = container.querySelectorAll('.tab-btn[data-tabs-container="' + id + '"]');
    var panels = document.querySelectorAll('.tab-panel[data-tabs-container="' + id + '"]');
    function resizePlotlyInPanel(panel) {
      if (typeof Plotly === 'undefined' || !panel) return;
      panel.querySelectorAll('.plotly-graph-div, .js-plotly-plot').forEach(function(el) {
        try {
          if (Plotly.Plots && Plotly.Plots.resize) Plotly.Plots.resize(el);
          else if (Plotly.Plot && Plotly.Plot.resize) Plotly.Plot.resize(el);
          if (Plotly.relayout) Plotly.relayout(el, {});
        } catch (e) {}
      });
    }
    btns.forEach(function(btn){
      btn.addEventListener('click', function(){
        var tabId = this.getAttribute('data-tab');
        btns.forEach(function(b){ b.classList.remove('active'); });
        panels.forEach(function(p){
          var show = p.id === tabId;
          p.style.display = show ? 'block' : 'none';
          if (show) requestAnimationFrame(function(){ resizePlotlyInPanel(p); });
        });
        this.classList.add('active');
      });
    });
  });
})();
  </script>
</body>
</html>
"""
    return html


def main():
    """CLI: By the Son of Man JSON Receives a path or doesn't have it Scenarios in folder *.json Generate report after loading."""
    import sys

    if len(sys.argv) >= 2:
        origin = sys.argv[1]
        scenarios = load_scenarios_from_json(origin)
    else:
        # Scenarios in folder JSON load files (No dependency on original modules)
        scenarios = []
        if SCENARIOS_DIR.exists():
            for path in sorted(SCENARIOS_DIR.glob("*.json")):
                scenarios.extend(load_scenarios_from_json(path))
    if not scenarios:
        print("Warning: No scenario data. Usage: python -m utils.generate_scenario_report_rev2 [path/file.json]")
    html = build_report_html(scenarios)
    out_dir = SCENARIOS_DIR if SCENARIOS_DIR.exists() else PROJECT_ROOT
    out_path = out_dir / "Scenario_Report_Rev2.html"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("Generated:", out_path)


if __name__ == "__main__":
    main()



# Materials taught to Claude
# this html Study the report and discover the insights you can find here., 
# this report html Adds insight into these graphs for each graph while maintaining the structure.
# Add comprehensive insights below!
# And this htmlWhen writing comprehensive insights based on , you can create additional graphs! However, it must be written well based on actual data.