"""
Relocation Master Save results/Load Utility.
Run analysis the result JSONSave it as , and load and display it without re-analysis..
"""
import io
import json
import os
import re
from datetime import datetime
import pandas as pd


STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "utils", "result_storage", "saved")
os.makedirs(STORAGE_DIR, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    """Remove characters that cannot be used in file names"""
    s = re.sub(r'[<>:"/\\|?*]', "_", str(name).strip())
    return s[:200] if s else "unnamed"


def _get_datetime_cols(df: pd.DataFrame) -> list:
    """datetime64, timedelta64 Return type column name list"""
    if df is None or len(df) == 0:
        return []
    cols = []
    for c in df.columns:
        dtype = str(df[c].dtype)
        if dtype.startswith("datetime64") or dtype.startswith("timedelta64"):
            cols.append(c)
    return cols


def _serialize_df(df: pd.DataFrame, datetime_cols: list = None) -> tuple[str, list]:
    """DataFramesecond JSON serialize to string. (json_str, datetime_cols) return."""
    if df is None or len(df) == 0:
        return json.dumps({"columns": [], "index": [], "data": []}), []
    if datetime_cols is None:
        datetime_cols = _get_datetime_cols(df)
    s = df.to_json(orient="split", date_format="iso", force_ascii=False)
    return s, datetime_cols


def _deserialize_df(val, datetime_cols: list = None) -> pd.DataFrame:
    """JSON string or dictat DataFrame restore. datetime_colsIf there is, the corresponding column datetimeconvert to."""
    if val is None:
        return pd.DataFrame()
    if isinstance(val, pd.DataFrame):
        return val
    if isinstance(val, dict) and "data" in val:
        df = pd.DataFrame(val["data"], columns=val.get("columns", []), index=val.get("index"))
    else:
        s = str(val).strip() if val else ""
        if not s:
            return pd.DataFrame()
        try:
            df = pd.read_json(io.StringIO(s), orient="split")
        except Exception:
            try:
                d = json.loads(s) if isinstance(val, str) else val
                if isinstance(d, dict) and "data" in d:
                    df = pd.DataFrame(d["data"], columns=d.get("columns", []), index=d.get("index"))
                else:
                    return pd.DataFrame()
            except Exception:
                return pd.DataFrame()
    # datetime/timedelta Column restoration (JSON round-trip Poetry string remains)
    if datetime_cols:
        for col in datetime_cols:
            if col in df.columns and df[col].notna().any():
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass
    # When saving datetime_cols Compared to old files loaded without: scheduled_gate_local Force conversion of frequently used columns, etc.
    _known_datetime_cols = ["scheduled_gate_local", "SHOW", "Time"]
    for col in _known_datetime_cols:
        if col in df.columns and col not in (datetime_cols or []):
            if df[col].dtype == object or str(df[col].dtype) == "string":
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass
    return df


def save_result(analysis, name: str) -> str:
    """
    RunAnalysis the result JSONSave it as.
    Returns: Saved file path
    """
    name_safe = _sanitize_filename(name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name_safe}_{ts}.json"
    filepath = os.path.join(STORAGE_DIR, filename)

    res_json, res_dt_cols = _serialize_df(analysis.results_df)
    filt_json, filt_dt_cols = _serialize_df(analysis.df_filtered)
    payload = {
        "name": name,
        "saved_at": datetime.now().isoformat(),
        "results_df": res_json,
        "df_filtered": filt_json,
        "df_filtered_datetime_cols": filt_dt_cols,
        "results_df_datetime_cols": res_dt_cols,
        "relocation_unit": analysis.relocation_unit,
        "fixed_per_loc": analysis.fixed_per_loc,
        "moving_unit": analysis.moving_unit,
        "allowed_targets_map": analysis.allowed_targets_map or {},
        "loc_names": list(analysis.loc_names) if analysis.loc_names else [],
        "unit_min": analysis.unit_min,
        "start_date": str(analysis.start_date) if analysis.start_date else None,
        "end_date": str(analysis.end_date) if analysis.end_date else None,
        "selected_metrics": list(analysis.selected_metrics) if analysis.selected_metrics else [],
        "selected_agg_col": analysis.selected_agg_col,
        "selected_agg_cols": list(analysis.selected_agg_cols) if getattr(analysis, "selected_agg_cols", None) else [analysis.selected_agg_col],
        "max_sample_count": getattr(analysis, "max_sample_count", None),
        "unassigned_units_map": getattr(analysis, "unassigned_units_map", None) or {},
        "analysis_count": getattr(analysis, "analysis_count", None),
        "total_combinations": getattr(analysis, "total_combinations", None),
        "is_sampled": getattr(analysis, "is_sampled", False),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return filepath


def load_result(filepath: str) -> dict:
    """
    saved JSONLoad the resulting data from.
    Returns: RunAnalysis.from_saved()forward to dict
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    filt_dt = data.get("df_filtered_datetime_cols") or []
    res_dt = data.get("results_df_datetime_cols") or []
    data["results_df"] = _deserialize_df(data.get("results_df", "{}"), datetime_cols=res_dt)
    data["df_filtered"] = _deserialize_df(data.get("df_filtered", "{}"), datetime_cols=filt_dt)
    return data


def list_saved_results() -> list[dict]:
    """Returns a list of saved result files (Latest)"""
    files = []
    for fname in os.listdir(STORAGE_DIR):
        if not fname.endswith(".json"):
            continue
        fp = os.path.join(STORAGE_DIR, fname)
        try:
            stat = os.stat(fp)
            with open(fp, "r", encoding="utf-8") as f:
                meta = json.load(f)
            name = meta.get("name", fname)
            saved_at = meta.get("saved_at", "")
            files.append({
                "filepath": fp,
                "filename": fname,
                "name": name,
                "saved_at": saved_at,
                "mtime": stat.st_mtime,
            })
        except Exception:
            continue
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return files


def delete_result(filepath: str) -> bool:
    """Delete saved result files. Returns: Success or failure"""
    if not filepath or not os.path.isfile(filepath):
        return False
    try:
        if os.path.dirname(os.path.abspath(filepath)) == os.path.abspath(STORAGE_DIR):
            os.remove(filepath)
            return True
    except Exception:
        pass
    return False
