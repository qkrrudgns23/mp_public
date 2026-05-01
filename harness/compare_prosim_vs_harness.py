"""
Compare wall time: last Layout ProSim run (NDJSON probe) vs same saved sim_input rerun with no progress_cb.

Usage after running Pro Sim from Streamlit / designer:
    python -m harness.compare_prosim_vs_harness

Optional:
    python -m harness.compare_prosim_vs_harness --input data/Result_storage/default_layout_sim_input.json
    python -m harness.compare_prosim_vs_harness --log debug-8ab4c9.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[1]
_RESULT_STORAGE = (_ROOT / "data" / "Result_storage").resolve()
_DEFAULT_LOG = (_ROOT / "debug-8ab4c9.log").resolve()


def _load_ndjson_records(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except json.JSONDecodeError:
            continue
    return rows


def _latest_prosim_thread_row(rows: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    picked: Optional[Dict[str, Any]] = None
    for r in rows:
        if (
            r.get("runId") == "prosim-thread"
            and r.get("hypothesisId") == "H1_H3_H4"
            and r.get("message") == "sim_core_vs_serialize_wall_sec"
        ):
            picked = r
    return picked


def _newest_sim_input_path(root: Path) -> Optional[Path]:
    """Most recently modified ``*_sim_input.json`` under ``Result_storage``."""
    cand: list[tuple[float, Path]] = []
    for p in root.glob("*_sim_input.json"):
        if not p.is_file():
            continue
        try:
            cand.append((p.stat().st_mtime, p.resolve()))
        except OSError:
            continue
    if not cand:
        return None
    cand.sort(key=lambda z: z[0], reverse=True)
    return cand[0][1]


def _stem_from_log_row(row: Optional[Dict[str, Any]]) -> Optional[str]:
    if row is None:
        return None
    data = row.get("data")
    if isinstance(data, dict):
        stem = data.get("resultStem")
        if stem is None:
            return None
        s = str(stem).strip()
        return s or None
    return None


def _utc_label(ts_unix: float) -> str:
    return datetime.fromtimestamp(ts_unix, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _append_compare_log(payload: Dict[str, Any]) -> None:
    # #region agent log
    row = dict(payload)
    row.setdefault("sessionId", "8ab4c9")
    row.setdefault("timestamp", int(time.time() * 1000))
    try:
        target = (_ROOT / "debug-8ab4c9.log").resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        default=str(_DEFAULT_LOG),
        help="NDJSON probe file (same as ProSim session log)",
    )
    parser.add_argument(
        "--input",
        default="",
        help="Explicit *_sim_input.json path; otherwise infer from last prosim-thread row stem, else newest file",
    )
    parser.add_argument("--dt", type=float, default=1.0, help="passed to run_simulation")
    parser.add_argument(
        "--no-append-log",
        action="store_true",
        help="Do not append H_CLI_COMPARE line to debug-8ab4c9.log",
    )
    args = parser.parse_args(argv)

    log_path = Path(args.log).expanduser().resolve()

    explicit = Path(args.input).expanduser().resolve() if args.input.strip() else None

    rows = _load_ndjson_records(log_path)
    prosim_row = _latest_prosim_thread_row(rows)
    stem_hint = _stem_from_log_row(prosim_row)

    sim_path: Optional[Path]
    mode: str
    if explicit and explicit.is_file():
        sim_path = explicit.resolve()
        mode = "explicit"
    elif stem_hint:
        cand = (_RESULT_STORAGE / f"{stem_hint}_sim_input.json").resolve()
        sim_path = cand if cand.is_file() else None
        mode = f"stem_from_log:{stem_hint}"
    else:
        sim_path = None
        mode = ""

    if sim_path is None or not sim_path.is_file():
        newest = _newest_sim_input_path(_RESULT_STORAGE)
        if newest:
            sim_path = newest.resolve()
            mode = "newest_mtime"
        else:
            print(
                "compare: no *_sim_input.json found under Result_storage.",
                file=sys.stderr,
            )
            return 2

    payload = json.loads(sim_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        print("compare: sim_input top-level must be object", file=sys.stderr)
        return 2

    from utils.airside_sim import run_simulation

    t0 = time.perf_counter()
    run_simulation(payload, dt=float(args.dt), progress_cb=None)
    harness_wall_sec = float(time.perf_counter() - t0)

    prosim_sec: Optional[float] = None
    if isinstance(prosim_row, dict):
        d = prosim_row.get("data")
        if isinstance(d, dict):
            try:
                prosim_sec = float(d["run_simulation_wall_sec"])  # type: ignore[arg-type]
            except (KeyError, TypeError, ValueError):
                prosim_sec = None

    ratio_txt = ""
    delta_txt = ""
    if prosim_sec is not None and prosim_sec > 1e-9:
        delta = prosim_sec - harness_wall_sec
        delta_txt = f"delta_vs_harness(+ProSim-progress): {delta:+.4f}s"
        ratio_txt = f"ratio(ProSim/harness_same_input): {prosim_sec / harness_wall_sec:.3f}x"

    sim_mtime_sec = float(sim_path.stat().st_mtime)

    print("=== ProSim vs harness (same persisted sim_input) ===")
    print(f"input_path={sim_path}")
    print(f"input_resolution_mode={mode}")
    print(f"sim_input_mtime_utc={_utc_label(sim_mtime_sec)}")
    if isinstance(prosim_row, dict):
        pts = prosim_row.get("timestamp")
        if isinstance(pts, (int, float)) and float(pts) > 0:
            log_sec = float(pts) / 1000.0
            skew = abs(sim_mtime_sec - log_sec)
            print(f"last_prosim_log_ts_utc={_utc_label(log_sec)}")
            print(f"log_vs_input_mtime_skew_sec={skew:.3f}")
            if skew > 120.0:
                print(
                    "WARN: skew>120s — last H1_H3_H4 line may not belong to this "
                    "sim_input file. Clear debug-8ab4c9.log, run Pro Sim once, "
                    "then rerun this script.",
                    file=sys.stderr,
                )
    print(f"harness_wall_sec(progress_cb=None, dt={args.dt})={harness_wall_sec:.4f}")
    if prosim_sec is None:
        print(
            "prosim_run_sim_wall_sec=(missing H1_H3_H4 prosim-thread line in "
            + str(log_path)
            + "); run Layout ProSim once after clearing log."
        )
    else:
        print(f"prosim_run_sim_wall_sec(last log line, same stem)={prosim_sec:.4f}")
        print(delta_txt or "(no harness delta)")
        print(ratio_txt or "(no ratio)")

    if not args.no_append_log:
        _append_compare_log(
            {
                "runId": "compare-cli",
                "hypothesisId": "H_CLI_COMPARE",
                "location": "harness/compare_prosim_vs_harness.py:main",
                "message": "shadow_run_same_sim_input_wall_sec",
                "data": {
                    "simInputPath": str(sim_path.relative_to(_ROOT))
                    if sim_path.is_relative_to(_ROOT)
                    else str(sim_path),
                    "inputResolutionMode": mode,
                    "harness_wall_sec_same_input_progress_cb_none": round(
                        harness_wall_sec, 6
                    ),
                    "logPath": str(log_path.relative_to(_ROOT))
                    if log_path.is_relative_to(_ROOT)
                    else str(log_path),
                    **(
                        {"prosim_thread_run_sim_wall_sec": round(prosim_sec, 6)}
                        if prosim_sec is not None
                        else {}
                    ),
                    **(
                        {
                            "prosim_vs_harness_ratio": round(
                                prosim_sec / harness_wall_sec, 6
                            )
                        }
                        if prosim_sec is not None and harness_wall_sec > 1e-9
                        else {}
                    ),
                },
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
