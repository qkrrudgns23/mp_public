from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _preview(v: Any, max_len: int = 200) -> str:
    s = repr(v)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _path_to_str(path: List[str]) -> str:
    if not path:
        return "$"
    out = ""
    for part in path:
        if part.startswith("["):
            out += part
        elif out == "":
            out = part
        else:
            out += "." + part
    return out


def _leaf_numeric_tol_match(exp: Any, act: Any, float_rtol: float, float_atol: float) -> bool:
    """Leaf JSON numbers only (excluding bool): math.isclose when rtol/atol > 0."""

    if float_rtol <= 0.0 and float_atol <= 0.0:
        return False
    if type(exp) is bool or type(act) is bool:
        return False
    if type(exp) not in (int, float) or type(act) not in (int, float):
        return False
    try:
        return bool(
            math.isclose(
                float(exp),
                float(act),
                rel_tol=max(float_rtol, 0.0),
                abs_tol=max(float_atol, 0.0),
            )
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _walk_diff(
    exp: Any,
    act: Any,
    path: List[str],
    first: List[Optional[Tuple[List[str], Any, Any]]],
    total: List[int],
    max_total: int,
    *,
    float_rtol: float = 0.0,
    float_atol: float = 0.0,
) -> None:
    if total[0] >= max_total:
        return
    if exp == act:
        return
    if _leaf_numeric_tol_match(exp, act, float_rtol, float_atol):
        return

    if type(exp) is not type(act):
        if first[0] is None:
            first[0] = (list(path), exp, act)
        total[0] += 1
        return

    if isinstance(exp, dict):
        keys_e = set(exp.keys())
        keys_a = set(act.keys())
        for k in sorted(keys_e - keys_a):
            if total[0] >= max_total:
                return
            if first[0] is None:
                first[0] = (path + [str(k)], exp[k], None)
            total[0] += 1
        for k in sorted(keys_a - keys_e):
            if total[0] >= max_total:
                return
            if first[0] is None:
                first[0] = (path + [str(k)], None, act[k])
            total[0] += 1
        for k in sorted(keys_e & keys_a):
            _walk_diff(
                exp[k],
                act[k],
                path + [str(k)],
                first,
                total,
                max_total,
                float_rtol=float_rtol,
                float_atol=float_atol,
            )
        return

    if isinstance(exp, list):
        le, la = len(exp), len(act)
        if le != la:
            if first[0] is None:
                first[0] = (list(path), f"<list len {le}>", f"<list len {la}>")
            total[0] += 1
            return
        for i, (ve, va) in enumerate(zip(exp, act)):
            _walk_diff(
                ve,
                va,
                path + [f"[{i}]"],
                first,
                total,
                max_total,
                float_rtol=float_rtol,
                float_atol=float_atol,
            )
        return

    if first[0] is None:
        first[0] = (list(path), exp, act)
    total[0] += 1


def compare_json(
    expected: Any,
    actual: Any,
    max_reported_diffs: int = 50,
    *,
    float_rtol: float = 0.0,
    float_atol: float = 0.0,
) -> Tuple[int, Optional[Tuple[str, str, str]]]:
    first: List[Optional[Tuple[List[str], Any, Any]]] = [None]
    total: List[int] = [0]
    _walk_diff(
        expected,
        actual,
        [],
        first,
        total,
        max_reported_diffs + 1,
        float_rtol=float_rtol,
        float_atol=float_atol,
    )
    count = total[0]
    fst = first[0]
    if fst is None:
        return 0, None
    pth, va, vb = fst
    summary = (_path_to_str(pth), _preview(va), _preview(vb))
    return count, summary


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Deep JSON vs golden (default strict). Optional numeric leaf tolerances via isclose.",
    )
    p.add_argument("expected_path", type=Path, help="golden / expected JSON path")
    p.add_argument("actual_path", type=Path, help="actual output JSON path")
    p.add_argument("--pair-id", default="", help="label for PASS line (e.g. default_layout)")
    p.add_argument("--max-diffs", type=int, default=50, help="stop counting after this many mismatches")
    p.add_argument(
        "--float-rtol",
        type=float,
        default=0.0,
        help="leaf number rel_tol for math.isclose (0 disables)",
    )
    p.add_argument(
        "--float-atol",
        type=float,
        default=0.0,
        help="leaf number abs_tol for math.isclose (0 disables)",
    )
    args = p.parse_args(argv)

    if not args.expected_path.exists():
        print(f"golden_compare: expected not found: {args.expected_path}", file=sys.stderr)
        return 2
    if not args.actual_path.exists():
        print(f"golden_compare: actual not found: {args.actual_path}", file=sys.stderr)
        return 2

    try:
        expected = _load_json(args.expected_path)
        actual = _load_json(args.actual_path)
    except Exception as e:
        print(f"golden_compare: JSON load failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    rtol = float(args.float_rtol)
    atol = float(args.float_atol)
    if rtol > 0.0 or atol > 0.0:
        print(
            f"golden_compare numeric mode rtol={rtol:g} atol={atol:g}",
            file=sys.stderr,
        )
    count, first = compare_json(
        expected,
        actual,
        max_reported_diffs=args.max_diffs,
        float_rtol=rtol,
        float_atol=atol,
    )
    if first is None:
        label = args.pair_id.strip() or args.actual_path.name
        tag = ""
        if rtol > 0.0 or atol > 0.0:
            tag = f" rtol={rtol:g} atol={atol:g}"
        print(f"PASS golden {label}{tag}")
        return 0

    path_s, va_s, vb_s = first
    at_least = f">= {args.max_diffs + 1}" if count > args.max_diffs else str(count)
    print(f"FAIL golden: first mismatch at `{path_s}`", file=sys.stderr)
    print(f"  expected: {va_s}", file=sys.stderr)
    print(f"  actual:   {vb_s}", file=sys.stderr)
    print(f"  mismatch count (capped report): {at_least}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
