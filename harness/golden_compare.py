from __future__ import annotations

import argparse
import json
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


def _walk_diff(
    exp: Any,
    act: Any,
    path: List[str],
    first: List[Optional[Tuple[List[str], Any, Any]]],
    total: List[int],
    max_total: int,
) -> None:
    if total[0] >= max_total:
        return
    if exp == act:
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
            _walk_diff(exp[k], act[k], path + [str(k)], first, total, max_total)
        return

    if isinstance(exp, list):
        le, la = len(exp), len(act)
        if le != la:
            if first[0] is None:
                first[0] = (list(path), f"<list len {le}>", f"<list len {la}>")
            total[0] += 1
            return
        for i, (ve, va) in enumerate(zip(exp, act)):
            _walk_diff(ve, va, path + [f"[{i}]"], first, total, max_total)
        return

    if first[0] is None:
        first[0] = (list(path), exp, act)
    total[0] += 1


def compare_json(expected: Any, actual: Any, max_reported_diffs: int = 50) -> Tuple[int, Optional[Tuple[str, str, str]]]:
    first: List[Optional[Tuple[List[str], Any, Any]]] = [None]
    total: List[int] = [0]
    _walk_diff(expected, actual, [], first, total, max_reported_diffs + 1)
    count = total[0]
    fst = first[0]
    if fst is None:
        return 0, None
    pth, va, vb = fst
    summary = (_path_to_str(pth), _preview(va), _preview(vb))
    return count, summary


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Deep JSON equality vs golden (no tolerance).")
    p.add_argument("expected_path", type=Path, help="golden / expected JSON path")
    p.add_argument("actual_path", type=Path, help="actual output JSON path")
    p.add_argument("--pair-id", default="", help="label for PASS line (e.g. default_layout)")
    p.add_argument("--max-diffs", type=int, default=50, help="stop counting after this many mismatches")
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

    count, first = compare_json(expected, actual, max_reported_diffs=args.max_diffs)
    if first is None:
        label = args.pair_id.strip() or args.actual_path.name
        print(f"PASS golden {label}")
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
