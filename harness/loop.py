from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    command: str
    input_path: str
    output_path: str
    exit_code: int
    wall_sec: float
    failure_type: Optional[str]


def _now_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _run_cmd(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(_ROOT / "data" / "Result_storage" / "default_layout_sim_input.json"))
    p.add_argument("--output", default=str(_ROOT / "data" / "Result_storage" / "_validation_sim_result.json"))
    p.add_argument("--dt", type=float, default=5.0)
    p.add_argument("--max-runs", type=int, default=2)
    p.add_argument("--record", default=str(_ROOT / "harness" / "run_records.jsonl"))
    args = p.parse_args(argv)

    max_runs = max(1, int(args.max_runs))
    record_path = Path(args.record)

    last_failure: Optional[str] = None
    same_failure_count = 0

    for _i in range(max_runs):
        run_id = _now_id()

        t0 = time.time()
        smoke = _run_cmd([sys.executable, "-m", "harness.smoke"])
        if smoke.returncode != 0:
            wall = time.time() - t0
            failure = "smoke"
            _append_jsonl(
                record_path,
                asdict(
                    RunRecord(
                        run_id=run_id,
                        command="python -m harness.smoke",
                        input_path=str(args.input),
                        output_path=str(args.output),
                        exit_code=int(smoke.returncode),
                        wall_sec=float(wall),
                        failure_type=failure,
                    )
                ),
            )
            sys.stderr.write(smoke.stderr or "")
            sys.stdout.write(smoke.stdout or "")
            return int(smoke.returncode)

        run = _run_cmd(
            [
                sys.executable,
                "-m",
                "harness.run",
                "--input",
                str(args.input),
                "--output",
                str(args.output),
                "--dt",
                str(args.dt),
            ]
        )
        wall = time.time() - t0

        failure_type: Optional[str] = None
        if run.returncode == 0:
            _append_jsonl(
                record_path,
                asdict(
                    RunRecord(
                        run_id=run_id,
                        command="python -m harness.run",
                        input_path=str(args.input),
                        output_path=str(args.output),
                        exit_code=0,
                        wall_sec=float(wall),
                        failure_type=None,
                    )
                ),
            )
            sys.stdout.write(run.stdout or "")
            sys.stderr.write(run.stderr or "")
            return 0

        failure_type = "validate" if run.returncode == 1 else "runtime"
        if failure_type == last_failure:
            same_failure_count += 1
        else:
            same_failure_count = 1
            last_failure = failure_type

        _append_jsonl(
            record_path,
            asdict(
                RunRecord(
                    run_id=run_id,
                    command="python -m harness.run",
                    input_path=str(args.input),
                    output_path=str(args.output),
                    exit_code=int(run.returncode),
                    wall_sec=float(wall),
                    failure_type=failure_type,
                )
            ),
        )

        sys.stdout.write(run.stdout or "")
        sys.stderr.write(run.stderr or "")

        if same_failure_count >= 2:
            sys.stderr.write(
                f"\nloop: same failure '{failure_type}' repeated {same_failure_count}x; stop (no blind retry).\n"
            )
            return int(run.returncode)

        time.sleep(0.2)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

