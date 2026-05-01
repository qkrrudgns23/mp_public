from __future__ import annotations

import argparse

from utils.layout_receiver import LAYOUT_RECEIVER_PORT, serve_layout_receiver_forever


def main() -> int:
    parser = argparse.ArgumentParser(description="Run standalone Layout Design server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=LAYOUT_RECEIVER_PORT)
    args = parser.parse_args()
    serve_layout_receiver_forever(host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
