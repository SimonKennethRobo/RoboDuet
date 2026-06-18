"""Unified benchmark entrypoint."""

from __future__ import annotations

import argparse
from typing import List, Optional


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="RoboDuet benchmark entrypoint.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dog_only", action="store_true", help="Run dog-policy benchmark")
    mode.add_argument("--inspect", action="store_true", help="Inspect candidate policy checkpoint layouts")
    mode.add_argument("--compare_results", action="store_true", help="Compare two saved benchmark result directories")
    mode.add_argument("--arm_only", action="store_true", help="[Reserved] Run arm-policy benchmark")
    mode.add_argument("--hybrid", action="store_true", help="[Reserved] Run dog+arm benchmark")
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def main(argv: Optional[List[str]] = None):
    args, remaining = parse_args(argv)

    if args.arm_only:
        raise NotImplementedError("arm_only benchmark mode is reserved but not implemented yet")
    if args.hybrid:
        raise NotImplementedError("hybrid benchmark mode is reserved but not implemented yet")
    if args.inspect:
        from benchmark.inspect import main as inspect_main

        inspect_main(remaining)
        return
    if args.compare_results:
        from benchmark.compare import main as compare_main

        compare_main(remaining)
        return

    from benchmark.dog_policy.cli import main as dog_main

    dog_main(remaining)


if __name__ == "__main__":
    main()
