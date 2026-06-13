from __future__ import annotations

import argparse
import json
import sys

from scattered_discovery.algos.registry import get_algorithm, list_algorithms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="grpo")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--allow-experimental",
        action="store_true",
        help="Print overrides for algorithms that require custom trainer patches.",
    )
    args = parser.parse_args()

    if args.list:
        payload = [algorithm.as_dict() for algorithm in list_algorithms()]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    algorithm = get_algorithm(args.algo)
    if algorithm.requires_custom_trainer and not args.allow_experimental:
        print(
            (
                f"{algorithm.name} requires a custom veRL trainer patch. "
                "Pass --allow-experimental only when that patch is installed."
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)

    if args.json:
        print(json.dumps(algorithm.as_dict(), indent=2, sort_keys=True))
        return

    for override in algorithm.verl_overrides():
        print(override)


if __name__ == "__main__":
    main()
