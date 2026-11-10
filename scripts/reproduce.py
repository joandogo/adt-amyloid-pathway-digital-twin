"""One entry point for the article's mechanistic and surrogate workflows."""

from importlib import import_module
import sys


COMMANDS = {
    "p14-run": "adt.final_cohort_cli",
    "p14-collect": "adt.final_cohort_collection_cli",
    "jt1-run": "adt.joint_interventions_cli",
    "jt1-collect": "adt.joint_interventions_collection_cli",
    "e1": "adt.therapeutic_surrogate_cli",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        names = ", ".join(COMMANDS)
        print(f"Usage: python scripts/reproduce.py <command> [options]\nCommands: {names}",
              file=sys.stderr)
        return 2
    module = import_module(COMMANDS[sys.argv[1]])
    return module.main(sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
