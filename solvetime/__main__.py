"""python -m solvetime run ... | report ..."""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "report"):
        sys.exit("usage: python -m solvetime run|report [options]  (--help after the command for its options)")
    command, argv = sys.argv[1], sys.argv[2:]
    if command == "run":
        from .run import main as run
        run(argv)
    else:
        from .report import main as report
        report(argv)


if __name__ == "__main__":
    main()
