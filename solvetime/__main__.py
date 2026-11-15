"""python -m solvetime run ... | report ..."""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "speed", "report"):
        sys.exit("usage: python -m solvetime run|speed|report [options]  (--help after the command for its options)")
    command, argv = sys.argv[1], sys.argv[2:]
    if command == "run":
        from .run import main as run
        run(argv)
    elif command == "speed":
        from .speed import main as speed
        speed(argv)
    else:
        from .report import main as report
        report(argv)


if __name__ == "__main__":
    main()
