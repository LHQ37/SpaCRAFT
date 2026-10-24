"""SpaCRAFT command-line dispatcher."""

from argparse import ArgumentParser


def main() -> None:
    parser = ArgumentParser(
        prog="spacraft",
        description="SpaCRAFT experiment entry points are provided under scripts/.",
    )
    parser.add_argument(
        "--version", action="version", version="SpaCRAFT 1.0.0"
    )
    parser.parse_args()


if __name__ == "__main__":
    main()
