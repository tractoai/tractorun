#!/usr/bin/env python3
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Internal tractorun bootstrapper.",
    )
    parser.add_argument("--yt-client-state-path", help="Serialized yt client's config", type=str)


if __name__ == "__main__":
    main()
