import sys

from tractorun.backend.generic import GenericBackend
from tractorun.run import prepare_and_get_toolbox


def main() -> None:
    toolbox = prepare_and_get_toolbox(backend=GenericBackend())
    self_index = toolbox.coordinator.get_self_index()
    inc = toolbox.coordinator.get_incarnation_id()
    print(f"first stdout line {self_index} {inc}", file=sys.stdout)
    print(f"first stderr line {self_index} {inc}", file=sys.stderr)
    print(f"second stdout line {self_index} {inc}", file=sys.stdout)
    print(f"second stderr line {self_index} {inc}", file=sys.stderr)


if __name__ == "__main__":
    main()
