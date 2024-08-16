import sys

from tractorun.backend.generic import GenericBackend
from tractorun.run import prepare_and_get_toolbox


def main() -> None:
    toolbox = prepare_and_get_toolbox(backend=GenericBackend())
    test_strings = toolbox.get_user_config()["test_strings"]
    for test_string in test_strings:
        print(test_string, file=sys.stderr)


if __name__ == "__main__":
    main()
