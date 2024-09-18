from tractorun.backend.generic import GenericBackend
from tractorun.run import prepare_and_get_toolbox


def main() -> None:
    _ = prepare_and_get_toolbox(backend=GenericBackend())


if __name__ == "__main__":
    main()
