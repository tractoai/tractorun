import os

from tractorun.backend.generic import GenericBackend
from tractorun.run import prepare_and_get_toolbox


def main() -> None:
    toolbox = prepare_and_get_toolbox(backend=GenericBackend())
    user_config = toolbox.get_user_config()
    assert os.environ["SECRET"] == user_config["secret_env_value"]
    assert os.environ["NOT_SECRET"] == user_config["not_secret_env_value"]


if __name__ == "__main__":
    main()
