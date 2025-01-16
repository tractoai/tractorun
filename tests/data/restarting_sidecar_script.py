import sys
import time

from tractorun.backend.generic import GenericBackend
from tractorun.run import prepare_and_get_toolbox


def main() -> None:
    toolbox = prepare_and_get_toolbox(backend=GenericBackend())
    user_config = toolbox.get_user_config()

    yt_path = user_config["yt_path"]
    attr_key = user_config["attr_key"]

    client = toolbox.yt_client
    value = None
    while value is None:
        attr_path = f"{yt_path}/@{attr_key}"
        value = client.get(attr_path)
        print(f"sidecar read {value}", file=sys.stderr)
        client.set(attr_path, value + 1)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
