import sys
import time

from yt import wrapper as yt

from tractorun.backend.generic import GenericBackend
from tractorun.run import prepare_and_get_toolbox


def main() -> None:
    toolbox = prepare_and_get_toolbox(backend=GenericBackend())
    user_config = toolbox.get_user_config()

    yt_path = user_config["yt_path"]
    attr_key = user_config["attr_key"]
    attr_value = user_config["attr_value"]

    client = toolbox.yt_client
    value = None
    attempts = 0
    while value is None:
        attr_path = f"{yt_path}/@{attr_key}"
        print(f"try to read value from {attr_path}", file=sys.stderr)
        try:
            value = client.get(attr_path)
        except yt.errors.YtResolveError:
            pass
        time.sleep(5)
        if attempts > 5:
            raise Exception("Something wrong with sidecar")
        attempts += 1
    print("its ok", file=sys.stderr)
    assert value == attr_value


if __name__ == "__main__":
    main()
