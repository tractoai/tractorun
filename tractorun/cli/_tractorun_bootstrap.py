#!/usr/bin/env python3
import os

from tractorun.bind import BindsPacker
from tractorun.bootstrapper import (
    BOOTSTRAP_CONFIG_YT_PATH,
    BootstrapConfig,
    bootstrap,
)
from tractorun.constants import (
    BIND_PATHS_ENV_VAR,
    BOOTSTRAP_CONFIG_FILENAME_ENV_VAR,
)
from tractorun.helpers import AttrSerializer


def main() -> None:
    binds_packer = BindsPacker.from_env(os.environ[BIND_PATHS_ENV_VAR])
    binds_packer.unpack()
    bootstrap_config_path = os.path.join(
        BOOTSTRAP_CONFIG_YT_PATH,
        os.environ[BOOTSTRAP_CONFIG_FILENAME_ENV_VAR],
    )
    with open(bootstrap_config_path, "r") as f:
        content = f.read()
        deserializer = AttrSerializer(BootstrapConfig)
        config: BootstrapConfig = deserializer.deserialize(data=content)
    bootstrap(
        mesh=config.mesh,
        path=config.path,
        yt_client_config=config.yt_client_config,
        command=config.command,
    )


if __name__ == "__main__":
    main()
