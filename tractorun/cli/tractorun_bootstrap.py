#!/usr/bin/env python3
import os
import sys

import cattrs

from tractorun.exception import TractorunVersionMismatchError
from tractorun.private.bind import BindsPacker
from tractorun.private.bootstrapper import (
    BootstrapConfig,
    bootstrap,
)
from tractorun.private.constants import (
    BIND_PATHS_ENV_VAR,
    BOOTSTRAP_CONFIG_FILENAME_ENV_VAR,
)
from tractorun.private.helpers import (
    AttrSerializer,
    create_attrs_converter,
)


def main() -> None:
    binds_packer = BindsPacker.from_env(os.environ[BIND_PATHS_ENV_VAR])
    binds_packer.unpack()
    bootstrap_config_path = os.environ[BOOTSTRAP_CONFIG_FILENAME_ENV_VAR]
    with open(bootstrap_config_path, "r") as f:
        content = f.read()
        deserializer = AttrSerializer(
            BootstrapConfig,
            # forward compatibility
            converter=create_attrs_converter(forbid_extra_keys=False),
        )
        try:
            config: BootstrapConfig = deserializer.deserialize(data=content)
        except cattrs.errors.BaseValidationError as e:
            raise TractorunVersionMismatchError(
                "Please check that the tractorun version locally and on YT are the same",
            ) from e
    bootstrap(
        mesh=config.mesh,
        training_dir=config.training_dir,
        yt_client_config=config.yt_client_config,
        sidecars=config.sidecars,
        env=config.env,
        command=sys.argv[1:],
        tensorproxy=config.tensorproxy,
        lib_versions=config.lib_versions,
        cluster_config=config.cluster_config,
        operation_log_mode=config.operation_log_mode,
    )


if __name__ == "__main__":
    main()
