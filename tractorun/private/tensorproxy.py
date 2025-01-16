import os
from pathlib import Path
import shutil
from typing import Optional

import attrs
from yt import yson

from tractorun.sidecar import (
    RestartPolicy,
    Sidecar,
)
from tractorun.tensorproxy import TensorproxySidecar


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TensorproxyBootstrap:
    restart_policy: RestartPolicy
    ports_count: int

    def prepare_and_get_sidecars(
        self, yt_proxy: str, grpc_port: int, monitoring_port: int, sandbox_path: Path
    ) -> list[Sidecar]:
        path = sandbox_path / "__tensorproxy_config.yson"
        with open(path, "wb") as f:
            config_content = get_config(proxy=yt_proxy, grpc_port=grpc_port, monitoring_port=monitoring_port)
            f.write(yson.dumps(config_content))
        new_tensorproxy_path = sandbox_path / "tensorproxy_fix_perm"
        shutil.copy(sandbox_path / "tensorproxy", new_tensorproxy_path)
        os.chmod(new_tensorproxy_path, 0o755)
        return [
            Sidecar(
                command=[str(new_tensorproxy_path.absolute()), "--config", str(path.absolute())],
                restart_policy=self.restart_policy,
            ),
        ]

    def get_environment(self, grpc_port: int) -> dict:
        return {
            "TS_GRPC_ADDRESS": f"localhost:{grpc_port}",
        }


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TensorproxyConfigurator:
    _tensorproxy: Optional[TensorproxySidecar]

    def generate_configuration(self) -> tuple[Optional[TensorproxyBootstrap], list[str], int]:
        if self._tensorproxy is None or self._tensorproxy.enabled is False:
            return None, [], 0
        ports_count = 2  # 1 for monitoring and 1 for grpc handler
        return (
            TensorproxyBootstrap(
                restart_policy=self._tensorproxy.restart_policy,
                ports_count=ports_count,
            ),
            [self._tensorproxy.yt_path],
            ports_count,
        )


def get_config(grpc_port: int, monitoring_port: int, proxy: str) -> dict:
    return {
        "monitoring_port": monitoring_port,
        "grpc_port": grpc_port,
        "store_kind": "cypress",
        "cypress_store": {"cluster_url": proxy, "file_writer": {"block_size": 262144}},
        "logging": {
            "flush_period": 100,
            "rules": [
                {"exclude_categories": ["Bus"], "family": "plain_text", "min_level": "debug", "writers": ["debug"]},
                {"family": "plain_text", "min_level": "info", "writers": ["info"]},
                {"family": "plain_text", "min_level": "error", "writers": ["error"]},
            ],
            "writers": {
                "debug": {
                    "enable_system_messages": True,
                    "file_name": "./tensorproxy.debug.log",
                    "format": "plain_text",
                    "rotation_policy": {
                        "max_segment_count_to_keep": 1000,
                        "max_total_size_to_keep": 130000000000,
                        "rotation_period": 900000,
                    },
                    "type": "file",
                },
                "error": {"enable_system_messages": True, "format": "plain_text", "type": "stderr"},
                "info": {
                    "enable_system_messages": True,
                    "file_name": "./tensorproxy.info.log",
                    "format": "plain_text",
                    "rotation_policy": {
                        "max_segment_count_to_keep": 1000,
                        "max_total_size_to_keep": 130000000000,
                        "rotation_period": 900000,
                    },
                    "type": "file",
                },
            },
        },
    }
