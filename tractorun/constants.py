from typing import Dict


YT_USER_CONFIG_ENV_VAR: str = "YT_USER_CONFIG"

GPU_TYPE_TO_POOLTREE: Dict[str, str] = {
    "H100": "H100",
}
DEFAULT_POOLTREE: str = "default"
DEFAULT_DOCKER_IMAGE: str = "cr.ai.nebius.cloud/crnf2coti090683j5ssi/tractorun/torchesaurus_runtime:2024-06-07-19-46-47"
