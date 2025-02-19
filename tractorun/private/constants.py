import logging


YT_USER_CONFIG_ENV_VAR: str = "YT_USER_CONFIG"
TRACTO_CONFIG_ENV_VAR: str = "TRACTO_CONFIG"
BIND_PATHS_ENV_VAR: str = "BIND_PATHS"
BOOTSTRAP_CONFIG_FILENAME_ENV_VAR: str = "BOOTSTRAP_CONFIG_FILENAME"
DEFAULT_CLUSTER_CONFIG_PATH: str = "//sys/tractorun/config"
DEFAULT_TENSORPROXY_PATH = "//sys/tractorun/tensorproxy"
JOB_SANDBOX_PATH: str = "/slot/sandbox"
LOG_LEVEL_INSIDE_JOB: int = logging.INFO

BOOTSTRAP_CONFIG_NAME: str = "__bootstrap_config"

USER_DESCRIPTION_MANAGER_NAME = "extra"
TRACTORUN_DESCRIPTION_MANAGER_NAME = "tractorun"
