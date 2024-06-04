import json
import os
from typing import (
    Any,
    Dict,
)

from tractorun import constants as const


def get_user_config() -> Dict[Any, Any]:
    return json.loads(os.environ[const.YT_USER_CONFIG_ENV_VAR])
