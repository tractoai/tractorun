import io
import json
import os
from typing import (
    Any,
    Dict,
)

import torch

from tractorun import constants as const


def save_tensor(tensor: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def load_tensor(tensor: bytes, device: torch.device = torch.device("cpu")) -> dict:
    buffer = io.BytesIO(tensor)
    return torch.load(buffer, map_location=device)


def get_user_config() -> Dict[Any, Any]:
    return json.loads(os.environ[const.YT_USER_CONFIG_ENV_VAR])
