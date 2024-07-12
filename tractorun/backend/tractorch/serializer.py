import io
from typing import Optional

import torch


class TensorSerializer:
    def serialize(self, tensor: object) -> bytes:
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()

    def desirialize(self, tensor: bytes, device: Optional[torch.device] = None) -> dict:
        if device is None:
            device = torch.device("cpu")
        buffer = io.BytesIO(tensor)
        return torch.load(buffer, map_location=device)
