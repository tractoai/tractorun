import io

import torch


__all__ = ["TensorSerializer"]


class TensorSerializer:
    def serialize(self, tensor: object) -> bytes:
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()

    def desirialize(self, tensor: bytes, device: torch.device | None = None) -> dict:
        if device is None:
            device = torch.device("cpu")
        buffer = io.BytesIO(tensor)
        return torch.load(buffer, map_location=device)
