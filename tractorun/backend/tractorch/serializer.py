import io

import torch


class TensorSerializer:
    def serialize(self, tensor: object) -> bytes:
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()

    def desirialize(self, tensor: bytes, device: torch.device = torch.device("cpu")) -> dict:
        buffer = io.BytesIO(tensor)
        return torch.load(buffer, map_location=device)
