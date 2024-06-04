import io

import torch


class TensorSerializer:
    def save_tensor(self, tensor: object) -> bytes:
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()

    def load_tensor(self, tensor: bytes, device: torch.device = torch.device("cpu")) -> dict:
        buffer = io.BytesIO(tensor)
        return torch.load(buffer, map_location=device)
