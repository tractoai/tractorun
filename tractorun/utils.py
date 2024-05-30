import io

import torch


def save_tensor(tensor: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def load_tensor(tensor: bytes, device: torch.device = torch.device("cpu")) -> dict:
    buffer = io.BytesIO(tensor)
    return torch.load(buffer, map_location=device)
