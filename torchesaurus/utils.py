import io
import torch


def save_tensor(tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()

def load_tensor(tensor, device: torch.device = torch.device('cpu')):
    buffer = io.BytesIO(tensor)
    return torch.load(buffer, map_location=device)
