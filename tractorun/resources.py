from dataclasses import dataclass


@dataclass
class Resources:
    cpu_limit: float = None
    memory_limit: int = None
