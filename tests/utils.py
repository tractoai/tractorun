import os
import string
import random


def get_data_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), "data", filename)


def get_random_string(length: int) -> str:
    return "".join(random.choice(string.ascii_letters) for _ in range(length))
