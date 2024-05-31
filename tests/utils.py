import os
import random
import string


DOCKER_IMAGE = "cr.ai.nebius.cloud/crnf2coti090683j5ssi/tractorun/torchesaurus_tests:2024-05-31-12-07-26"


def get_data_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), "data", filename)


def get_random_string(length: int) -> str:
    return "".join(random.choice(string.ascii_letters) for _ in range(length))
