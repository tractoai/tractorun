import os
import random
import string


DOCKER_IMAGE: str = "cr.ai.nebius.cloud/crnf2coti090683j5ssi/tractorun/torchesaurus_tests:2024-06-17-15-21-24"
DOCKER_IMAGE_TRTRCH: str = (
    "cr.ai.nebius.cloud/crnf2coti090683j5ssi/tractorun/torchesaurus_tests_with_tractorch:2024-07-15-22-51-37"
)


def get_data_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), "data", filename)


def get_random_string(length: int) -> str:
    return "".join(random.choice(string.ascii_letters) for _ in range(length))
