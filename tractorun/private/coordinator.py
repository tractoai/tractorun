import yt.wrapper as yt
from yt.wrapper.errors import YtResolveError

from tractorun.private.training_dir import TrainingDir


def get_incarnation_id(yt_client: yt.YtClient, training_dir: TrainingDir, raise_if_not_exists: bool = False) -> int:
    try:
        return yt_client.get(training_dir.base_path + "/@incarnation_id")
    except YtResolveError:
        if raise_if_not_exists:
            raise
        return -1
