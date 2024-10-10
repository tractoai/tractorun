import pathlib

from tractorun.backend.tractorch import Tractorch
from tractorun.run import prepare_and_get_toolbox


if __name__ == "__main__":
    toolbox = prepare_and_get_toolbox(backend=Tractorch())
    user_config = toolbox.get_user_config()
    cypress_file_to_check = user_config["CYPRESS_FILE_TO_CHECK"]

    assert pathlib.Path(cypress_file_to_check).exists()
    assert pathlib.Path(cypress_file_to_check).is_file()
