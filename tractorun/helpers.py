import typing as tp

import yt.wrapper as yt


def create_prerequisite_client(yt_cli: yt.YtClient, prerequisite_transaction_ids: tp.List[str]) -> yt.YtClient:
    if yt_cli:
        try:
            prerequisite_transaction_ids = prerequisite_transaction_ids + yt_cli.get_option(
                "prerequisite_transaction_ids"
            )
        except Exception:
            pass
    return yt.create_client_with_command_params(yt_cli, prerequisite_transaction_ids=prerequisite_transaction_ids)
