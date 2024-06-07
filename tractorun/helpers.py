import typing as tp

import yt.wrapper as yt


def create_prerequisite_client(yt_client: yt.YtClient, prerequisite_transaction_ids: tp.List[str]) -> yt.YtClient:
    if yt_client:
        try:
            prerequisite_transaction_ids = prerequisite_transaction_ids + yt_client.get_option(
                "prerequisite_transaction_ids"
            )
        except Exception:
            pass
    return yt.create_client_with_command_params(yt_client, prerequisite_transaction_ids=prerequisite_transaction_ids)
