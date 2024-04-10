import yt.wrapper as yt

import typing as tp


def create_prerequisite_client(client: yt.YtClient, prerequisite_transaction_ids: tp.List[str]) -> yt.YtClient:
    if client:
        try:
            prerequisite_transaction_ids = prerequisite_transaction_ids + client.get_option('prerequisite_transaction_ids')
        except:
            pass
    return yt.create_client_with_command_params(client, prerequisite_transaction_ids=prerequisite_transaction_ids)
