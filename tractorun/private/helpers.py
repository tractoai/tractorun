import json
from typing import (
    Generic,
    Type,
    TypeVar,
)

import attrs
import cattrs
import yt.wrapper as yt


_T = TypeVar("_T")


@attrs.define(slots=True)
class AttrSerializer(Generic[_T]):
    _type: Type[_T] = attrs.field()

    def serialize(self, data: _T) -> str:
        return json.dumps(cattrs.unstructure(data))

    def deserialize(self, data: str) -> _T:
        return cattrs.structure(json.loads(data), self._type)


def create_prerequisite_client(yt_client: yt.YtClient, prerequisite_transaction_ids: list[str]) -> yt.YtClient:
    if yt_client:
        try:
            prerequisite_transaction_ids = prerequisite_transaction_ids + yt_client.get_option(
                "prerequisite_transaction_ids"
            )
        except Exception:
            pass
    return yt.create_client_with_command_params(yt_client, prerequisite_transaction_ids=prerequisite_transaction_ids)
