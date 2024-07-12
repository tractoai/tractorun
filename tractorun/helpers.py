import json
import typing

import attr
import cattrs
import yt.wrapper as yt


_T = typing.TypeVar("_T")


@attr.s(slots=True)
class AttrSerializer(typing.Generic[_T]):
    _type: typing.Type[_T] = attr.field()

    def serialize(self, data: _T) -> str:
        return json.dumps(cattrs.unstructure(data))

    def deserialize(self, data: str) -> _T:
        return cattrs.structure(json.loads(data), self._type)


def create_prerequisite_client(yt_client: yt.YtClient, prerequisite_transaction_ids: typing.List[str]) -> yt.YtClient:
    if yt_client:
        try:
            prerequisite_transaction_ids = prerequisite_transaction_ids + yt_client.get_option(
                "prerequisite_transaction_ids"
            )
        except Exception:
            pass
    return yt.create_client_with_command_params(yt_client, prerequisite_transaction_ids=prerequisite_transaction_ids)
