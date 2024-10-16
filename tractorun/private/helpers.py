import json
import os
from typing import (
    Generic,
    Type,
    TypeVar,
)

import attrs
import cattrs
import yt.wrapper as yt


_T = TypeVar("_T")


def create_attrs_converter(forbid_extra_keys: bool = True) -> cattrs.Converter:
    converter = cattrs.Converter()
    converter.register_structure_hook_factory(
        attrs.has,
        lambda cl: cattrs.gen.make_dict_structure_fn(cl, converter, _cattrs_forbid_extra_keys=forbid_extra_keys),
    )
    return converter


@attrs.define(slots=True)
class AttrSerializer(Generic[_T]):
    _type: Type[_T] = attrs.field()
    _converter: cattrs.Converter = attrs.field(factory=create_attrs_converter)

    def serialize(self, data: _T) -> str:
        return json.dumps(self._converter.unstructure(data))

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


def get_default_docker_image() -> str | None:
    # use the same env var as yt sdk
    return os.environ.get("YT_BASE_LAYER") or os.environ.get("YT_JOB_DOCKER_IMAGE") or None
