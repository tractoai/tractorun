from typing import Union

import attrs
from yt import wrapper as yt


__all__ = ["Link", "DescriptionManager", "Description"]

from tractorun.private.yt_cluster import make_cypress_link as _make_cypress_link


@attrs.define(kw_only=True, slots=True, auto_attribs=True, frozen=True)
class Link:
    _value: str | None

    def to_yson(self) -> yt.yson.yson_types.YsonUnicode | None:
        if self._value is None:
            return None
        value = yt.yson.yson_types.YsonUnicode(self._value)
        value.attributes = {"_type_tag": "url"}
        return value


_PRIMITIVE_DESCRIPTION = str | bool | bytes | int | float | Link | None
_COMPLEX_DESCRIPTION = (
    dict[_PRIMITIVE_DESCRIPTION, Union[_PRIMITIVE_DESCRIPTION, "_COMPLEX_DESCRIPTION"]]
    | list[Union[_PRIMITIVE_DESCRIPTION, "_COMPLEX_DESCRIPTION"]]
)
_DESCRIPTION = _COMPLEX_DESCRIPTION | _PRIMITIVE_DESCRIPTION
Description = dict[str, _DESCRIPTION]


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class DescriptionManager:
    _operation_id: str
    _yt_client: yt.YtClient
    _cypress_link_template: str | None
    _key: list[str] = attrs.field(default=[])

    def get_child(self, key: str) -> "DescriptionManager":
        return DescriptionManager(
            operation_id=self._operation_id,
            yt_client=self._yt_client,
            key=self._key + [key],
            cypress_link_template=self._cypress_link_template,
        )

    def make_cypress_link(self, value: str) -> Link | None:
        raw_link = _make_cypress_link(
            cypress_link_template=self._cypress_link_template,
            path=value,
        )
        if raw_link is None:
            return None
        return Link(value=raw_link)

    @classmethod
    def _convert_yson(cls, description: _DESCRIPTION) -> _DESCRIPTION:
        match description:
            case dict():
                return {k: cls._convert_yson(v) for k, v in description.items()}
            case list():
                return [cls._convert_yson(v) for v in description]
            case Link():
                return description.to_yson()
        return description

    @classmethod
    def _make_description(cls, key: list[str], description: Description) -> _DESCRIPTION:
        result_description: dict = {}
        current_part = result_description
        for k in key:
            current_part[k] = {}
            current_part = current_part[k]
        current_part.update(description)
        converted_description = cls._convert_yson(result_description)
        return converted_description

    def set(self, description: Description) -> None:
        new_description = self._make_description(self._key, description)
        self._yt_client.update_operation_parameters(
            self._operation_id,
            parameters={
                "annotations": {"description": new_description},
            },
        )
