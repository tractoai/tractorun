from typing import Optional as _Optional

import attrs as _attrs


@_attrs.define(kw_only=True, slots=True, auto_attribs=True)
class EnvVariable:
    name: str
    value: _Optional[str] = None
    cypress_path: _Optional[str] = None
