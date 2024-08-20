from typing import Optional as _Optional

import attrs as _attrs


@_attrs.define
class Resources:
    cpu_limit: _Optional[float] = None
    memory_limit: _Optional[int] = None
