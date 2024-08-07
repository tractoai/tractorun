from typing import Optional

import attrs


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class EnvVariable:
    name: str
    value: Optional[str] = None
    cypress_path: Optional[str] = None
