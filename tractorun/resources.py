from typing import Optional

import attrs


@attrs.define
class Resources:
    cpu_limit: Optional[float] = None
    memory_limit: Optional[int] = None
