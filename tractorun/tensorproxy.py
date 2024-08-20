import attrs

from tractorun.sidecar import RestartPolicy as _RestartPolicy


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TensorproxySidecar:
    enabled: bool
    restart_policy: _RestartPolicy
    yt_path: str
