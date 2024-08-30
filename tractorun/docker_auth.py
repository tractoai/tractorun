import abc

import attrs


__all__ = ["DockerAuthSecret", "DockerAuthData"]


class DockerAuthData(abc.ABC):
    pass


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class DockerAuthSecret(DockerAuthData):
    cypress_path: str
