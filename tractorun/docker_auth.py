import abc

import attrs


__all__ = ["DockerAuthSecret", "DockerAuthPlainText", "DockerAuthData"]


class DockerAuthData(abc.ABC):
    pass


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class DockerAuthSecret(DockerAuthData):
    cypress_path: str


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class DockerAuthPlainText(DockerAuthData):
    username: str | None
    password: str | None
    auth: str | None
