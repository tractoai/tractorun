from typing import Any

import attrs
from yt import wrapper as yt

from tractorun.docker_auth import (
    DockerAuthData,
    DockerAuthSecret,
)
from tractorun.exception import DockerAuthDataError


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class DockerAuthInternal:
    username: str | None
    password: str | None
    auth: str | None

    def to_spec(self) -> dict:
        data = {}
        if self.username:
            data["username"] = self.username
        if self.password:
            data["password"] = self.password
        if self.auth:
            data["auth"] = self.auth
        return data


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class DockerAuthDataExtractor:
    _yt_client: yt.YtClient

    def _is_raw_auth_data_valid(self, raw_auth_data: Any) -> bool:
        if not isinstance(raw_auth_data, dict):
            return False
        if set(raw_auth_data.keys()) != {"username", "password"} and set(raw_auth_data.keys()) != {"auth"}:
            return False
        if not all([isinstance(value, str) for value in raw_auth_data.values()]):
            return False
        return True

    def extract(self, auth_data: DockerAuthData) -> DockerAuthInternal:
        match auth_data:
            case DockerAuthSecret():
                assert isinstance(auth_data, DockerAuthSecret)  # skip type warning in pycharm
                raw_auth_data = self._yt_client.get(auth_data.cypress_path)
                if not self._is_raw_auth_data_valid(raw_auth_data):
                    format_options = '{username="..."; password="...";} or {auth="...";}'
                    raise DockerAuthDataError(f"docker_auth document should have format {format_options}")
                return DockerAuthInternal(
                    username=raw_auth_data.get("username"),
                    password=raw_auth_data.get("password"),
                    auth=raw_auth_data.get("auth"),
                )
            case _:
                raise DockerAuthDataError("Unknown auth data type")
