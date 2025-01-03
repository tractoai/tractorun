import logging
from typing import Any
import warnings

import attrs
from yt import wrapper as yt

from tractorun.docker_auth import (
    DockerAuthData,
    DockerAuthSecret,
)
from tractorun.exception import DockerAuthDataError


LOGGER = logging.getLogger(__name__)


CANONICAL_SECRET_FORMAT = """
{
  secrets: {
    username: {
      value: some_username
    },
    password: {
      value: some_password
    },
  }
}
"""


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

    def _parse_raw_auth_data_valid_v1(self, raw_auth_data: Any) -> DockerAuthInternal | None:
        LOGGER.debug("Parsing raw auth data from yt client, v1")
        # just backward compatibility
        if not isinstance(raw_auth_data, dict):
            LOGGER.debug("Skip because auth data is not dict")
            return None
        if set(raw_auth_data.keys()) != {"username", "password"} and set(raw_auth_data.keys()) != {"auth"}:
            LOGGER.debug("Skip because of invalid keys: %s", raw_auth_data.keys())
            return None
        if not all([isinstance(value, str) for value in raw_auth_data.values()]):
            LOGGER.debug("Skip because of invalid values types")
            return None
        return DockerAuthInternal(
            username=raw_auth_data.get("username"),
            password=raw_auth_data.get("password"),
            auth=raw_auth_data.get("auth"),
        )

    def _parse_raw_auth_data_valid_v2(self, raw_auth_data: Any) -> DockerAuthInternal | None:
        LOGGER.debug("Parsing raw auth data from yt client, v2")
        if not isinstance(raw_auth_data, dict):
            LOGGER.debug("Skip because auth data is not dict")
            return None
        secrets = raw_auth_data.get("secrets", {})
        if not isinstance(secrets, dict):
            LOGGER.debug("Skip because there is no 'secrets' key at the top level")
            return None
        if set(secrets.keys()) != {"username", "password"} and set(secrets.keys()) != {"auth"}:
            LOGGER.debug("Skip because of invalid keys: %s", secrets.keys())
            return None
        if not all([isinstance(value, dict) for value in secrets.values()]):
            LOGGER.debug("Skip because of invalid values types")
            return None
        if not all([isinstance(value.get("value"), str) for value in secrets.values()]):
            LOGGER.debug("Skip because of invalid values types")
            return None
        return DockerAuthInternal(
            username=secrets.get("username", {}).get("value"),
            password=secrets.get("password", {}).get("value"),
            auth=secrets.get("auth", {}).get("value"),
        )

    def extract(self, auth_data: DockerAuthData) -> DockerAuthInternal:
        match auth_data:
            case DockerAuthSecret():
                assert isinstance(auth_data, DockerAuthSecret)  # skip type warning in pycharm
                LOGGER.debug("Parsing raw auth data from yt client %s", auth_data.cypress_path)
                raw_auth_data = self._yt_client.get(auth_data.cypress_path)
                if parsed_auth_data := self._parse_raw_auth_data_valid_v2(raw_auth_data):
                    return parsed_auth_data
                elif parsed_auth_data := self._parse_raw_auth_data_valid_v1(raw_auth_data):
                    warnings.warn(f"Please use new docker secret format {CANONICAL_SECRET_FORMAT}")
                    return parsed_auth_data
                # TODO: describe how to create new secret in UI
                raise DockerAuthDataError(f"Unknown secret format, please use {CANONICAL_SECRET_FORMAT}")
            case _:
                raise DockerAuthDataError("Unknown auth data type")
