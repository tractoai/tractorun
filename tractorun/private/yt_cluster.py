from typing import Any
from urllib.parse import quote
import warnings

import attrs
import yt.wrapper as yt


def _to_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def make_cypress_link(cypress_link_template: str | None, path: str) -> str | None:
    if cypress_link_template is None:
        return None
    return cypress_link_template.format(path=quote(path))


def make_job_stderr_link(job_stderr_link_template: str | None, operation_id: str, job_id: str) -> str | None:
    if job_stderr_link_template is None:
        return None
    return job_stderr_link_template.format(
        job_id=quote(job_id),
        operation_id=quote(operation_id),
    )


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TractorunClusterConfig:
    cypress_link_template: str | None
    job_stderr_link_template: str | None

    @staticmethod
    def load_from_yt(yt_client: yt.YtClient, path: str, fail_if_not_exist: bool = False) -> "TractorunClusterConfig":
        if not fail_if_not_exist and not yt_client.exists(path):
            warnings.warn(
                f"Cluster config {path} does not exist. Some functions are not available. Please specify config's path by tractorun params.",
            )
            return TractorunClusterConfig(
                cypress_link_template=None,
                job_stderr_link_template=None,
            )

        config = yt_client.get(path)
        if not config:
            config = {}

        return TractorunClusterConfig(
            cypress_link_template=_to_str(config.get("cypress_link_template")),
            job_stderr_link_template=_to_str(config.get("job_stderr_link_template")),
        )

    def to_dict(self) -> dict:
        return attrs.asdict(self)  # type: ignore
