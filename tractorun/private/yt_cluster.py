import warnings

import attrs
import yt.wrapper as yt


def make_cypress_link(cypress_link_template: str | None, path: str) -> str | None:
    if cypress_link_template is None:
        return None
    return cypress_link_template.format(path=path)


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TractorunClusterConfig:
    cypress_link_template: str | None

    @staticmethod
    def load_from_yt(yt_client: yt.YtClient, path: str, fail_if_not_exist: bool = False) -> "TractorunClusterConfig":
        if not fail_if_not_exist and not yt_client.exists(path):
            warnings.warn(
                f"Cluster config {path} does not exist. Some functions are not available. Please specify config's path by tractorun params.",
            )
            return TractorunClusterConfig(
                cypress_link_template=None,
            )

        config = yt_client.get(path)

        return TractorunClusterConfig(
            cypress_link_template=str(config["cypress_link_template"]),
        )

    def to_dict(self) -> dict:
        return attrs.asdict(self)  # type: ignore
