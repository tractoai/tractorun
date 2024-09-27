import attrs
import yt.wrapper as yt


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TractorunClusterConfig:
    cypress_link_template: str

    @staticmethod
    def load_from_yt(yt_client: yt.YtClient, path: str) -> "TractorunClusterConfig":
        config = yt_client.get(path)

        return TractorunClusterConfig(
            cypress_link_template=str(config["cypress_link_template"]),
        )

    def to_dict(self) -> dict:
        return attrs.asdict(self)  # type: ignore
