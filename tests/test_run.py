from tests.yt_instances import YtInstance


def test_simple(yt_instance: YtInstance, mnist_ds_path: str) -> None:
    yt_cli = yt_instance.get_client()
    row_count = yt_cli.get(f"{mnist_ds_path}/@row_count")
    assert row_count == 100
