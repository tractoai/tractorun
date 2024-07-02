import attrs


@attrs.define
class Bind:
    source: str
    destination: str
