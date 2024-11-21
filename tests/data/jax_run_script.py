import sys

from jax import (
    grad,
    jit,
)

from tractorun.backend.tractorax import Tractorax
from tractorun.run import prepare_and_get_toolbox


def main() -> None:
    _ = prepare_and_get_toolbox(backend=Tractorax())

    @jit
    def f(x: int) -> int:
        return x**2 + 3 * x + 1

    grad_f = grad(f)
    print(grad_f(1.0), file=sys.stderr)


if __name__ == "__main__":
    main()
