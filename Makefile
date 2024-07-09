.PHONY: black-check black-fix ruff-check ruff-fix isort-check all-check all-fix

black-check:
	python -m black --check .

black-fix:
	python -m black .

ruff-check:
	ruff check

ruff-fix:
	ruff check --fix

isort-check:
	isort --profile black --check-only .

mypy-check:
	mypy ./tractorun ./tests ./tools ./examples/jax ./examples/pytorch

mypy-papyrax-check:
	mypy ./examples/papyrax

isort-fix:
	isort --profile black .

all-check: black-check ruff-check isort-check mypy-check

all-all-check: black-check ruff-check isort-check mypy-check mypy-papyrax-check

all-fix: black-fix ruff-fix isort-fix
