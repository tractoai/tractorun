.PHONY: black-check black-fix ruff-check ruff-fix isort-check all-check all-fix

black-check:
	python -m black --check .

black-fix:
	python -m black .

ruff-check:
	ruff check

ruff-fix:
	ruff --fix

isort-check:
	isort --check-only .

isort-fix:
	isort .

all-check: black-check ruff-check isort-check

all-fix: black-fix ruff-fix isort-fix
