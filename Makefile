SHELL=/bin/bash
PATHS=./

lint:
	flake8 ${LINT_PATHS} --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 ${LINT_PATHS} --count --exit-zero --statistics

format:
	isort ${LINT_PATHS}
	black -l 127 ${LINT_PATHS}

check-codestyle:
	isort --check ${LINT_PATHS}
	black --check -l 127 ${LINT_PATHS}