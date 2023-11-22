.PHONY: format verify_format test

format:
	isort ./cool_graph/ ./tests/
	black ./cool_graph/ ./tests/
	no_implicit_optional ./cool_graph/ ./tests/

verify_format:
	isort --check --diff ./cool_graph/ ./tests/
	black --check --diff ./cool_graph/ ./tests/

test:
	coverage run --source=cool_graph -m pytest . && coverage report -m

