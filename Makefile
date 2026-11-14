PYTHON ?= .venv/bin/python
TESTS ?= tests
PYTEST_ARGS ?=

.PHONY: qa test test-host test-qualified preflight

qa:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m pyright --pythonpath $(PYTHON)

test:
	MLX_ENABLE_TF32=0 $(PYTHON) -m pytest $(TESTS) $(PYTEST_ARGS)

test-host:
	$(MAKE) test PYTEST_ARGS='-m "host and not qualification" $(PYTEST_ARGS)'

preflight:
	$(PYTHON) scripts/preflight.py

test-qualified: preflight
	$(MAKE) test
