PYTHON := .venv/bin/python

format:
	@set -e
	@echo "Formatting code with ruff..."
	@$(PYTHON) -m ruff format .
	@$(PYTHON) -m ruff check --fix .
	@echo "Formatting completed."