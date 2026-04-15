.PHONY: install install-runtime install-dev train train-cheap train-expensive test lint format typecheck predict process

PYTHON_TARGETS=apps src tests pages scripts monitoring app.py api.py main.py predict_customer.py save_processed_data.py

install:
	.venv\Scripts\python.exe -m pip install -r requirements.txt

install-runtime:
	.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt

install-dev:
	.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

train:
	.venv\Scripts\python.exe -m src.cli.pipeline --seed 42 --data-dir data --log-level INFO --decision-policy balanceada

train-cheap:
	.venv\Scripts\python.exe -m src.cli.pipeline --seed 42 --data-dir data --log-level INFO --decision-policy campanha_barata

train-expensive:
	.venv\Scripts\python.exe -m src.cli.pipeline --seed 42 --data-dir data --log-level INFO --decision-policy campanha_cara

test:
	.venv\Scripts\python.exe -m pytest -q

lint:
	.venv\Scripts\python.exe -m ruff check $(PYTHON_TARGETS)
	.venv\Scripts\python.exe -m black --check $(PYTHON_TARGETS)
	.venv\Scripts\python.exe -m isort --check-only $(PYTHON_TARGETS)

format:
	.venv\Scripts\python.exe -m black $(PYTHON_TARGETS)
	.venv\Scripts\python.exe -m isort $(PYTHON_TARGETS)

typecheck:
	.venv\Scripts\python.exe -m mypy

predict:
	.venv\Scripts\python.exe -m src.cli.predict_customer

process:
	.venv\Scripts\python.exe -m src.cli.save_processed_data
