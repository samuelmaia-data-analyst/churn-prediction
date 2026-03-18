.PHONY: install train train-cheap train-expensive test lint format typecheck predict process

install:
	.venv\Scripts\python.exe -m pip install -r requirements.txt

train:
	.venv\Scripts\python.exe -m src.cli.pipeline --seed 42 --data-dir data --log-level INFO --decision-policy balanceada

train-cheap:
	.venv\Scripts\python.exe -m src.cli.pipeline --seed 42 --data-dir data --log-level INFO --decision-policy campanha_barata

train-expensive:
	.venv\Scripts\python.exe -m src.cli.pipeline --seed 42 --data-dir data --log-level INFO --decision-policy campanha_cara

test:
	.venv\Scripts\python.exe -m pytest -q

lint:
	.venv\Scripts\python.exe -m ruff check app.py api.py main.py predict_customer.py save_processed_data.py apps src tests pages
	.venv\Scripts\python.exe -m black --check app.py api.py main.py predict_customer.py save_processed_data.py apps src tests pages
	.venv\Scripts\python.exe -m isort --check-only app.py api.py main.py predict_customer.py save_processed_data.py apps src tests pages

format:
	.venv\Scripts\python.exe -m black app.py api.py main.py predict_customer.py save_processed_data.py apps src tests pages
	.venv\Scripts\python.exe -m isort app.py api.py main.py predict_customer.py save_processed_data.py apps src tests pages

typecheck:
	.venv\Scripts\python.exe -m mypy src

predict:
	.venv\Scripts\python.exe -m src.cli.predict_customer

process:
	.venv\Scripts\python.exe -m src.cli.save_processed_data
