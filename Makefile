install:
	pip install -r requirements.txt

train:
	python scripts/train.py

run-api:
	uvicorn app.app:app --reload --host 0.0.0.0 --port 8000

test-api:
	pytest tests/test_api.py
