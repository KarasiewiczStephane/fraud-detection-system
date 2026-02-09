.PHONY: install test lint clean run docker docker-up docker-down docker-logs

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=src

lint:
	ruff check src/ tests/
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000

docker:
	docker build -t fraud-detection-api .

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f
