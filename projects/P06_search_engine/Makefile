SHELL := /bin/bash

# -------------------------------------------------------------------
# Weaviate
# -------------------------------------------------------------------
.PHONY: weaviate-up weaviate-down

weaviate-up:
	docker compose up -d weaviate
	@echo "✅ Weaviate is starting on http://localhost:8080"

weaviate-down:
	docker compose down weaviate
	@echo "🛑 Weaviate stopped"

# -------------------------------------------------------------------
# MLflow
# -------------------------------------------------------------------
.PHONY: mlflow-up mlflow-down

mlflow-up:
	poetry run mlflow ui --backend-store-uri file:mlruns/ --port 5000 &
	@echo "✅ MLflow UI running on http://localhost:5000"

mlflow-down:
	@pkill -f "mlflow ui" || true
	@echo "🛑 MLflow stopped"
