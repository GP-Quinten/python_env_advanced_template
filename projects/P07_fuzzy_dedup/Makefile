# Makefile to manage local Weaviate (uses docker-compose.yaml)

.PHONY: up start stop down restart logs status wait_weaviate

COMPOSE_FILE := docker-compose.yaml

up: ## Start weaviate and wait until healthy
	docker compose -f $(COMPOSE_FILE) up -d weaviate || { echo "[ERROR] docker compose up failed"; exit 1; }
	@$(MAKE) wait_weaviate
	@echo "[SUCCESS] Weaviate started."

start: up

stop: ## Stop weaviate container
	docker compose -f $(COMPOSE_FILE) stop weaviate || { echo "[ERROR] docker compose stop failed"; exit 1; }
	@echo "[SUCCESS] Container stopped."

down: ## Stop and remove container, network and volume
	docker compose -f $(COMPOSE_FILE) down -v || { echo "[ERROR] docker compose down failed"; exit 1; }
	@echo "[SUCCESS] Container removed."

restart: stop up

logs: ## Follow logs for weaviate
	docker compose -f $(COMPOSE_FILE) logs -f weaviate

status: ## Show docker-compose ps
	docker compose -f $(COMPOSE_FILE) ps || { echo "[ERROR] docker-compose ps failed"; exit 1; }
	@echo "[SUCCESS] docker-compose ps displayed."

wait_weaviate: ## Wait until Weaviate HTTP API responds (timeout ~30s)
	@echo "Waiting for weaviate (http://localhost:8889) to respond..."
	i=0; until curl -fsS http://localhost:8889/v1/.well-known/ready >/dev/null 2>&1 || curl -fsS http://localhost:8889/v1/.well-known/live >/dev/null 2>&1 || [ $$i -ge 30 ]; do \
		sleep 1; \
		echo -n "."; \
		i=$$((i+1)); \
	done; \
	if [ $$i -ge 30 ]; then \
		echo "[ERROR] Timed out waiting for weaviate"; \
		echo "[FAIL] weaviate did not start responding."; \
		exit 1; \
	fi; \
	echo "[SUCCESS] weaviate is responding."
