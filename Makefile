.DEFAULT_GOAL := help
SHELL := /bin/bash

BACKEND ?= http://localhost:8080

.PHONY: help
help: ## Buyruqlar ro'yxati
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## .env yaratadi, sirlarni to'ldiradi va anti-spoofing vaznlarini yuklaydi
	@test -f .env || cp .env.example .env
	@$(MAKE) --no-print-directory secrets
	@bash python-ai-service/scripts/download_models.sh
	@echo "Tayyor. Endi: make up"

.PHONY: secrets
secrets: ## .env dagi bo'sh sirlarni tasodifiy qiymatlar bilan to'ldiradi
	@# Faqat BO'SH qatorlarni to'ldiradi. Mavjud FACE_ID_PEPPER hech qachon
	@# qayta yozilmaydi — uni almashtirish butun yuz bazasini kuchsizlantiradi.
	@test -f .env || cp .env.example .env
	@for key in FACE_ID_PEPPER ADMIN_BOOTSTRAP_KEY; do \
		current=$$(grep -E "^$$key=" .env | head -1 | cut -d= -f2-); \
		if [ -n "$$current" ]; then \
			echo "  $$key — allaqachon o'rnatilgan, tegilmadi"; \
			continue; \
		fi; \
		value=$$(openssl rand -base64 48 | tr -d '\n=' | tr '+/' '-_'); \
		if grep -qE "^$$key=" .env; then \
			sed -i "s|^$$key=.*|$$key=$$value|" .env; \
		else \
			printf '%s=%s\n' "$$key" "$$value" >> .env; \
		fi; \
		echo "  $$key — yasaldi"; \
	done
	@echo "⚠️  FACE_ID_PEPPER ning zaxira nusxasini saqlang: uni yo'qotish = bazani qayta to'ldirish."

.PHONY: proto
proto: ## .proto dan Go va Python kodini qayta generatsiya qiladi
	@# grpcio-tools versiyasi requirements-dev.txt ga MOS bo'lishi shart —
	@# skript buni o'zi tekshiradi va mos kelmasa to'xtaydi.
	bash scripts/gen-proto.sh

.PHONY: grpc-list
grpc-list: ## AI servisning gRPC metodlarini ko'rsatadi (grpcurl kerak)
	grpcurl -plaintext localhost:$(or $(AI_GRPC_PORT),50051) list face.v1.FaceService

.PHONY: pull
pull: ## Base image'larni oldindan yuklab oladi (tarmoq beqaror bo'lsa foydali)
	@# Build paytida registry'ga murojaat qilinmasligi uchun. Har biri alohida —
	@# bittasi yiqilsa qolganlari yuklanaveradi.
	-docker pull python:3.11-slim
	-docker pull golang:1.25-alpine
	-docker pull alpine:3.20
	-docker pull qdrant/qdrant:v1.12.4
	-docker pull postgres:16-alpine

.PHONY: up
up: ## Barcha servislarni ishga tushiradi
	docker compose up -d --build

.PHONY: up-offline
up-offline: ## Build'siz ishga tushiradi (image'lar allaqachon qurilgan bo'lsa)
	docker compose up -d

.PHONY: down
down: ## Servislarni to'xtatadi
	docker compose down

.PHONY: clean
clean: ## Servislarni va MA'LUMOTLARNI o'chiradi (qaytarib bo'lmaydi!)
	docker compose down -v

.PHONY: logs
logs: ## Loglarni kuzatadi
	docker compose logs -f

.PHONY: ps
ps: ## Konteynerlar holati
	docker compose ps

.PHONY: health
health: ## Tizim holatini tekshiradi
	@curl -s $(BACKEND)/health | python3 -m json.tool

.PHONY: verify-liveness
verify-liveness: ## Anti-spoofing modeli to'g'ri ishlayotganini tekshiradi
	docker compose exec ai-service python scripts/verify_liveness.py

.PHONY: test-go
test-go: ## Golang unit testlari (Qdrant kerak emas)
	cd golang-backend && go vet ./... && go test ./...

.PHONY: test-integration
test-integration: ## To'liq testlar — vaqtinchalik Qdrant ko'taradi (auth holati ham)
	@docker rm -f qdrant-itest qdrant-itest-auth >/dev/null 2>&1 || true
	@docker run -d --name qdrant-itest -p 16333:6333 qdrant/qdrant:v1.12.4 >/dev/null
	@docker run -d --name qdrant-itest-auth -e QDRANT__SERVICE__API_KEY=itest-key \
		-p 16335:6333 qdrant/qdrant:v1.12.4 >/dev/null
	@echo "Qdrant kutilmoqda..."; sleep 9
	@cd golang-backend && \
		QDRANT_TEST_URL=http://localhost:16333 \
		QDRANT_AUTH_TEST_URL=http://localhost:16335 \
		QDRANT_AUTH_TEST_KEY=itest-key \
		go test ./... -v; \
		status=$$?; docker rm -f qdrant-itest qdrant-itest-auth >/dev/null; exit $$status

# ============================================================================
#  DOCKER'SIZ REJIM (bare-metal / systemd)
#
#  Yuqoridagi `up`/`down` Docker uchun; quyidagilar to'g'ridan-to'g'ri serverga
#  o'rnatish uchun. Ikkalasi bir-biriga xalaqit qilmaydi, lekin BIR VAQTDA
#  ishlatib bo'lmaydi — portlar (8080, 6333, 5432) to'qnashadi.
#  Batafsil: DOCKERSIZ_ISHGA_TUSHIRISH.md
# ============================================================================

.PHONY: bare-install
bare-install: ## [Docker'siz] Serverga to'liq o'rnatish (root kerak)
	sudo bash deploy/bare-metal/install.sh

.PHONY: bare-bootstrap
bare-bootstrap: ## [Docker'siz] Birinchi FACE_API_KEY ni oladi
	sudo bash deploy/bare-metal/install.sh bootstrap

.PHONY: bare-update
bare-update: ## [Docker'siz] Yangi kodni yoyadi va servislarni qayta ishga tushiradi
	sudo bash deploy/bare-metal/install.sh code
	sudo bash deploy/bare-metal/install.sh build
	sudo systemctl restart face-ai face-backend
	@sleep 5
	sudo bash deploy/bare-metal/install.sh status

.PHONY: bare-status
bare-status: ## [Docker'siz] Servislar holati va endpointlar
	sudo bash deploy/bare-metal/install.sh status

.PHONY: bare-logs
bare-logs: ## [Docker'siz] Uchala servis loglari
	sudo journalctl -u qdrant -u face-ai -u face-backend -f

.PHONY: bare-restart
bare-restart: ## [Docker'siz] Servislarni qayta ishga tushiradi
	sudo systemctl restart qdrant face-ai face-backend

.PHONY: bare-stop
bare-stop: ## [Docker'siz] Servislarni to'xtatadi (Docker rejimiga o'tishdan oldin)
	sudo systemctl stop face-backend face-ai qdrant

.PHONY: register
register: ## Ro'yxatga qo'shish: make register NAME="Ali Valiyev" IMG=ali.jpg [ID=...]
	@curl -s -X POST $(BACKEND)/api/v1/persons/register \
		$(if $(ID),-F "person_id=$(ID)") -F "name=$(NAME)" -F "image=@$(IMG)" | python3 -m json.tool

.PHONY: verify
verify: ## Tekshirish: make verify IMG=frame.jpg
	@curl -s -w "\nHTTP %{http_code}\n" -X POST $(BACKEND)/api/v1/persons/verify \
		-F "image=@$(IMG)" | python3 -m json.tool 2>/dev/null || true

.PHONY: persons
persons: ## Ro'yxatdan o'tganlar
	@curl -s $(BACKEND)/api/v1/persons | python3 -m json.tool
