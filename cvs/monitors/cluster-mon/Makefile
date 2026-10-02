# Standalone Go cluster-mon. From this directory: `make ut`, `make build`.
# The repo-root Makefile invokes `ut` here from `make ut` / `make test`.

GO          ?= go
MODULE      := ./...
BIN_DIR     := bin
BIN         := $(BIN_DIR)/cluster-mon
LISTEN      ?= :8001
CONFIG_DIR  ?= ./config
STATIC_DIR  ?=
CMD         := ./cmd/cluster-mon

.PHONY: help ut test build tidy vet fmt fmt-check run docker-build docker-up docker-down clean

help:
	@echo "cluster-mon Go targets:"
	@echo "  ut           - Run Go unit tests (go test ./...)"
	@echo "  test         - Same as ut"
	@echo "  build        - Build $(BIN)"
	@echo "  tidy         - go mod tidy"
	@echo "  vet          - go vet ./..."
	@echo "  fmt          - gofmt -w on Go sources"
	@echo "  fmt-check    - gofmt -l (fail if files need formatting)"
	@echo "  run          - go run the server (LISTEN=$(LISTEN) CONFIG_DIR=$(CONFIG_DIR))"
	@echo "  docker-build - docker compose build"
	@echo "  docker-up    - docker compose up -d"
	@echo "  docker-down  - docker compose down"
	@echo "  clean        - Remove $(BIN_DIR)/"

ut:
	@echo "Unit testing cluster-mon (Go)..."
	$(GO) test $(MODULE)

test: ut

build:
	@mkdir -p $(BIN_DIR)
	$(GO) build -o $(BIN) $(CMD)

tidy:
	$(GO) mod tidy

vet:
	$(GO) vet $(MODULE)

fmt:
	gofmt -w cmd internal

fmt-check:
	@out=$$(gofmt -l cmd internal); \
	if [ -n "$$out" ]; then \
		echo "gofmt needed:"; echo "$$out"; exit 1; \
	fi

run:
	$(GO) run $(CMD) -listen $(LISTEN) -config-dir $(CONFIG_DIR) $(if $(STATIC_DIR),-static $(STATIC_DIR),)

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	rm -rf $(BIN_DIR)
