# SPDX-FileCopyrightText: 2019–2026 Pynguin Contributors
#
# SPDX-License-Identifier: MIT

IMAGE := pynguin
VERSION=$(shell git rev-parse --short HEAD)

.PHONY: build_docker_benchmarks
build_docker_benchmarks:
	@echo Building docker $(IMAGE)-benchmark:$(VERSION) ...
	cd pynguin && \
	docker build \
	  -t $(IMAGE)-benchmark:$(VERSION) . \
	  -f ./benchmark/Dockerfile
	@echo Building docker $(IMAGE)-base-benchmark:$(VERSION) ...
	cd pynguin-base-mod && \
	docker build \
	  -t $(IMAGE)-base-benchmark:$(VERSION) . \
	  -f ./benchmark/Dockerfile

.PHONY: clean_docker_benchmarks
clean_docker_benchmarks:
	@echo Removing docker $(IMAGE)-benchmark:$(VERSION) ...
	docker rmi -f $(IMAGE)-benchmark:$(VERSION)
	@echo Removing docker $(IMAGE)-base-benchmark:$(VERSION) ...
	docker rmi -f $(IMAGE)-base-benchmark:$(VERSION)

.PHONY: run_docker_benchmarks
run_docker_benchmarks:
	@echo Running docker benchmark
	cd pynguin/benchmark && \
	docker compose up --build --abort-on-container-exit --exit-code-from benchmark
	@echo Running docker benchmark
	cd pynguin-base-mod/benchmark && \
	docker compose up --build --abort-on-container-exit --exit-code-from benchmark

.PHONY: export_docker_benchmarks
export_docker_benchmarks: build_docker_benchmarks
	@echo Exporting docker benchmark containers to tar files
	docker save $(IMAGE)-benchmark:$(VERSION) > $(IMAGE)-benchmark-$(VERSION).tar
	docker save $(IMAGE)-base-benchmark:$(VERSION) > $(IMAGE)-base-benchmark-$(VERSION).tar
