.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: all
all: ## runs the targets: cl, env, build
	make cl
	make env
	make build

.PHONY: cl
cl: ## create conda lock for multiple platforms
	# the linux-aarch64 is used for ARM Macs using linux docker container
	conda-lock lock \
		--file environment.yml \
		-p linux-64 \
		-p osx-64 \
		-p osx-arm64 \
		-p win-64 \

<<<<<<< Updated upstream
=======
.PHONY: docker-up docker-down docker-shell docker-build docker-up-shell

docker-up: ## Start Docker container in detached mode
	docker compose up -d

docker-up-shell: ## Start container and open bash shell
	docker compose up -d
	docker compose exec proj-522 bash

docker-down: ## Stop Docker container
	docker compose down

docker-shell: ## Open bash shell in running container
	docker compose exec proj-522 bash

docker-build: ## Build Docker image locally
	docker compose build

all: reports/term-deposit-predictor-analysis.html reports/term-deposit-predictor-analysis.pdf ## Run full analysis pipeline
>>>>>>> Stashed changes

.PHONY: env
env: ## remove previous and create environment from lock file
	# remove the existing env, and ignore if missing
	conda env remove dockerlock || true
	conda-lock install -n dockerlock conda-lock.yml

.PHONY: build
build: ## build the docker image from the Dockerfile
	docker build -t dockerlock --file Dockerfile .

.PHONY: run
run: ## alias for the up target
	make up

.PHONY: up
up: ## stop and start docker-compose services
	# by default stop everything before re-creating
	make stop
	docker-compose up -d

<<<<<<< Updated upstream
.PHONY: stop
stop: ## stop docker-compose services
	docker-compose stop
=======
# Step 5: Evaluate model
results/test_metrics.csv results/classification_report.csv results/confusion_matrix.csv: data/processed/X_test_transformed.csv \
data/processed/y_test.csv \
results/models/logistic_regression_model.pkl \
scripts/05_evaluate_model.py
	python scripts/05_evaluate_model.py \
		--test-dir=data/processed \
		--model-dir=results/models \
		--output-dir=results

# Step 6: Generate HTML report
reports/term-deposit-predictor-analysis.html: reports/term-deposit-predictor-analysis.qmd \
results/test_metrics.csv \
results/figures/feature_distributions.png
	quarto render reports/term-deposit-predictor-analysis.qmd --to html

# Step 7: Generate PDF report
reports/term-deposit-predictor-analysis.pdf: reports/term-deposit-predictor-analysis.qmd \
results/test_metrics.csv \
results/figures/feature_distributions.png \
results/classification_report.csv \
results/confusion_matrix.csv \
results/figures/feature_correlations.png \
results/figures/summary_statistics.csv
	quarto render reports/term-deposit-predictor-analysis.qmd --to pdf
>>>>>>> Stashed changes

.PHONY: clean

clean: ## remove all generated data and results
	rm -rf data/raw data/processed results
