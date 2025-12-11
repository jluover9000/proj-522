.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: cl
cl: ## create conda lock for multiple platforms
	# the linux-aarch64 is used for ARM Macs using linux docker container
	conda-lock lock \
		--file environment.yml \
		-p linux-64 \
		-p osx-64 \
		-p osx-arm64 \
		-p win-64 \

all: reports/term-deposit-predictor-analysis.html reports/term-deposit-predictor-analysis.pdf ## Run full analysis pipeline

# Step 1: Download and extract data
data/raw/bank_marketing_features.csv data/raw/bank_marketing_targets.csv: scripts/01_download_data.py
	python scripts/01_download_data.py \
		--dataset-id=222 \
		--output-dir=data/raw

# Step 2: Preprocess, validate, and split data
data/processed/X_train_transformed.csv data/processed/X_test_transformed.csv data/processed/y_train.csv data/processed/y_test.csv data/processed/X_train_unprocessed.csv: data/raw/bank_marketing_features.csv \
data/raw/bank_marketing_targets.csv \
scripts/02_clean_validate_preprocess.py
	python scripts/02_clean_validate_preprocess.py \
		--input-dir=data/raw \
		--output-dir=data/processed

# Step 3: Exploratory Data Analysis
results/figures/feature_distributions.png results/figures/feature_correlations.png results/figures/summary_statistics.csv: data/processed/X_train_unprocessed.csv \
data/processed/y_train.csv scripts/03_eda.py
	python scripts/03_eda.py \
		--input-dir=data/processed \
		--output-dir=results/figures

# Step 4: Train model
results/models/logistic_regression_model.pkl results/models/label_encoder.pkl results/models/cv_results.csv: data/processed/X_train_transformed.csv \
data/processed/y_train.csv \
scripts/04_fit_model.py
	python scripts/04_fit_model.py \
		--input-dir=data/processed \
		--output-dir=results/models

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
results/figures/feature_distributions.png
	quarto render reports/term-deposit-predictor-analysis.qmd --to pdf

.PHONY: clean
clean: ## Remove all generated data and results
	rm -rf data/raw data/processed results
