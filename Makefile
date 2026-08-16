# Why this file exists: exposes repeatable local and CI pipeline commands.
# Responsible for: composing independently runnable stage entrypoints.
# Must not: contain pipeline logic or bypass a stage's own error handling.
.PHONY: install validate features train test test-validation test-features test-training pipeline

install:
	pip install -r requirements.txt

validate:
	python -m src.validation.validate_data

features:
	python -m src.features.build_features

train:
	python -m src.training.train_model

test:
	pytest tests/

test-validation:
	pytest tests/test_validate_data.py -v --tb=short

test-features:
	pytest tests/test_build_features.py -v --tb=short

test-training:
	pytest tests/test_train_model.py -v --tb=short

pipeline: validate features train
	@echo "Pipeline stages complete."
