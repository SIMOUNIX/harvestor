.PHONY: help install test test-cov test-quick lint format clean

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install package with dev dependencies
	uv sync --extra dev

test: ## Run all tests with verbose output
	pytest -v

test-cov: ## Run tests with coverage report
	pytest --cov=src/harvestor --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

test-quick: ## Run tests without coverage (faster)
	pytest -x --tb=short

test-watch: ## Run tests in watch mode (requires pytest-watch)
	pytest-watch

lint: ## Run linting checks
	ruff check src/ tests/

format: ## Format code with black
	black src/ tests/
	ruff check --fix src/ tests/

clean: ## Clean up generated files
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf **/__pycache__
	rm -rf **/*.pyc
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

.DEFAULT_GOAL := help
