.PHONY: help install clean run check test

# =========================
# Help
# =========================
help:
	@echo "Available commands:"
	@echo ""
	@echo "🛠️  Environment & Setup"
	@echo "  make install       - Install dependencies & setup pre-commit"
	@echo ""
	@echo "🚀 Development"
	@echo "  make run           - Run FastAPI development server"
	@echo ""
	@echo "🧪 Quality & CI"
	@echo "  make check         - Run lint, type check, dependency check (CI-like)"
	@echo "  make test          - Run tests with coverage"
	@echo ""
	@echo "🧹 Maintenance"
	@echo "  make clean         - Clean cache and build artifacts"

# =========================
# Install dependencies
# =========================
install:
	@echo "🚀 Installing dependencies using uv"
	@uv sync
	@echo "🚀 Installing pre-commit hooks"
	@uv run pre-commit install
	@echo "✅ Note: Run 'uv lock' separately if you modified pyproject.toml"

# =========================
# Development
# =========================
run:
	@echo "🚀 Starting FastAPI development server"
	@uv run fastapi dev

# =========================
# Quality / CI checks
# =========================
check:
	@echo "🚀 Checking lock file consistency"
	@uv lock --locked
	@echo "🚀 Running pre-commit checks"
	@uv run pre-commit run -a
	@echo "🚀 Running static type checks (mypy)"
	@uv run mypy app
	@echo "🚀 Checking for obsolete dependencies (deptry)"
	@uv run deptry app

test:
	@echo "🚀 Running tests with coverage"
	@uv run python -m pytest --cov=app --cov-report=xml

# =========================
# Cleanup
# =========================
clean:
	@echo "🧹 Cleaning cache and build artifacts"
	find . -type d -name "__pycache__" -exec rm -rf {} \; 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} \; 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} \; 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} \; 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf dist/ build/ .coverage htmlcov/
