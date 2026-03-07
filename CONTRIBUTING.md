# Contributing to Football Quant Engine

Thanks for your interest in contributing.

## Recommended workflow

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Run lint and tests
5. Open a pull request

## Code style

This project uses:

- black
- isort
- ruff
- pytest

Before submitting changes, run:

python -m pytest -q
ruff check .
black --check .
isort --check-only .

## Suggested contribution areas

- mathematical models
- offline engine improvements
- CSV import/export
- UI improvements
- reporting
- testing
- documentation

