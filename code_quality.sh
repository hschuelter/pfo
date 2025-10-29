#!/bin/bash
echo "Checking code formatting..."
black --check src/ tests/

echo "Checking import sorting..."
isort --check-only src/ tests/

echo "Type check..."
mypy src/ --ignore-missing-imports