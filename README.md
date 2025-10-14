# water-column-sonar-resampling
Water Column Sonar Data Reprocessing Project for Warren Tech Capstone 2026

## Reminders
Please update the patch number in `pyproject.toml` !!!

To get the most fresh copy of the project run `uv venv` and then sun the source command given `source .venv/bin/activate` -- opens a venv with all of the most up to date packages and code.

To add a new package to the enviorment you can install it inside of the venv or use `uv add <package name>` followed by `uv sync` instead.

To sync packages like Pytest that are dev dependencies use `uv sync --all-extras` after your `uv sync`

To run Pytest as it would appear in Github make sure all packages are synced and then run `uv run pytest`