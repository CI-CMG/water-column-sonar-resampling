# water-column-sonar-resampling
Water Column Sonar Data Reprocessing Project for Warren Tech Capstone 2026

## Reminders
Please update the patch number in `pyproject.toml` !!!

## UV Commands
To get the most fresh copy of the project run `uv venv` and then run the source command given `source .venv/bin/activate` (for Linux) -- opens a venv with all of the most up to date packages and code. Be sure to run `uv sync` afterwards if it's a brand new enviorment.

To add a new package to the enviorment you can install it inside of the venv or use `uv add <package name>` followed by `uv sync` instead.

To sync packages like Pytest that are dev dependencies use `uv sync --all-extras` after your `uv sync`

To run Pytest as it would appear in Github make sure all packages are synced and then run `uv run pytest`

## Tagging Commands

`git tag -a v25.11.x -m "Releasing v25.11.x"`

`git push origin --tags`

_Make sure to increase x by one every time. Will reset when a new month starts (December will reset to 25.12.0)_

If you forget to tag a push use the following command:

`git tag -a v25.9.x [(part of or all) commit checksum] -m "Releasing v25.9.x"`