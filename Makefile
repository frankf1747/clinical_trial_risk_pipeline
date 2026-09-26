.PHONY: setup test ingest-aact trials

setup:
	uv sync

test:
	uv run pytest -q

# Usage: make ingest-aact AACT=<url-or-path-to-aact-zip>
ingest-aact:
	uv run python -m ctrisk.ingest.aact $(AACT)

trials:
	uv run python -m ctrisk.spark.clean_trials
