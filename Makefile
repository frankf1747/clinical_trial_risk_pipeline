.PHONY: setup test ingest-aact ingest-faers trials faers match check-faers

setup:
	uv sync

test:
	uv run pytest -q

# Usage: make ingest-aact AACT=<url-or-path-to-aact-zip>
ingest-aact:
	uv run python -m ctrisk.ingest.aact $(AACT)

ingest-faers:
	uv run python -m ctrisk.ingest.faers

trials:
	uv run python -m ctrisk.spark.clean_trials

faers:
	uv run python -m ctrisk.spark.flatten_faers

match:
	uv run python -m ctrisk.spark.match_drugs

check-faers:
	uv run python -m ctrisk.checks.faers_api
