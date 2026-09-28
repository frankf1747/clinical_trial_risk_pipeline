-include .env

.PHONY: setup test ingest-aact ingest-faers trials faers match check-faers attributes upload warehouse train score

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

attributes:
	uv run python -m ctrisk.spark.trial_attributes

# Mirror the Parquet Snowflake loads to GCS (~75 MB). Deleting stale objects matters:
# Spark part-file names change every run, and leftovers would load twice.
upload:
	@test -n "$(strip $(GCP_BUCKET))" || (echo "GCP_BUCKET is not set in .env" && exit 1)
	gcloud storage rsync --recursive --delete-unmatched-destination-objects \
	  --exclude='.*\.crc$$|.*_SUCCESS$$|^drug_interventions/.*' \
	  data/parquet gs://$(strip $(GCP_BUCKET))/parquet

warehouse:
	uv run python -m ctrisk.warehouse.snowflake

train:
	uv run python -m ctrisk.ml.train

score:
	uv run python -m ctrisk.ml.score
