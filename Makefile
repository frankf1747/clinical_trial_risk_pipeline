-include .env

.PHONY: setup test ingest-aact ingest-faers trials faers match check-faers attributes text upload warehouse train score \
	dashboard audit-build audit backtest

setup:
	uv sync

test:
	uv run pytest -q

# Usage: make ingest-aact AACT=<url-or-path-to-aact-zip> [DEST=<folder>]
ingest-aact:
	uv run python -m ctrisk.ingest.aact $(AACT) $(DEST)

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

text:
	uv run python -m ctrisk.spark.trial_text

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

# Serving views in Snowflake + docs/dashboard/index.html (after `make score`)
dashboard:
	uv run python -m ctrisk.serving.dashboard

# M6 point-in-time audit. AACT_ARCHIVES (in .env or on the command line): space-separated monthly
# AACT flat-file archives, URLs or local zips, named YYYYMMDD_*. Needs `make trials` first.
audit-build:
	@test -n "$(strip $(AACT_ARCHIVES))" || (echo "set AACT_ARCHIVES to the archive zips or URLs" && exit 1)
	uv run python -m ctrisk.spark.point_in_time $(AACT_ARCHIVES)

audit:
	uv run python -m ctrisk.ml.audit

backtest:
	uv run python -m ctrisk.ml.backtest
