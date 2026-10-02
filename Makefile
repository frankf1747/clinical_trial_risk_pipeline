-include .env

.PHONY: lookup publish upload-raw deploy setup test ingest-aact ingest-faers trials faers match check-faers attributes text upload warehouse train score \
	dashboard audit-build audit backtest report m6

setup:
	uv sync

test:
	uv run pytest -q

# Usage: make ingest-aact AACT=<url-or-path-to-aact-zip> [DEST=<folder>]
ingest-aact:
	uv run python -m ctrisk.ingest.aact $(AACT) $(DEST)

ingest-faers:
	uv run python -m ctrisk.ingest.faers

# Spark jobs run locally by default; `make <job> MODE=cloud` submits the same module to Dataproc Serverless
# (data in gs://$GCP_BUCKET, see scripts/gcp_setup.sh). Add DRY_RUN=1 to print the command and cost only.
SPARK = $(if $(filter cloud,$(MODE)),uv run python -m ctrisk.cloud.dataproc $(if $(DRY_RUN),--dry-run),uv run python -m)

trials:
	$(SPARK) ctrisk.spark.clean_trials

faers:
	$(SPARK) ctrisk.spark.flatten_faers

match:
	$(SPARK) ctrisk.spark.match_drugs

check-faers:
	uv run python -m ctrisk.checks.faers_api

attributes:
	$(SPARK) ctrisk.spark.trial_attributes

text:
	$(SPARK) ctrisk.spark.trial_text

# Mirror the Parquet Snowflake loads to GCS (~75 MB). Deleting stale objects matters:
# Spark part-file names change every run, and leftovers would load twice.
upload:
	@test -n "$(strip $(GCP_BUCKET))" || (echo "GCP_BUCKET is not set in .env" && exit 1)
	gcloud storage rsync --recursive --delete-unmatched-destination-objects \
	  --exclude='.*\.crc$$|.*_SUCCESS$$|^drug_interventions/.*' \
	  data/parquet gs://$(strip $(GCP_BUCKET))/parquet

# Raw inputs for cloud-mode Spark (once, and after a new AACT snapshot): AACT tables and FAERS zips (~18 GB)
upload-raw:
	@test -n "$(strip $(GCP_BUCKET))" || (echo "GCP_BUCKET is not set in .env" && exit 1)
	gcloud storage rsync --recursive data/raw/aact gs://$(strip $(GCP_BUCKET))/raw/aact
	gcloud storage rsync --recursive data/raw/faers gs://$(strip $(GCP_BUCKET))/raw/faers

warehouse:
	uv run python -m ctrisk.warehouse.snowflake

train:
	uv run python -m ctrisk.ml.train

score:
	uv run python -m ctrisk.ml.score

# Every trial in the modelled population scored without seeing its own outcome -> TRIAL_LOOKUP_SCORES
lookup:
	uv run python -m ctrisk.ml.lookup

# TRIAL_LOOKUP unloaded by Snowflake to gs://$GCP_BUCKET/serving/, then current.json points the app at it
publish:
	uv run python -m ctrisk.serving.publish

# The public lookup on Cloud Run: scales to zero, reads gs://$GCP_BUCKET/serving/ at start (after `make publish`)
deploy:
	gcloud run deploy ctrisk-lookup --source lookup_app --region $(or $(strip $(GCP_REGION)),us-central1) \
	  --allow-unauthenticated --set-env-vars LOOKUP_SOURCE=gs://$(strip $(GCP_BUCKET))/serving \
	  --memory 1Gi --cpu 1 --min-instances 0 --max-instances 2

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

# The whole M6 run, in order, from the AACT snapshot and FAERS Parquet already under data/ (M1-M2).
# Rebuilds trials and their drug map (the population now keeps finished post-2020 trials), reloads
# Snowflake, trains, scores, and writes the model card and dashboard. The point-in-time audit runs only
# when AACT_ARCHIVES is set. Stops at the first failure; run without -j so steps stay in order.
m6: trials match attributes text upload warehouse train score report dashboard
	@if [ -n "$(strip $(AACT_ARCHIVES))" ]; then $(MAKE) audit-build audit report; \
	else echo "AACT_ARCHIVES is not set in .env: skipped the point-in-time audit"; fi
	@echo "Next: commit models/ (new version), docs/model_card.md and docs/dashboard/index.html"

# docs/model_card.md from the latest models/vN (after train, and again after audit or backtest)
report:
	uv run python -m ctrisk.ml.report
