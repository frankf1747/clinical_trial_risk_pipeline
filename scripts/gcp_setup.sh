#!/usr/bin/env bash
# One-time GCP setup for M7, run by the project owner (it changes APIs, network and IAM).
#   scripts/gcp_setup.sh <snowflake-service-account-email>
# The email is STORAGE_GCP_SERVICE_ACCOUNT from `DESC STORAGE INTEGRATION CTRISK_GCS` in Snowflake.
set -euo pipefail
SNOWFLAKE_SA="${1:?usage: scripts/gcp_setup.sh <snowflake-service-account-email>}"
set -a; source "$(dirname "$0")/../.env"; set +a
PROJECT="$(gcloud config get-value project)"
REGION="${GCP_REGION:-us-central1}"
NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
COMPUTE_SA="${NUMBER}-compute@developer.gserviceaccount.com"     # runs Dataproc batches, builds and serves Cloud Run

echo "== APIs: Dataproc, Cloud Run, Cloud Build, Artifact Registry"
gcloud services enable dataproc.googleapis.com run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

echo "== Dataproc Serverless needs Private Google Access on the subnet it runs in"
gcloud compute networks subnets update default --region="$REGION" --enable-private-ip-google-access

echo "== Compute service account: run Dataproc batches, build the app image, read and write the bucket"
gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$COMPUTE_SA" --role=roles/dataproc.worker --condition=None >/dev/null
gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$COMPUTE_SA" --role=roles/cloudbuild.builds.builder --condition=None >/dev/null
gcloud storage buckets add-iam-policy-binding "gs://$GCP_BUCKET" --member="serviceAccount:$COMPUTE_SA" --role=roles/storage.objectAdmin >/dev/null

echo "== Snowflake: may write (and overwrite) only under serving/"
gcloud storage buckets update "gs://$GCP_BUCKET" --uniform-bucket-level-access
gcloud storage buckets add-iam-policy-binding "gs://$GCP_BUCKET" --member="serviceAccount:$SNOWFLAKE_SA" \
  --role=roles/storage.objectAdmin \
  --condition="expression=resource.name.startsWith('projects/_/buckets/$GCP_BUCKET/objects/serving/'),title=snowflake-serving-only" >/dev/null

echo "done: $PROJECT, $REGION"
