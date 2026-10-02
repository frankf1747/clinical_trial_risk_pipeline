-- One-time Snowflake setup for M7: let the storage integration write the lookup to gs://BUCKET/serving/.
-- Run in Snowsight, top to bottom. Altering an integration needs ACCOUNTADMIN.
USE ROLE ACCOUNTADMIN;
ALTER STORAGE INTEGRATION CTRISK_GCS SET STORAGE_ALLOWED_LOCATIONS = (
    'gcs://clinical-trial-risk-frankfu/parquet/', 'gcs://clinical-trial-risk-frankfu/serving/');
DESC STORAGE INTEGRATION CTRISK_GCS;    -- copy STORAGE_GCP_SERVICE_ACCOUNT for scripts/gcp_setup.sh

USE ROLE SYSADMIN;
CREATE STAGE IF NOT EXISTS CTRISK.PIPELINE.SERVING_STAGE
    URL = 'gcs://clinical-trial-risk-frankfu/serving/'
    STORAGE_INTEGRATION = CTRISK_GCS;
