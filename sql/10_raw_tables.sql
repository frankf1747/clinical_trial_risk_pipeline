-- Typed landing tables. Snowflake fills them with COPY (11_copy.sql); tests fill them with INSERTs.

CREATE OR REPLACE TABLE RAW_TRIALS (
    nct_id STRING, status STRING, phase STRING, start_date DATE, number_of_arms INT, label INT,
    allocation STRING, intervention_model STRING, primary_purpose STRING, masking STRING,
    sponsor_name STRING, sponsor_class STRING
);

CREATE OR REPLACE TABLE RAW_TRIAL_ATTRIBUTES (
    nct_id STRING, n_countries INT, us_only BOOLEAN, min_age_years FLOAT, max_age_years FLOAT,
    healthy_volunteers BOOLEAN, sex STRING, criteria_count INT, criteria_chars INT,
    area_neoplasms BOOLEAN, area_cardiovascular BOOLEAN, area_nervous_system BOOLEAN,
    area_mental BOOLEAN, area_infections BOOLEAN, area_respiratory BOOLEAN, area_digestive BOOLEAN,
    area_metabolic BOOLEAN, area_immune BOOLEAN, area_skin BOOLEAN, area_musculoskeletal BOOLEAN,
    area_urogenital BOOLEAN, area_blood BOOLEAN, area_endocrine BOOLEAN
);

CREATE OR REPLACE TABLE RAW_SPONSOR_OUTCOMES (
    nct_id STRING, terminated INT, completion_date DATE, sponsor_name STRING
);

CREATE OR REPLACE TABLE RAW_TRIAL_DRUG_MAP (
    nct_id STRING, substance STRING
);

CREATE OR REPLACE TABLE RAW_FAERS_DRUG_EVENTS (
    safetyreportid STRING, substance STRING, receivedate DATE, receiptdate DATE,
    serious BOOLEAN, death BOOLEAN, suspect BOOLEAN, harmonized BOOLEAN, active_substance BOOLEAN
);
