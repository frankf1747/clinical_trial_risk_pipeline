from ctrisk.cloud.dataproc import batch_command, estimate_usd


def test_batch_command_runs_the_module_on_serverless_with_caps_and_cloud_settings():
    cmd = batch_command("ctrisk.spark.clean_trials", bucket="b", region="us-central1",
                        code=["gs://b/code/ctrisk.whl", "gs://b/code/deps.zip"], entry="gs://b/code/run_module.py",
                        env={"MODE": "cloud", "GCP_BUCKET": "b"}, batch_id="ctrisk-trials-1", max_executors=4)
    assert cmd[:6] == ["gcloud", "dataproc", "batches", "submit", "pyspark", "gs://b/code/run_module.py"]
    joined = " ".join(cmd)
    assert "--region=us-central1" in joined and "--batch=ctrisk-trials-1" in joined and "--ttl=" in joined
    assert "--py-files=gs://b/code/ctrisk.whl,gs://b/code/deps.zip" in joined
    props = next(a for a in cmd if a.startswith("--properties="))
    assert "spark.dynamicAllocation.maxExecutors=4" in props
    assert "spark.dataproc.driverEnv.MODE=cloud" in props and "spark.executorEnv.GCP_BUCKET=b" in props
    assert cmd[-2:] == ["--", "ctrisk.spark.clean_trials"]


def test_estimate_counts_driver_and_executors_at_four_dcus_each():
    assert estimate_usd(max_executors=4, minutes=30) == round(4 * (1 + 4) * 0.5 * 0.06, 2)
