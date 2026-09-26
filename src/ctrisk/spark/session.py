from pyspark.sql import SparkSession


def get_spark(app: str, mode: str = "local") -> SparkSession:
    builder = SparkSession.builder.appName(app).config("spark.sql.session.timeZone", "UTC")
    if mode == "local":
        builder = builder.master("local[*]")  # on Dataproc the cluster sets the master
    return builder.getOrCreate()
