# Run with: spark-submit genai_analytics/analytics/spark_job.py data.csv
from pyspark.sql import SparkSession, functions as F
import sys

if __name__ == '__main__':
    spark = SparkSession.builder.appName('GenAIAnalyticsSpark').getOrCreate()
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        df = spark.read.csv(path, header=True, inferSchema=True)
    else:
        df = spark.createDataFrame([(1, 10.0), (2, 12.5), (3, 11.0)], ['id','value'])
    out = df.agg(F.mean('value').alias('mean_value'))
    out.show()
    spark.stop()
