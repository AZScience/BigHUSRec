from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.sql import functions as F

def load_clickstream(spark, file_path):
    schema = StructType([
        StructField("Session ID", IntegerType(), True),
        StructField("Timestamp", StringType(), True),
        StructField("Item ID", IntegerType(), True),
        StructField("Category", StringType(), True)
    ])
    df = spark.read.csv(file_path, header=False, schema=schema)
    return df.dropna(subset=["Session ID", "Item ID"])

def load_purchases(spark, file_path):
    schema = StructType([
        StructField("Session ID", IntegerType(), True),
        StructField("Timestamp", StringType(), True),
        StructField("Item ID", IntegerType(), True),
        StructField("Price", FloatType(), True),
        StructField("Quantity", IntegerType(), True)
    ])
    df = spark.read.csv(file_path, header=False, schema=schema)
    
    # Tính Profit (Utility) = Price * Quantity
    df = df.withColumn("Utility", F.col("Price") * F.col("Quantity"))
    return df.dropna(subset=["Session ID", "Item ID", "Utility"])
