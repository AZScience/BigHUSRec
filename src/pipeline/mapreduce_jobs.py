from src.algorithms.prefixspan_spark import run_prefixspan
from src.algorithms.hus_span import run_hus_span_partition
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

def prepare_click_sequences(clicks_df):
    """
    Chuyển Clickstream thành dạng Sequence format cho PrefixSpan
    (Array of Arrays). Group by Session ID.
    """
    # Gom nhóm theo Session ID, tạo array các Item ID
    # PrefixSpan của Spark ML yêu cầu format sequence: Array(Array(item))
    seq_df = clicks_df.groupBy("Session ID").agg(F.collect_list("Item ID").alias("items"))
    
    # UDF chuyển [1, 2, 3] thành [[1], [2], [3]]
    transform_udf = F.udf(lambda items: [[item] for item in items], 'array<array<int>>')
    
    seq_df = seq_df.withColumn("sequence", transform_udf("items"))
    return seq_df

def run_huspm_pipeline(purchases_df, min_utility):
    """
    Chạy quá trình MapReduce cho High Utility Sequential Pattern Mining
    Sử dụng HUS-SPAN trên các partitions.
    """
    # Gom nhóm purchases theo Session ID
    grouped_rdd = purchases_df.select("Session ID", "Item ID", "Utility") \
        .rdd.map(lambda row: (row["Session ID"], [(row["Item ID"], row["Utility"])])) \
        .reduceByKey(lambda a, b: a + b)
        
    # Áp dụng HUS-SPAN logic trên từng Partition (MapPhase)
    # Gom kết quả lại (ReducePhase)
    hus_patterns_rdd = grouped_rdd.mapPartitions(lambda it: run_hus_span_partition(it, min_utility))
    
    # Lấy các pattern có utility cao nhất
    # format của HUS-SPAN output: ( [item], utility )
    if hus_patterns_rdd.isEmpty():
        return None
        
    # Group by sequence, sum utilities (Reduce Phase)
    # Lược bỏ list wrapper cho item vì ta tính item đơn cho WHUOM
    reduced_patterns = hus_patterns_rdd.map(lambda x: (x[0][0], x[1])) \
                                       .reduceByKey(lambda a, b: max(a, b)) # Lấy max utility hoặc tổng
                                       
    spark = purchases_df.sparkSession
    schema = StructType([
        StructField("Item", IntegerType(), True),
        StructField("Utility", FloatType(), True)
    ])
    
    if reduced_patterns.isEmpty():
        return spark.createDataFrame([], schema)
        
    return spark.createDataFrame(reduced_patterns, schema)
