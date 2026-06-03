from pyspark.ml.fpm import PrefixSpan

def run_prefixspan(spark_df, min_support=0.1, max_pattern_length=5, sequence_col="sequence"):
    """
    Chạy thuật toán PrefixSpan bằng PySpark MLlib để tìm các mẫu chuỗi nhấp chuột.
    
    Args:
        spark_df: DataFrame chứa cột sequence (Array of Arrays của Items)
        min_support: Ngưỡng support tối thiểu (0.0 đến 1.0)
        max_pattern_length: Độ dài chuỗi tối đa
        sequence_col: Tên cột chứa chuỗi
        
    Returns:
        DataFrame chứa các mẫu (sequence) và tần suất (freq)
    """
    prefixSpan = PrefixSpan(
        minSupport=min_support, 
        maxPatternLength=max_pattern_length,
        maxLocalProjDBSize=32000000,
        sequenceCol=sequence_col
    )
    
    patterns = prefixSpan.findFrequentSequentialPatterns(spark_df)
    return patterns
