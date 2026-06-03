import os
import sys

# Thêm đường dẫn dự án vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.pipeline.data_loader import load_clickstream, load_purchases
from src.pipeline.mapreduce_jobs import prepare_click_sequences, run_huspm_pipeline
from src.matrix.cps_calculator import calculate_cps
from src.matrix.whuom import build_whuom
from src.models.hybrid_cf import HybridCFRecommender

def main():
    print("--- Khởi tạo PySpark cho BigHUSRec ---")
    spark = SparkSession.builder \
        .appName("BigHUSRec_Pipeline") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
        
    # Thiết lập mức log cảnh báo để tránh nhiễu
    spark.sparkContext.setLogLevel("ERROR")

    # 1. Load Data
    print("1. Nạp dữ liệu (Clickstream & Purchases)...")
    click_file = "yoochoose-clicks-mock.csv"
    purchase_file = "yoochoose-purchases-mock.csv"
    
    if not os.path.exists(click_file) or not os.path.exists(purchase_file):
        print(f"Lỗi: Không tìm thấy file dữ liệu giả lập. Vui lòng chạy 'python generate_mock_data.py' trước.")
        return

    clicks_df = load_clickstream(spark, click_file)
    purchases_df = load_purchases(spark, purchase_file)
    
    print(f"Số dòng Clickstream: {clicks_df.count()}")
    print(f"Số dòng Purchases: {purchases_df.count()}")

    # 2. Chạy PrefixSpan (Optional: Khai phá mẫu nhấp chuột)
    # Lược bỏ bước này trong pipeline chính để tối ưu, chỉ tập trung vào HUS-SPAN
    
    # 3. Chạy HUS-SPAN (MapReduce Pipeline)
    print("2. Phân tán dữ liệu & Khai phá Chuỗi Tiện ích Cao (HUS-SPAN)...")
    min_utility = 1000.0  # Ngưỡng lợi nhuận tối thiểu
    hus_patterns_df = run_huspm_pipeline(purchases_df, min_utility)
    
    if hus_patterns_df is None or hus_patterns_df.count() == 0:
        print("Không tìm thấy chuỗi tiện ích cao nào với min_utility =", min_utility)
        print("Thử giảm min_utility xuống!")
        return
        
    print(f"Tìm thấy {hus_patterns_df.count()} Items/Chuỗi Tiện ích Cao.")
    hus_patterns_df.show(5)

    # 4. Tính toán CPS
    print("3. Tính toán độ tương đồng Click-Purchase (CPS)...")
    cps_scores_df = calculate_cps(clicks_df, purchases_df)
    print(f"Tổng số cặp CPS được tính: {cps_scores_df.count()}")
    
    # 5. Xây dựng WHUOM
    print("4. Xây dựng Ma trận WHUOM...")
    whuom_df = build_whuom(cps_scores_df, hus_patterns_df)
    print(f"Số lượng trọng số trong WHUOM: {whuom_df.count()}")
    whuom_df.show(5)
    
    # 6. Recommendation
    print("5. Khởi chạy Mô hình Lọc cộng tác lai (Hybrid CF) & Gợi ý...")
    recommender = HybridCFRecommender(whuom_df)
    
    # Lấy ra các user (session) có click để test
    test_users_df = clicks_df.select("Session ID", "Item ID").withColumnRenamed("Item ID", "Click_Item").distinct()
    
    recommendations_df = recommender.recommend(test_users_df, top_k=3)
    
    print("==== KẾT QUẢ GỢI Ý TOP-K (BigHUSRec) ====")
    recommendations_df.show(10, truncate=False)
    
    print("Pipeline hoàn tất thành công!")
    spark.stop()

if __name__ == "__main__":
    main()
