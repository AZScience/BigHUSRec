from pyspark.sql import functions as F

def build_whuom(cps_scores_df, item_utility_df):
    """
    Xây dựng Ma trận WHUOM (Weighted High Utility Occupancy Matrix).
    Kết hợp điểm tương đồng CPS với giá trị tiện ích (Utility) của item.
    
    Args:
        cps_scores_df: DataFrame từ calculate_cps (Click_Item, Purchase_Item, CPS_Score)
        item_utility_df: DataFrame chứa Utility của các item mua (Item, Utility)
        
    Returns:
        DataFrame đại diện cho WHUOM (Click_Item, Purchase_Item, WHUOM_Weight)
    """
    # Join bảng CPS với bảng Utility
    whuom_df = cps_scores_df.join(
        item_utility_df, 
        cps_scores_df["Purchase_Item"] == item_utility_df["Item"],
        how="left"
    ).fillna({"Utility": 0.0})
    
    # Tính trọng số WHUOM: Weight = CPS * Utility
    # Có thể điều chỉnh công thức này (VD: log(CPS) * Utility) tùy vào phân phối dữ liệu
    whuom_df = whuom_df.withColumn("WHUOM_Weight", F.col("CPS_Score") * F.col("Utility"))
    
    # Chỉ giữ lại các liên kết có trọng số > 0
    whuom_df = whuom_df.filter(F.col("WHUOM_Weight") > 0) \
                       .select("Click_Item", "Purchase_Item", "WHUOM_Weight")
                       
    return whuom_df
