from pyspark.sql import functions as F

def calculate_cps(clicks_df, purchases_df):
    """
    Tính điểm Click-Purchase Similarity (CPS) giữa các item.
    Phương pháp: Đếm tần suất đồng xuất hiện (co-occurrence frequency)
    giữa một item được click và một item được mua trong cùng một Session.
    
    Args:
        clicks_df: DataFrame của Clickstream data
        purchases_df: DataFrame của Purchase data
        
    Returns:
        DataFrame với schema (Click_Item, Purchase_Item, CPS_Score)
    """
    # Xóa trùng lặp item trong cùng 1 session để tránh đếm vống
    clicks_session = clicks_df.select("Session ID", "Item ID").distinct() \
                              .withColumnRenamed("Item ID", "Click_Item")
                              
    purchases_session = purchases_df.select("Session ID", "Item ID").distinct() \
                                    .withColumnRenamed("Item ID", "Purchase_Item")
                                    
    # Join để tìm cặp click - purchase trong cùng session
    co_occurrences = clicks_session.join(purchases_session, on="Session ID", how="inner")
    
    # Tính số lần đồng xuất hiện
    cps_scores = co_occurrences.groupBy("Click_Item", "Purchase_Item") \
                               .count() \
                               .withColumnRenamed("count", "CPS_Score")
                               
    return cps_scores
