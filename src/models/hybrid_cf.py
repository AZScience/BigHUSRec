from pyspark.sql import functions as F
from pyspark.sql.window import Window

class HybridCFRecommender:
    def __init__(self, whuom_df):
        """
        Khởi tạo mô hình bằng Ma trận WHUOM
        whuom_df schema: (Click_Item, Purchase_Item, WHUOM_Weight)
        """
        self.whuom_df = whuom_df

    def recommend(self, user_clicks_df, top_k=5):
        """
        Gợi ý Top-K sản phẩm cho mỗi Session/User dựa trên các click gần đây.
        
        Args:
            user_clicks_df: DataFrame schema (Session ID, Click_Item)
            top_k: Số lượng sản phẩm muốn gợi ý
            
        Returns:
            DataFrame chứa (Session ID, Recommendations)
        """
        # Join các click của user với ma trận WHUOM để lấy ra các item tiềm năng
        recommendations = user_clicks_df.join(
            self.whuom_df,
            on="Click_Item",
            how="inner"
        )
        
        # Tổng hợp trọng số (Weight) cho từng Purchase_Item theo Session
        # Nếu user click nhiều item cùng dẫn tới 1 Purchase_Item thì trọng số sẽ được cộng dồn
        agg_recommendations = recommendations.groupBy("Session ID", "Purchase_Item") \
            .agg(F.sum("WHUOM_Weight").alias("Total_Score"))
            
        # Lọc bỏ các item mà user đã mua rồi (nếu cần, nhưng ở đây ưu tiên gợi ý dựa trên click)
        # Sắp xếp và lấy Top-K cho mỗi Session
        window_spec = Window.partitionBy("Session ID").orderBy(F.desc("Total_Score"))
        
        ranked_recs = agg_recommendations.withColumn("rank", F.row_number().over(window_spec)) \
            .filter(F.col("rank") <= top_k)
            
        # Gom nhóm kết quả thành dạng danh sách (Array) cho mỗi Session
        final_recs = ranked_recs.groupBy("Session ID") \
            .agg(F.collect_list("Purchase_Item").alias("Top_K_Recommendations"),
                 F.collect_list("Total_Score").alias("Scores"))
                 
        return final_recs
