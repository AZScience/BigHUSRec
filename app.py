import streamlit as st
import pandas as pd
import os
import sys

# Thiết lập JAVA_HOME bắt buộc cho Streamlit Cloud (Môi trường Linux/Debian)
if sys.platform.startswith('linux'):
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/default-java"

# Thêm đường dẫn dự án vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.pipeline.data_loader import load_clickstream, load_purchases
from src.pipeline.mapreduce_jobs import run_huspm_pipeline
from src.matrix.cps_calculator import calculate_cps
from src.matrix.whuom import build_whuom
from src.models.hybrid_cf import HybridCFRecommender

st.set_page_config(page_title="BigHUSRec Pipeline", page_icon="🛍️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #0066cc; color: white; border-radius: 5px; padding: 10px 24px; font-weight: bold; }
    .stButton>button:hover { background-color: #0052a3; }
    .step-box { background-color: #e9ecef; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid #0066cc; }
    .academic-box { background-color: #fdfdfe; padding: 10px; border-left: 3px solid #ff9900; margin-bottom: 15px; font-size: 0.9em; }
    </style>
""", unsafe_allow_html=True)

st.title("🛍️ Hệ thống Gợi ý BigHUSRec (Bản Đầy đủ Minh chứng & Lý thuyết)")

# ================= SIDEBAR (NHÃN & GIẢI THÍCH KHOA HỌC) =================
st.sidebar.header("⚙️ Nhãn Cấu hình Tham số")

st.sidebar.markdown("""
<div style="background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
<b>Nhãn: Ngưỡng Lợi nhuận (Min Utility)</b><br>
- <i>Khoa học:</i> $\\xi$. Hệ số dùng để đối sánh với hàm Sequence-Weighted Utility $SWU(s)$.<br>
- <i>Thực tiễn:</i> Xác định mức sinh lời tối thiểu mà doanh nghiệp muốn nhắm tới. Nếu đặt quá thấp, hệ thống bị OOM.
</div>
""", unsafe_allow_html=True)
min_utility = st.sidebar.slider("Điều chỉnh Min Utility", min_value=100.0, max_value=5000.0, value=1000.0, step=100.0)

st.sidebar.markdown("""
<div style="background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
<b>Nhãn: Số lượng gợi ý (Top-K)</b><br>
- <i>Khoa học:</i> Giới hạn không gian bài toán $P$ sao cho kích thước $|P|=K$.<br>
- <i>Thực tiễn:</i> Cắt giảm thời gian truy vấn, chỉ trả về số lượng sản phẩm vừa vặn với kích thước màn hình UI/UX của khách.
</div>
""", unsafe_allow_html=True)
top_k = st.sidebar.number_input("Điều chỉnh Top-K", min_value=1, max_value=20, value=5)

# Helper Render
def render_academic_block(title, theory, math, practical, situation, recommendation, evidence_md):
    with st.expander(f"📖 {title} (Chi tiết Học thuật & Minh chứng)", expanded=False):
        st.markdown(f"**1. Cơ sở lý thuyết:** {theory}")
        st.markdown(f"**2. Mô hình toán học:**\n{math}")
        st.markdown(f"**3. Ý nghĩa thực tế:** {practical}")
        st.markdown(f"**4. Hiện trạng (Current Situation):** {situation}")
        st.markdown(f"**5. Kiến nghị:** {recommendation}")
        st.markdown("**📊 Minh chứng Dữ liệu / Biểu đồ Kiến trúc:**")
        st.markdown(evidence_md)

tab_intro, tab1, tab2, tab3 = st.tabs(["0. Giới thiệu Bài toán", "1. Chuẩn bị Dữ liệu (ETL)", "2. MapReduce & Thuật toán", "3. Phục vụ Gợi ý (Serving)"])

# ================= TAB INTRO: PROBLEM MODEL =================
with tab_intro:
    st.header("Kiến trúc Tổng thể & Mô hình Bài toán (BigHUSRec)")
    st.markdown("Hệ thống Gợi ý Chuỗi Tiện ích Cao (Big High Utility Sequential Pattern Recommendation) hoạt động theo luồng kiến trúc dưới đây:")
    
    import streamlit.components.v1 as components
    mermaid_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: sans-serif; }
            .mermaid { display: flex; justify-content: center; }
        </style>
    </head>
    <body>
      <div class="mermaid">
        graph TD;
            subgraph Phase1 [Pha 1: Chuẩn bị Dữ liệu ETL]
                A[Raw Clickstream] --> C[Data Loader and Cleaning];
                B[Raw Purchases] --> C;
                C --> D[Tính toán Utility];
            end
            
            subgraph Phase2 [Pha 2: MapReduce và Khai phá Mẫu]
                D --> E{HUS-SPAN MapReduce};
                E -->|Cắt tỉa LQS-Tree| F[High Utility Patterns];
                D --> G[Tính điểm đồng xuất hiện];
                G --> H[CPS Scores Matrix];
                F --> I[Nhân chéo Trọng số Lợi nhuận];
                H --> I;
                I --> J[(Ma trận WHUOM)];
            end
            
            subgraph Phase3 [Pha 3: Phục vụ Khách hàng Real-time]
                K[Lịch sử Click trong Phiên] --> L{Lọc cộng tác Hybrid CF};
                J --> L;
                L -->|Hàm Argmax| M[Danh sách Top-K Gợi ý Sinh lời nhất];
            end
            
            style A fill:#f9f,stroke:#333
            style B fill:#f9f,stroke:#333
            style J fill:#ff9,stroke:#333,stroke-width:2px
            style M fill:#bbf,stroke:#333,stroke-width:4px
      </div>
      <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({ startOnLoad: true, theme: 'default' });
      </script>
    </body>
    </html>
    """
    components.html(mermaid_html, height=700, scrolling=True)
    
    render_academic_block(
        title="Lý thuyết: Phát biểu Mô hình Bài toán Tối ưu Doanh thu",
        theory="Trong E-commerce, việc gợi ý dựa theo số lượt click (Popularity) thường dẫn đến việc hệ thống chỉ ưu tiên các mặt hàng giá rẻ, nhiều người xem (Ví dụ: Ốp lưng điện thoại). Mô hình BigHUSRec đặt mục tiêu chuyển dịch từ 'Tối đa hóa tỷ lệ nhấp' sang 'Tối đa hóa Lợi nhuận kinh tế'.",
        math="""
Mục tiêu của hệ thống là học một hàm bản đồ $f(S_{active}) \\rightarrow P_{Top-K}$ sao cho Kỳ vọng Lợi nhuận (Expected Profit) là lớn nhất:
$$E[Profit] = \\sum_{p \\in P_{Top-K}} P(Buy | Click) \\times Utility(p)$$
        """,
        practical="Mang lại doanh thu thực tế cao hơn cho doanh nghiệp thay vì các chỉ số 'ảo' (Pageviews). Đồng thời cung cấp trải nghiệm mua sắm cá nhân hóa cho người dùng.",
        situation="Khai phá tập mục tiện ích cao (HUI) rất tốn kém tài nguyên (NP-Hard). Xử lý hàng chục triệu lượt clicks trên Server đơn là bất khả thi.",
        recommendation="Thiết kế kiến trúc hệ thống 3 pha (như sơ đồ trên) để xử lý Offline nặng (PySpark) và phản hồi Online siêu tốc qua Ma trận thưa.",
        evidence_md="**Minh chứng:** Lưu đồ (Flowchart) kiến trúc hệ thống bên trên chính là mô hình tiêu chuẩn đang được áp dụng trong BigHUSRec."
    )

# ================= TAB 1: DATA PREPARATION =================
with tab1:
    st.header("Bước 1: Chuẩn bị & Tiền xử lý Dữ liệu Lớn (ETL)")
    
    st.markdown('<div class="step-box"><b>Khâu 1.1: Giới thiệu Tập dữ liệu YOOCHOOSE (RecSys Challenge 2015)</b></div>', unsafe_allow_html=True)
    st.markdown("""
    **Bối cảnh (Context):** YOOCHOOSE là tập dữ liệu thương mại điện tử công khai được phát hành trong cuộc thi ACM RecSys Challenge 2015. Nó chứa lịch sử nhấp chuột (click) và mua hàng (purchase) của người dùng trong khoảng 6 tháng.
    
    **Đặc điểm Dữ liệu:**
    - **Khối lượng:** Hơn 33 triệu lượt Click và 1.1 triệu lượt Mua hàng.
    - **Vấn đề cốt lõi:** Tỷ lệ mua hàng cực kỳ thấp (Sparsity). Dữ liệu thô không có tiêu đề (Header) và hoàn toàn chưa được chuẩn hóa lợi nhuận.
    
    **Định nghĩa Cột (Schema):**
    - `yoochoose-clicks.dat`: Session ID, Timestamp, Item ID, Category.
    - `yoochoose-buys.dat`: Session ID, Timestamp, Item ID, Price, Quantity.
    """)
    st.markdown(
        """<a href="https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z" target="_blank" style="text-decoration:none;"><button style="background-color: #28a745; color: white; border-radius: 5px; padding: 10px 24px; font-weight: bold; border: none; cursor: pointer;">📥 Tải xuống YOOCHOOSE Dataset Thực tế (1.4 GB)</button></a><br><br>""",
        unsafe_allow_html=True
    )
    
    click_path = st.text_input("Đường dẫn Clickstream (.dat):", value="yoochoose-clicks.dat")
    purchase_path = st.text_input("Đường dẫn Purchase (.dat):", value="yoochoose-buys.dat")
    
    if st.button("🚀 Chạy Pipeline Tiền Xử Lý"):
        st.session_state["etl_run"] = True

    if st.session_state.get("etl_run", False):
        st.markdown('<div class="step-box"><b>Khâu 1.2: Minh chứng Dữ liệu Thô (Raw Data Extraction)</b></div>', unsafe_allow_html=True)
        render_academic_block(
            title="Lý thuyết: Nạp dữ liệu phi cấu trúc",
            theory="Trích xuất (Extract) dữ liệu từ các file Log thô. Dữ liệu này phân mảnh thành 2 luồng độc lập: hành vi xem (Click) và hành vi thanh toán (Buy).",
            math="Không gian dữ liệu: $D_{raw} = \\{C, P\\}$",
            practical="Kiểm tra cấu trúc tệp ban đầu xem có bị hỏng hoặc sai định dạng hay không.",
            situation="File YOOCHOOSE phân cách bằng dấu phẩy nhưng không có dòng tiêu đề.",
            recommendation="Sử dụng Pandas/PySpark để định danh lại cột.",
            evidence_md="Xem bảng DataFrame thực tế ở bên dưới."
        )
        
        # Tạo mock data fallback nếu file không tồn tại
        raw_clicks_df = pd.DataFrame([
            [11, '2014-04-03T10:44:35.672Z', 214536502, 0],
            [11, '2014-04-03T10:45:01.423Z', 214536500, 0],
            [12, '2014-04-02T10:22:15.111Z', 214536502, 0]
        ], columns=['SessionID', 'Timestamp', 'ItemID', 'Category'])
        
        raw_buys_df = pd.DataFrame([
            [11, '2014-04-03T10:45:11.233Z', 214536502, 1500, 2],
            [12, '2014-04-02T10:25:10.000Z', 214536502, 1500, 1]
        ], columns=['SessionID', 'Timestamp', 'ItemID', 'Price', 'Quantity'])
        
        if os.path.exists(click_path) and os.path.exists(purchase_path):
            raw_clicks_df = pd.read_csv(click_path, header=None, names=['SessionID', 'Timestamp', 'ItemID', 'Category'], nrows=5)
            raw_buys_df = pd.read_csv(purchase_path, header=None, names=['SessionID', 'Timestamp', 'ItemID', 'Price', 'Quantity'], nrows=5)
            st.success("✅ Đã trích xuất dữ liệu thô từ file thực tế.")
        else:
            st.warning("⚠️ Không tìm thấy file dữ liệu thực. Hệ thống đang hiển thị Dữ liệu Mẫu (Mock) của YOOCHOOSE để minh chứng.")
    
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Bảng Minh chứng: Raw Clickstream**")
            st.dataframe(raw_clicks_df, use_container_width=True)
        with col2:
            st.markdown("**Bảng Minh chứng: Raw Purchases**")
            st.dataframe(raw_buys_df, use_container_width=True)
    
        st.markdown('<div class="step-box"><b>Khâu 1.3: Minh chứng Tiền xử lý Dữ liệu (Data Preprocessing)</b></div>', unsafe_allow_html=True)
        render_academic_block(
            title="Lý thuyết: Làm sạch (Data Cleaning) & Xử lý Ngoại lai",
            theory="Làm sạch dữ liệu nhằm loại bỏ các giá trị bị khuyết (Missing Values) hoặc lỗi định dạng. Một mô hình chuẩn yêu cầu các bản ghi phải đầy đủ Session ID và Item ID.",
            math="Tập hợp hợp lệ: $D_{clean} = \\{x \\in D_{raw} \\mid x_{session} \\neq \\emptyset \\wedge x_{item} \\neq \\emptyset \\}$",
            practical="Ngăn chặn các lỗi Null Pointer Exception trong MapReduce và sai lệch do rác.",
            situation="YOOCHOOSE đôi khi chứa các bản ghi bị ngắt quãng hoặc thiếu thông tin sản phẩm.",
            recommendation="Sử dụng hàm `.dropna()` và ép kiểu nghiêm ngặt (Casting).",
            evidence_md="Dữ liệu sau khi làm sạch sẽ mất đi các dòng lỗi."
        )
        
        # Preprocessing
        clean_buys_df = raw_buys_df.dropna(subset=['SessionID', 'ItemID']).copy()
        clean_buys_df['Price'] = clean_buys_df['Price'].astype(float)
        clean_buys_df['Quantity'] = clean_buys_df['Quantity'].astype(int)
        
        st.markdown("**Bảng Minh chứng: Purchases Sau khi Làm sạch (Ép kiểu chuẩn)**")
        st.dataframe(clean_buys_df, use_container_width=True)
    
        st.markdown('<div class="step-box"><b>Khâu 1.4: Minh chứng Chuẩn hóa & Tính Tiện ích (Normalization & Utility Calculation)</b></div>', unsafe_allow_html=True)
        render_academic_block(
            title="Lý thuyết: Biến đổi Chuẩn hóa Kinh tế (Utility Transformation)",
            theory="Trong HUSPM, các đại lượng như Giá (Price) và Số lượng (Quantity) không thể sử dụng độc lập. Chuẩn hóa ở đây là việc tổng hợp các trường này thành một thang đo lợi ích duy nhất gọi là Tiện ích (Utility).",
            math="Hàm chuẩn hóa: $Utility(x) = p_x \\times q_x$. Ví dụ: Mua 2 cái áo giá 1500 -> Utility = 3000.",
            practical="Chuyển hướng hệ thống từ việc chỉ gợi ý các món hàng rẻ tiền (nhưng mua nhiều) sang gợi ý đồ sinh lời cao cho doanh nghiệp.",
            situation="YOOCHOOSE gốc không cung cấp cột Utility.",
            recommendation="Tạo cột phái sinh (Derived Column) trước khi đẩy vào HUS-SPAN.",
            evidence_md="Xem cột mới `Utility` được thêm vào bảng dưới."
        )
        
        # Normalization
        clean_buys_df['Utility'] = clean_buys_df['Price'] * clean_buys_df['Quantity']
        st.markdown("**Bảng Minh chứng: Output Chuẩn hóa (Sẵn sàng cho MapReduce)**")
        st.dataframe(clean_buys_df, use_container_width=True)
        
        with st.expander("⚙️ Chi tiết Kỹ thuật: Các Hàm (Functions) & Bước thực thi ETL"):
            st.markdown("""
            **1. Hàm `load_clickstream(spark, file_path)`**
            - **Mục đích:** Nạp dữ liệu Log Click thô vào hệ thống phân tán PySpark.
            - **Bước thực hiện:** 
              1. Khởi tạo `spark.read.csv` không có header.
              2. Đổi tên cột thành `SessionID, Timestamp, ItemID, Category`.
              3. Ép kiểu (Casting) `SessionID` và `ItemID` sang Integer.
            
            **2. Hàm `load_purchases(spark, file_path)`**
            - **Mục đích:** Nạp dữ liệu Mua hàng, xử lý ngoại lai và tính Tiện ích (Utility).
            - **Bước thực hiện:**
              1. Nạp CSV, định danh `SessionID, Timestamp, ItemID, Price, Quantity`.
              2. Lọc bỏ các dòng lỗi (`dropna`).
              3. Thêm cột phái sinh: `.withColumn("Utility", col("Price") * col("Quantity"))`.
              4. Trả về DataFrame sạch (Cleaned DataFrame) sẵn sàng cho cụm tính toán.
            """)
        
        st.markdown('<div class="step-box"><b>Khâu 1.5: Khám phá Phân phối Dữ liệu (Exploratory Data Analysis - EDA)</b></div>', unsafe_allow_html=True)
        
        st.subheader("Biểu đồ 1: Phân phối Tần suất Click (Power-law Distribution)")
        render_academic_block(
            title="Lý thuyết Biểu đồ: Phân phối Tần suất (Đuôi dài - Long-tail)",
            theory="Trong thương mại điện tử, tần suất click/mua hàng thường tuân theo phân phối Power-law (Định luật Pareto 80/20). Rất ít mặt hàng 'Hot' chiếm đại đa số lượt tương tác.",
            math="Hàm mật độ: $f(k) = c \\cdot k^{-\\alpha}$, với $k$ là thứ hạng mặt hàng.",
            practical="Biểu đồ chứng minh dữ liệu mất cân bằng nghiêm trọng (Sparsity & Popularity Bias). Nếu chỉ dùng thuật toán đếm (Frequent Itemset), hệ thống sẽ bỏ qua các mặt hàng sinh lời cao nhưng ít người xem.",
            situation="Nhìn vào biểu đồ dưới, mặt hàng Top 1 chiếm áp đảo so với phần còn lại.",
            recommendation="Sử dụng mô hình BigHUSRec kết hợp Utility thay vì chỉ đếm Support để tránh thiên lệch.",
            evidence_md="**Minh chứng Biểu đồ Bar Chart:** Hiển thị 10 sản phẩm có lượt Click cao nhất."
        )
        
        if os.path.exists(click_path):
            clicks_sample = pd.read_csv(click_path, header=None, names=['SessionID', 'Timestamp', 'ItemID', 'Category'], nrows=100000)
            click_counts = clicks_sample['ItemID'].value_counts().head(10)
            click_counts.index = click_counts.index.astype(str)
        else:
            # Mock data tuân theo Power-law
            item_ids = [f"Item {i}" for i in range(1, 11)]
            counts = [10000, 4500, 2000, 1100, 600, 350, 200, 120, 80, 50]
            click_counts = pd.Series(counts, index=item_ids)
            st.info("💡 Đang hiển thị Dữ liệu Mẫu (Mock) cho biểu đồ do chưa tìm thấy file thực tế.")
        st.bar_chart(click_counts)
    
        st.subheader("Biểu đồ 2: Phân phối Mức giá (Price Distribution)")
        render_academic_block(
            title="Lý thuyết Biểu đồ: Phân phối & Xử lý Ngoại lai (Outliers)",
            theory="Giá của sản phẩm trong E-commerce thường phân bố lệch phải (Right-skewed) hoặc tuân theo Log-Normal.",
            math="Mật độ Log-Normal: $P(x) = \\frac{1}{x \\sigma \\sqrt{2\\pi}} e^{- \\frac{(\\ln x - \\mu)^2}{2\\sigma^2}}$.",
            practical="Cho phép xác định khoảng giá sinh lời chính yếu. Các giao dịch có giá cực đoan (Ví dụ > 95th Percentile) thường là lỗi ghi nhận Log.",
            situation="Trục hoành hiển thị các dải mức giá (Bins), trục tung là số lượng mặt hàng trong dải đó.",
            recommendation="Sử dụng phép cắt đuôi (Quantile cutoff 95%) trước khi tính toán để tránh làm nhiễu hệ số Utility.",
            evidence_md="**Minh chứng Biểu đồ Bar Chart:** Hiển thị mật độ phân phối giá sau khi loại bỏ 5% ngoại lai cao nhất."
        )
        
        if os.path.exists(purchase_path):
            purchases_sample = pd.read_csv(purchase_path, header=None, names=['SessionID', 'Timestamp', 'ItemID', 'Price', 'Quantity'], nrows=100000)
            q_price = purchases_sample[purchases_sample['Price'] < purchases_sample['Price'].quantile(0.95)]
            price_bins = pd.cut(q_price['Price'], bins=20).value_counts().sort_index()
            price_bins.index = price_bins.index.astype(str)
        else:
            # Mock Log-normal bins
            import numpy as np
            np.random.seed(42)
            mock_prices = np.random.lognormal(mean=7.0, sigma=0.5, size=5000)
            mock_prices = mock_prices[mock_prices < np.percentile(mock_prices, 95)]
            price_bins = pd.cut(mock_prices, bins=20).value_counts().sort_index()
            price_bins.index = [f"{int(i.left)}-{int(i.right)}" for i in price_bins.index]
            st.info("💡 Đang hiển thị Dữ liệu Mẫu (Mock) cho phân phối giá.")
        st.bar_chart(price_bins)
    
# ================= TAB 2: MAPREDUCE & ALGORITHMS =================
with tab2:
    st.header("Bước 2: Hệ thống MapReduce & Khai phá Mẫu Sinh lời (HUS-SPAN)")
    
    st.markdown('<div class="step-box"><b>Khâu 2.1: Tại sao phải sử dụng MapReduce & Mô hình Toán học?</b></div>', unsafe_allow_html=True)
    
    import streamlit.components.v1 as components
    mapreduce_mermaid = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>body { font-family: sans-serif; display: flex; justify-content: center; }</style>
    </head>
    <body>
      <div class="mermaid">
        graph TD
            subgraph Master_Node
                A["Dữ liệu YOOCHOOSE"]
                B("Bộ điều phối DAG")
            end

            subgraph Map_Phase
                C["Executor 1"]
                D["Executor 2"]
                E["Executor 3"]
                F{"Tính SWU và Cắt tỉa"}
                G{"Tính SWU và Cắt tỉa"}
                H{"Tính SWU và Cắt tỉa"}
                I["Mẫu cục bộ Local HUS"]
                J["Mẫu cục bộ Local HUS"]
                K["Mẫu cục bộ Local HUS"]
            end

            subgraph Reduce_Phase
                L(("Shuffle Layer"))
                M["Reducer Gom nhóm"]
                N["Cộng dồn Utility"]
                O[("Global HUS Patterns")]
            end

            A --> B
            B -->|Chia nho| C
            B -->|Chia nho| D
            B -->|Chia nho| E
            
            C --> F
            D --> G
            E --> H
            
            F -->|Giu lai nhanh| I
            G -->|Giu lai nhanh| J
            H -->|Giu lai nhanh| K
            
            I --> L
            J --> L
            K --> L
            
            L --> M
            M --> N
            N --> O

            style A fill:#f9f,stroke:#333,stroke-width:2px
            style L fill:#ff9,stroke:#333
            style O fill:#bbf,stroke:#333,stroke-width:4px
      </div>
      <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({ startOnLoad: true, theme: 'default' });
      </script>
    </body>
    </html>
    """
    components.html(mapreduce_mermaid, height=650, scrolling=True)

    render_academic_block(
        title="Giải đáp: Tại sao hệ thống này BẮT BUỘC phải dùng MapReduce?",
        theory="Bài toán Khai phá Chuỗi (Sequential Pattern Mining) thuộc nhóm NP-Hard. Việc tìm kiếm tất cả các tổ hợp mua hàng có thể xảy ra trên 33 triệu lượt Click sẽ gây ra hiện tượng 'Bùng nổ Tổ hợp' (Combinatorial Explosion). Một máy chủ đơn lẻ (Single Node) sẽ lập tức bị Tràn bộ nhớ RAM (Out-Of-Memory) ngay từ giây đầu tiên.",
        math="Độ phức tạp không gian (Space Complexity): $O(2^{|I|})$. MapReduce chia nhỏ bài toán này thành $N$ tập nhỏ chạy trên $N$ Executors.",
        practical="MapReduce (PySpark) đóng vai trò như một cỗ máy phân mảnh và xử lý siêu tốc. Như sơ đồ trên, dữ liệu được 'Chặt nhỏ' (Map) đưa cho hàng chục máy tính xử lý song song, sau đó 'Gom lại' (Reduce).",
        situation="YOOCHOOSE gốc có hàng triệu Transaction, thuật toán Apriori hay PrefixSpan truyền thống sẽ thất bại.",
        recommendation="Thiết lập Spark RDDs. Theo dõi `spark.executor.memory` để tinh chỉnh Cluster.",
        evidence_md="**Minh chứng:** Sơ đồ Mô phỏng Kiến trúc Hệ thống Phân tán (MapReduce Simulation Model) ở phía trên giải thích luồng hoạt động."
    )

    st.markdown('<div class="step-box"><b>Khâu 2.2: Mô hình Thuật toán HUS-SPAN (LQS-Tree)</b></div>', unsafe_allow_html=True)
    
    tree_mermaid = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>body { font-family: sans-serif; display: flex; justify-content: center; }</style>
    </head>
    <body>
      <div class="mermaid">
        graph TD
            Root(("Root"))
            S1(("Item A SWU 6000"))
            S2(("Item B SWU 2000"))
            S1A(("A tiến tới C SWU 5500"))
            S1B(("A tiến tới D SWU 4000"))
            DeadX["Không duyệt nhánh B"]

            Root --> S1
            Root --> S2
            
            S1 --> S1A
            S1 --> S1B
            
            S2 -->|Bi chat do SWU nho| DeadX
            
            style S2 fill:#ffcccc,stroke:#ff0000,stroke-width:2px
            style DeadX fill:#f9f9f9,stroke:#999,stroke-width:1px
            style S1A fill:#ccffcc,stroke:#00aa00
      </div>
      <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({ startOnLoad: true, theme: 'default' });
      </script>
    </body>
    </html>
    """
    components.html(tree_mermaid, height=350, scrolling=True)

    render_academic_block(
        title="Lý thuyết: Cắt tỉa nhánh trên Cây LQS-Tree",
        theory="Thuật toán xây dựng một Cây LQS (Lexicographic Quantitative Sequence Tree). Để chống bùng nổ tổ hợp, nó cắt bỏ (prune) các nhánh vô ích từ sớm thông qua độ đo SWU.",
        math="$SWU(S) = \\sum_{S \\subseteq S_q} u(S_q)$. Cắt bỏ hoàn toàn nếu $SWU(S) < \\xi$ (Min Utility).",
        practical="Giữ lại hành vi sinh lời, vứt bỏ hàng tỷ chuỗi rác gây lãng phí bộ nhớ.",
        situation="Sơ đồ LQS-Tree phía trên đang đặt $\\xi = 5000$. Nhánh Item B có SWU = 2000 nên bị cắt bỏ ngay lập tức (đứt đoạn màu đỏ).",
        recommendation="Chỉnh tham số $\\xi$ (Min Utility Threshold) hợp lý để đạt tốc độ cao nhất.",
        evidence_md="**Minh chứng:** Sơ đồ Cây LQS-Tree minh họa nguyên lý cắt tỉa nhánh."
    )
    
    with st.expander("⚙️ Chi tiết Kỹ thuật: Hàm HUS-SPAN & MapReduce"):
        st.markdown("""
        **1. Hàm `calculate_swu(sequence, item_utilities)`**
        - **Mục đích:** Tính cận trên SWU (Sequence-Weighted Utility) cho một chuỗi $S$.
        - **Bước thực hiện:** Quét qua toàn bộ chuỗi $S$, cộng dồn tiện ích tối đa của các hạng mục.
        
        **2. Hàm `run_huspm_pipeline(purchases_df, min_utility)`**
        - **Mục đích:** Thuật toán lõi chạy trên PySpark Cluster để tìm Mẫu sinh lời cao.
        - **Bước thực hiện:**
          1. **Map Phase:** Nhóm các `ItemID` và `Utility` theo từng `SessionID` sử dụng `groupBy`.
          2. Thu thập danh sách Tiện ích mặt hàng (Item Utilities Dict) và Broadcast tới các Worker Nodes.
          3. **Reduce/Prune Phase (LQS-Tree):** Sử dụng `rdd.mapPartitions`. Tại mỗi Node, hệ thống tính SWU. Nếu $SWU < Min Utility$, Node sẽ ngắt nhánh cây (Pruning), từ chối duyệt các mẫu mở rộng của nhánh đó.
          4. Trả về DataFrame chứa các Chuỗi (Patterns) vượt ngưỡng.
        """)

    st.markdown('<div class="step-box"><b>Khâu 2.2: Tính CPS & Xây dựng Ma trận WHUOM</b></div>', unsafe_allow_html=True)
    evidence_2_2 = """
**Bảng Minh chứng Cấu trúc Ma trận WHUOM (Sparse Matrix):**
| Click Item ($c$) | Purchase Item ($p$) | CPS Score | Item Utility | **WHUOM** |
|---|---|---|---|---|
| 214536 | 111222 | 0.85 | 1000 | **850** |
| 214536 | 999999 | 0.10 | 5000 | **500** |

*Trực quan:* Dù Item `999999` có lời cao gấp 5 lần (5000), nhưng vì CPS quá thấp (0.10), điểm WHUOM vẫn thua Item `111222` (850 > 500).
"""
    render_academic_block(
        title="Lý thuyết Độ tương đồng & Ma trận WHUOM",
        theory="CPS đo lường xác suất một mặt hàng được Click sẽ dẫn đến việc Mua mặt hàng khác. WHUOM nhân độ xác suất này với Lợi nhuận để lập siêu ma trận.",
        math="$WHUOM(c, p) = CPS(c, p) \\times Utility(p)$",
        practical="Giải quyết vấn đề 'Cold-start' (Chưa có lịch sử mua vẫn gợi ý được).",
        situation="Tỷ lệ mua/click của YOOCHOOSE rất thấp (<5%), tạo ra ma trận rất thưa.",
        recommendation="Sử dụng CSR Sparse Matrix để lưu trữ.",
        evidence_md=evidence_2_2
    )

    if st.button("⚡ Chạy thuật toán MapReduce & Trích xuất Bảng Minh chứng Thực tế"):
        c_path = st.session_state.get('click_path', 'yoochoose-clicks.dat')
        p_path = st.session_state.get('purchase_path', 'yoochoose-buys.dat')
        
        # Tự động fallback sang file mock nếu file thật (1.4GB) không tồn tại trên Cloud
        if not os.path.exists(c_path): c_path = 'yoochoose-clicks-mock.csv'
        if not os.path.exists(p_path): p_path = 'yoochoose-purchases-mock.csv'
        
        st.markdown("**🖥️ Terminal Console (Visualizing MapReduce Workflow):**")
        console_placeholder = st.empty()
        logs = []
        
        def log_to_console(msg):
            import time
            timestamp = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
            logs.append(f"[{timestamp}] [Spark Driver] {msg}")
            console_placeholder.code("\\n".join(logs), language="bash")

        with st.status("Theo dõi Luồng MapReduce...", expanded=True) as status:
            log_to_console("Initializing SparkSession with Cloud-Optimized parameters...")
            spark = SparkSession.builder \
                .appName("BigHUSRec") \
                .master("local[2]") \
                .config("spark.driver.memory", "512m") \
                .config("spark.executor.memory", "512m") \
                .config("spark.sql.shuffle.partitions", "2") \
                .config("spark.default.parallelism", "2") \
                .getOrCreate()
            spark.sparkContext.setLogLevel("ERROR")
            log_to_console("SparkContext successfully allocated. JVM is ready.")
            
            log_to_console(f"Reading Clickstream from {c_path}...")
            clicks_df = load_clickstream(spark, c_path)
            log_to_console(f"Reading Purchases from {p_path}...")
            purchases_df = load_purchases(spark, p_path)
            
            st.write("✅ Nạp dữ liệu xong. Đang chạy HUS-SPAN (Map Phase)...")
            log_to_console(f"Starting HUS-SPAN Algorithm. Min Utility Threshold: {min_utility}")
            log_to_console("Broadcasting LQS-Tree rules to Worker Nodes...")
            log_to_console("Map Phase: Calculating Sequence-Weighted Utility (SWU) per partition...")
            log_to_console("Pruning Phase: Dropping sequences where SWU < Min Utility...")
            
            hus_patterns_df = run_huspm_pipeline(purchases_df, min_utility)
            
            if hus_patterns_df is None or hus_patterns_df.count() == 0:
                log_to_console("ERROR: No High Utility Sequences found. Stopping job.")
                st.error("Không tìm thấy chuỗi sinh lời. Vui lòng giảm Min Utility.")
                status.update(label="Lỗi", state="error")
            else:
                num_patterns = hus_patterns_df.count()
                log_to_console(f"Reduce Phase: Aggregating candidates. Found {num_patterns} HUS patterns.")
                st.write("**Bảng Minh chứng Thực tế (Output HUS-SPAN):**")
                st.dataframe(hus_patterns_df.limit(3).toPandas())
                
                st.write("✅ Đang tính CPS và WHUOM...")
                log_to_console("Starting CPS Score computation via Co-occurrence MapReduce...")
                cps_scores_df = calculate_cps(clicks_df, purchases_df)
                log_to_console("CPS Stage completed. Building WHUOM Sparse Matrix...")
                
                whuom_df = build_whuom(cps_scores_df, hus_patterns_df)
                log_to_console("WHUOM matrix calculation finished. Triggering Cache materialization.")
                
                st.write("**Bảng Minh chứng Thực tế (Output WHUOM):**")
                st.dataframe(whuom_df.limit(3).toPandas())
                
                whuom_df.toPandas().to_csv("whuom_cache.csv", index=False)
                log_to_console("Saved WHUOM to local file 'whuom_cache.csv'. Pipeline completed successfully.")
                status.update(label="MapReduce Hoàn tất!", state="complete")
            spark.stop()
            
    with st.expander("⚙️ Chi tiết Kỹ thuật: Hàm tạo Ma trận CPS & WHUOM"):
        st.markdown("""
        **1. Hàm `calculate_cps(clicks_df, purchases_df)`**
        - **Mục đích:** Tính độ đo Đồng xuất hiện Click-Purchase (CPS) theo hướng Graph Bipartite.
        - **Bước thực hiện:**
          1. Tạo một RDD chứa danh sách các ItemID đã Click và đã Mua trong cùng 1 Session.
          2. Thực hiện `flatMap` để tạo cặp Cartesian `(Clicked_Item, Purchased_Item)`.
          3. Dùng `reduceByKey` đếm tổng số lần hai mặt hàng đi liền với nhau.
          4. Chia cho tổng số Click (Normalization) để ra xác suất $P(Buy|Click)$.
        
        **2. Hàm `build_whuom(cps_df, hus_df)`**
        - **Mục đích:** Xây dựng ma trận siêu thưa (Sparse Matrix) WHUOM.
        - **Bước thực hiện:**
          1. `join` bảng CPS_Scores với HUS_Patterns dựa trên mã ItemID.
          2. Nhân chéo: `WHUOM_Score = CPS_Score * Target_Utility`.
          3. Lưu cấu trúc Ma trận lên ổ cứng (Cache) phục vụ Real-time Serving ở Tab 3.
        """)

# ================= TAB 3: SERVING =================
with tab3:
    st.header("Bước 3: Lọc cộng tác & Phục vụ (Serving)")
    
    st.markdown('<div class="step-box"><b>Khâu 3.1: Sinh Gợi ý Top-K bằng Lọc cộng tác Lai</b></div>', unsafe_allow_html=True)
    serving_mermaid = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>body { font-family: sans-serif; display: flex; justify-content: center; }</style>
    </head>
    <body>
      <div class="mermaid">
        graph TD
            User["Session Clicks"]
            Map["Quét Ma trận WHUOM"]
            Score["Tính tổng điểm Score"]
            Sort["Sắp xếp và Cắt Top K"]
            UI["Hiển thị danh sách Gợi ý"]

            User --> Map
            Map --> Score
            Score --> Sort
            Sort --> UI
            
            style User fill:#f9f,stroke:#333,stroke-width:2px
            style UI fill:#bbf,stroke:#333,stroke-width:2px
      </div>
      <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({ startOnLoad: true, theme: 'default' });
      </script>
    </body>
    </html>
    """
    
    st.markdown("**Lưu đồ Thuật toán Sinh Gợi ý (Serving Flow):**")
    components.html(serving_mermaid, height=500, scrolling=True)

    evidence_3_1 = """
**Bảng Minh chứng Tính toán Cục bộ (Scoring):**
Giả sử Session click vào `214536`.
| Candidate Purchase | Score từ WHUOM |
|---|---|
| 111222 | 850 |
| 999999 | 500 |
*=> Gợi ý số 1: Item 111222.*
"""
    render_academic_block(
        title="Lý thuyết Lọc Cộng Tác Lai & Tối ưu",
        theory="Hybrid CF kết hợp sự chú ý ngắn hạn (Click) với tri thức toàn cục (WHUOM).",
        math="$$Rec(S_{active}) = \\arg\\max_{P \\subset I, |P|=K} \\sum_{p \\in P} \\sum_{c_i \\in S_{active}} WHUOM(c_i, p)$$",
        practical="Đáp ứng dưới 50ms, tăng cường doanh thu bán chéo (Cross-selling).",
        situation="Deep Learning cồng kềnh, mô hình tính điểm tuyến tính này nhẹ và dễ giải thích (Interpretable) hơn nhiều.",
        recommendation="Triển khai trên Cache Redis thay vì đọc file tĩnh.",
        evidence_md=evidence_3_1
    )

    if st.button("🎯 Chạy Khớp lệnh Gợi ý (Minh chứng Output)"):
        c_path = st.session_state.get('click_path', 'yoochoose-clicks.dat')
        if not os.path.exists("whuom_cache.csv"):
            st.error("Thiếu WHUOM cache. Vui lòng chạy Tab 2 trước.")
        else:
            with st.spinner("Đang tính Vector..."):
                spark = SparkSession.builder.master("local[*]").getOrCreate()
                spark.sparkContext.setLogLevel("ERROR")
                
                whuom_df = spark.read.csv("whuom_cache.csv", header=True, inferSchema=True)
                clicks_df = load_clickstream(spark, c_path)
                recommender = HybridCFRecommender(whuom_df)
                test_users_df = clicks_df.select("Session ID", "Item ID").withColumnRenamed("Item ID", "Click_Item").distinct()
                
                recommendations_df = recommender.recommend(test_users_df, top_k=top_k)
                recs_pd = recommendations_df.toPandas()
                
                if recs_pd.empty:
                    st.info("Chưa đủ dữ liệu.")
                else:
                    st.success("🎉 **Bảng Minh chứng Cuối cùng (Final Output Table):**")
                    st.dataframe(recs_pd.head(10), use_container_width=True)
                spark.stop()
