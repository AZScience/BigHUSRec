# Simplified implementation of HUS-SPAN for a MapReduce (Spark) context.
# In a real Big Data system, this would run inside MapPartitions.

class UtilityList:
    def __init__(self, item):
        self.item = item
        self.elements = [] # List of (tid, iutil, rutil)
        
    def add_element(self, tid, iutil, rutil):
        self.elements.append({"tid": tid, "iutil": iutil, "rutil": rutil})

class LQSTreeNode:
    def __init__(self, sequence):
        self.sequence = sequence
        self.utility_list = None
        self.children = []
        
    def add_child(self, child_node):
        self.children.append(child_node)

def calculate_swu(database):
    """
    Tính Sequence-Weighted Utility (SWU) cho mỗi item.
    database: list of tuples (sequence_of_itemsets, total_utility_of_sequence)
    """
    swu = {}
    for seq, tu in database:
        # Tìm unique items trong sequence
        unique_items = set()
        for itemset in seq:
            for item, item_util in itemset:
                unique_items.add(item)
        
        for item in unique_items:
            swu[item] = swu.get(item, 0) + tu
    return swu

def hus_span_local(database, min_utility):
    """
    Chạy HUS-SPAN trên một phân vùng (partition) dữ liệu.
    database: format [( [ (item1, util1), (item2, util2) ], TU ), ...]
    Returns: list of (high_utility_sequence, utility)
    """
    # 1. Tính SWU và lọc các item không hứa hẹn
    swu_map = calculate_swu(database)
    promising_items = {item for item, swu in swu_map.items() if swu >= min_utility}
    
    # 2. Xây dựng cây LQS-Tree ảo và Utility Lists ban đầu (cho 1-sequences)
    # Đây là phiên bản rút gọn, logic thực tế sẽ thực hiện I-Extension và S-Extension
    
    hus_patterns = []
    
    # Dummy logic để tìm chuỗi (thực tế cần đệ quy tạo Utility-Lists)
    # Ở đây chúng ta duyệt qua database để giả lập việc khai phá
    item_utilities = {}
    for seq, tu in database:
        for itemset in seq:
            for item, util in itemset:
                if item in promising_items:
                    item_utilities[item] = item_utilities.get(item, 0) + util
                    
    # Lọc ra High Utility 1-Sequences
    for item, util in item_utilities.items():
        if util >= min_utility:
            hus_patterns.append(( [item], util ))
            
    # Trong phiên bản đầy đủ, từ 1-sequences ta tiếp tục đệ quy (I-ext, S-ext)
    # kết hợp tính toán upper-bounds (như PEU) để cắt tỉa (pruning) LQS-Tree.
    
    return hus_patterns

def run_hus_span_partition(iterator, min_utility):
    """
    Hàm này được dùng trong rdd.mapPartitions của Spark
    """
    local_db = list(iterator)
    if not local_db:
        return []
    
    # Parse format
    database = []
    for row in local_db:
        # row: (session_id, [ (item_id, profit), ... ])
        # Chuyển đổi thành dạng sequence of itemsets 
        # (Ở đây giả định mỗi click/purchase là 1 itemset size 1)
        seq = [ [(item_id, profit)] for item_id, profit in row[1] ]
        total_utility = sum(profit for item_id, profit in row[1])
        database.append((seq, total_utility))
        
    patterns = hus_span_local(database, min_utility)
    return patterns
