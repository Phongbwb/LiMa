import re

class LiMaQueryParser:
    """
    Bộ phân tích cú pháp siêu nhẹ dành riêng cho truy vấn phương tiện giao thông.
    Nhiệm vụ: Cắt câu lệnh phức tạp thành Main Query (Magno) và Sub Query (Parvo).
    """
    def __init__(self):
        # 1. Tập từ khóa tách biệt (Separators)
        # Đây là các động từ/giới từ thường dùng để ngăn cách đối tượng chính và chi tiết phụ
        self.separators = [
            "chở", "kéo theo", "kéo", "mang", "có", 
            "chứa", "đèo", "với", "gắn", "phía sau có"
        ]
        
        # 2. Tập từ nhiễu (Stopwords) cần loại bỏ để Vector Embedding chính xác hơn
        self.stopwords = [
            "tìm cho tôi", "truy vết", "tìm kiếm", "hãy tìm", 
            "hiển thị", "cái", "chiếc", "loại"
        ]

    def clean_text(self, text):
        """Tiền xử lý: Xóa từ nhiễu và khoảng trắng thừa."""
        text = text.lower().strip()
        for word in self.stopwords:
            text = text.replace(word, "")
        # Xóa khoảng trắng kép
        return re.sub(r'\s+', ' ', text).strip()

    def parse(self, raw_query):
        """
        Hàm phân tích chính.
        Trả về: (main_query, sub_query)
        """
        query = self.clean_text(raw_query)
        
        main_query = query
        sub_query = None
        
        # Tìm từ khóa ngăn cách xuất hiện sớm nhất trong câu
        earliest_idx = len(query)
        best_separator = None
        
        for sep in self.separators:
            # Tìm từ khóa, có khoảng trắng 2 bên để tránh bắt nhầm (VD: "có" trong "xe bò")
            pattern = rf'\b{sep}\b'
            match = re.search(pattern, query)
            
            if match and match.start() < earliest_idx:
                earliest_idx = match.start()
                best_separator = sep
                
        # Nếu tìm thấy từ ngăn cách, tiến hành cắt đôi câu lệnh
        if best_separator:
            # Main query là phần trước từ khóa
            main_query = query[:earliest_idx].strip()
            
            # Sub query là phần từ từ khóa trở về sau (giữ lại từ khóa để CLIP hiểu ngữ cảnh)
            # VD: "chở thùng xốp" sẽ tốt hơn chỉ là "thùng xốp"
            sub_query = query[earliest_idx:].strip()

        return main_query, sub_query

# ==========================================
# KIỂM THỬ (TESTING)
# ==========================================
if __name__ == "__main__":
    parser = LiMaQueryParser()
    
    test_queries = [
        "xe tải màu đỏ chở tôn",
        "tìm cho tôi chiếc xe máy đèo 3 người",
        "xe khách 45 chỗ màu xanh",
        "truy vết xe bán tải kéo theo rơ moóc",
        "xe ô tô màu đen có cửa sổ trời"
    ]
    
    print(f"{'CÂU LỆNH GỐC':<40} | {'MAIN QUERY (MAGNO)':<25} | {'SUB QUERY (PARVO)':<25}")
    print("-" * 95)
    
    for q in test_queries:
        main_q, sub_q = parser.parse(q)
        print(f"{q:<40} | {str(main_q):<25} | {str(sub_q):<25}")