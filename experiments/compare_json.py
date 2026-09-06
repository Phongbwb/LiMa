import json

def extract_target_id(id_value):
    """
    Hàm xử lý: Nếu là list thì lấy phần tử đầu tiên, 
    nếu không phải list thì lấy chính nó.
    """
    if isinstance(id_value, list):
        return id_value[0] if len(id_value) > 0 else None
    return id_value

def compare_json_files(file1_path, file2_path):
    # Đọc dữ liệu từ 2 file
    with open(file1_path, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
    with open(file2_path, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)

    # Chuyển values của dict thành danh sách để ép thứ tự từ trên xuống
    items1 = list(data1.values())
    items2 = list(data2.values())
    
    # Lấy key (UUID) để in ra cho dễ theo dõi
    keys1 = list(data1.keys())
    keys2 = list(data2.keys())

    # Số lượng vòng lặp bằng độ dài của file ngắn hơn
    total = min(len(items1), len(items2))
    matched_count = 0

    print(f"{'STT':<4} | {'ID File 1':<10} | {'ID File 2':<10} | {'Kết quả':<7} | Ghi chú (UUID 1)")
    print("-" * 80)

    for i in range(total):
        # Trích xuất dữ liệu gốc
        raw_id1 = items1[i].get("id")
        raw_id2 = items2[i].get("id")

        # Áp dụng logic: Lấy list[0] hoặc lấy chính nó
        val1 = extract_target_id(raw_id1)
        val2 = extract_target_id(raw_id2)

        # So khớp
        is_match = (val1 == val2)
        if is_match:
            matched_count += 1
            
        status = "GIỐNG" if is_match else "KHÁC"
        
        # In kết quả từng dòng
        uuid_short = keys1[i].split("-")[0] # Cắt ngắn UUID cho dễ nhìn
        print(f"{i+1:<4} | {str(val1):<10} | {str(val2):<10} | {status:<7} | {uuid_short}...")

    # Tổng kết
    print("-" * 80)
    print(f"Tổng số track đã duyệt: {total}")
    print(f"Số lượng khớp: {matched_count}/{total} ({matched_count/total*100:.2f}%)")

# ==========================================
# CÁCH SỬ DỤNG
# ==========================================
if __name__ == "__main__":
    # Thay đường dẫn tới 2 file json 
    file1 = "./data/json/test_queries_color.json" 
    file2 = "./data/json/recognition/test-queries-color.json"
    
    compare_json_files(file1, file2)