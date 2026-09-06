import json
import os

def load_json_file(file_path):
    """Hàm đọc file JSON và xử lý lỗi."""
    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy file tại '{file_path}'")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"❌ Lỗi: File '{file_path}' không đúng định dạng JSON.")
        return None
    except Exception as e:
        print(f"❌ Lỗi không xác định khi đọc file '{file_path}': {e}")
        return None

def is_match(pred_val, gt_val):
    """
    Hàm so sánh thông minh: 
    Tự động chuẩn hóa mọi giá trị (số, chuỗi, list) về dạng set (tập hợp) để so sánh.
    Bỏ qua thứ tự (VD: [0, 2] sẽ khớp với [2, 0])
    """
    # Nếu giá trị không phải là list, ta bọc nó vào trong list rồi chuyển thành set
    set_pred = set(pred_val) if isinstance(pred_val, list) else {pred_val}
    set_gt = set(gt_val) if isinstance(gt_val, list) else {gt_val}
    
    # So sánh 2 tập hợp
    return set_pred == set_gt

def calculate_accuracy(pred_file_path, gt_file_path):
    """Hàm chính để tính toán Accuracy."""
    # Đọc dữ liệu
    pred_data = load_json_file(pred_file_path)
    gt_data = load_json_file(gt_file_path)
    
    # Dừng lại nếu 1 trong 2 file bị lỗi
    if pred_data is None or gt_data is None:
        return
        
    # Trích xuất dữ liệu theo thứ tự từ trên xuống
    try:
        pred_ids = [val['id'] for val in pred_data.values()]
        gt_ids = [val['id'] for val in gt_data.values()]
    except KeyError:
        print("❌ Lỗi: Dữ liệu JSON không chứa key 'id' ở các phần tử con.")
        return
    
    # Kiểm tra số lượng phần tử
    if len(pred_ids) != len(gt_ids):
        print(f"⚠️ Cảnh báo: Số lượng khác nhau! (Dự đoán: {len(pred_ids)}, Thực tế: {len(gt_ids)})")
    
    total_items = min(len(pred_ids), len(gt_ids))
    
    if total_items == 0:
        print("⚠️ Không có dữ liệu để so sánh.")
        return
        
    # Đếm số lượng khớp nhau bằng hàm is_match()
    correct_count = sum(1 for p, g in zip(pred_ids, gt_ids) if is_match(p, g))
    
    # Tính Accuracy
    accuracy = correct_count / total_items
    
    # In kết quả
    print("\n" + "="*45)
    print("📊 KẾT QUẢ ĐÁNH GIÁ (ACCURACY)")
    print("="*45)
    print(f"File dự đoán : {pred_file_path}")
    print(f"File thực tế : {gt_file_path}")
    print("-" * 45)
    print(f"Tổng số mẫu so sánh : {total_items}")
    print(f"Số mẫu khớp hoàn toàn: {correct_count}")
    print(f"Độ chính xác (Acc)  : {accuracy * 100:.2f}%")
    print("="*45 + "\n")

if __name__ == "__main__":
    # =========================================================
    # ĐIỀN ĐƯỜNG DẪN FILE CỦA BẠN VÀO 2 BIẾN DƯỚI ĐÂY
    # Lưu ý: Dùng dấu gạch chéo xuôi (/) thay vì ngược (\)
    # =========================================================
    predict_path = "./submission.json"  # <--- Sửa đường dẫn ở đây
    ground_truth_path = "./data/json/recognition/test-queries-direction.json"  # <--- Sửa đường dẫn ở đây

    # Chạy hàm tính toán
    calculate_accuracy(predict_path, ground_truth_path)