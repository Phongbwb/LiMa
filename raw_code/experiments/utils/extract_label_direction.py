import json
import os
import re
import numpy as np
from collections import Counter

# Hàm hỗ trợ lấy số thứ tự từ tên file để sắp xếp (VD: "img_015.jpg" -> 15)
def get_sort_key(f_path):
    nums = re.findall(r'\d+', str(f_path))
    return (int(nums[-1]), f_path) if nums else (0, f_path)

def extract_5_frames_per_label(direction_json_path, track_json_path, output_json_path):
    print("1. Đang đọc dữ liệu đầu vào...")
    with open(direction_json_path, 'r', encoding='utf-8') as f:
        direction_data = json.load(f)
        
    with open(track_json_path, 'r', encoding='utf-8') as f:
        track_data = json.load(f)

    # ---------------------------------------------------------
    # BƯỚC 1: TẠO TỪ ĐIỂN TRA CỨU NHÃN
    # ---------------------------------------------------------
    label_map = {}
    for key, item in direction_data.items():
        # Giữ nguyên logic cũ: map 1 frame + 1 box -> label
        lookup_key = (item["frames"], tuple(item["boxes"]))
        label_map[lookup_key] = item["id"]

    # ---------------------------------------------------------
    # BƯỚC 2: TÁCH NHÓM VÀ ÁP DỤNG SLIDING WINDOW
    # ---------------------------------------------------------
    final_dataset = {}
    sample_idx = 0
    class_counts = Counter() 

    print("2. Đang quét track, áp dụng Cửa Sổ Trượt (Sliding Window)...")
    
    for track_uuid, track_info in track_data.items():
        raw_frames = track_info["frames"]
        raw_boxes = track_info["boxes"]

        # 🔥 QUAN TRỌNG: Sắp xếp lại khung hình và hộp tọa độ theo chuẩn thời gian
        combined = list(zip(raw_frames, raw_boxes))
        combined.sort(key=lambda x: get_sort_key(x[0]))
        
        if not combined:
            continue
            
        frames, boxes = zip(*combined)
        frames, boxes = list(frames), list(boxes)

        current_label = None
        segment_frames = []
        segment_boxes = []

        # Hàm phụ xử lý cắt mẫu bằng SLIDING WINDOW
        def save_segment(seg_frames, seg_boxes, label):
            nonlocal sample_idx
            num_frames = len(seg_frames)
            if num_frames == 0 or label is None: 
                return

            seq_len = 5 # Số frame cố định cho mỗi mẫu
            
            # 🔥 CHIẾN LƯỢC BƯỚC NHẢY (STRIDE) THEO NHÃN
            if label in [1, 2]: 
                # Rẽ phải (1), Rẽ trái (2): Trượt từng frame một để ép ra tối đa data
                stride = 1
            elif label == 3:
                # Dừng/Khác (3): Dữ liệu cũng hiếm, trượt dày
                stride = 1
            else:
                # Đi thẳng (0): Dữ liệu quá nhiều, nhảy thưa để chống quá tải
                stride = 5 

            # Cắt cửa sổ trượt
            if num_frames >= seq_len:
                for start_idx in range(0, num_frames - seq_len + 1, stride):
                    indices = list(range(start_idx, start_idx + seq_len))
                    
                    final_dataset[f"sample_{sample_idx}"] = {
                        "track_id": track_uuid,
                        "frames": [seg_frames[i] for i in indices],
                        "boxes": [seg_boxes[i] for i in indices],
                        "label": label
                    }
                    sample_idx += 1
                    class_counts[label] += 1
            else:
                # Fallback: Nếu track quá ngắn (< 5 frames), đệm (pad) bằng frame đầu tiên.
                # Sửa (0, seq_len - num_frames) thành (seq_len - num_frames, 0) để đệm ở đầu (quá khứ).
                indices = np.pad(np.arange(num_frames), (seq_len - num_frames, 0), mode='edge')
                final_dataset[f"sample_{sample_idx}"] = {
                    "track_id": track_uuid,
                    "frames": [seg_frames[i] for i in indices],
                    "boxes": [seg_boxes[i] for i in indices],
                    "label": label
                }
                sample_idx += 1
                class_counts[label] += 1

        # Duyệt qua từng frame để gom nhóm
        for f_path, box in zip(frames, boxes):
            lookup_key = (f_path, tuple(box))
            
            # Cập nhật nhãn nếu tìm thấy, nếu không dùng nhãn gần nhất
            label = label_map.get(lookup_key, current_label)

            if current_label is None:
                current_label = label

            if label == current_label:
                segment_frames.append(f_path)
                segment_boxes.append(box)
            else:
                # Chuyển nhãn (vd: đang thẳng -> rẽ) -> Lưu đoạn cũ
                save_segment(segment_frames, segment_boxes, current_label)
                
                # Bắt đầu ghi hình đoạn mới
                current_label = label
                segment_frames = [f_path]
                segment_boxes = [box]

        # Lưu nốt đoạn còn sót lại khi hết track
        if current_label is not None:
            save_segment(segment_frames, segment_boxes, current_label)

    # ---------------------------------------------------------
    # BƯỚC 3: LƯU KẾT QUẢ VÀ IN BÁO CÁO
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=4, ensure_ascii=False)
        
    print("\n" + "="*50)
    print(" BÁO CÁO KẾT QUẢ TẠO DỮ LIỆU (SLIDING WINDOW)")
    print("="*50)
    print(f" - Tổng số track gốc       : {len(track_data)}")
    print(f" - Số lượng mẫu sinh ra    : {len(final_dataset)} (Mỗi mẫu đúng 5 frames)")
    
    print("-" * 50)
    print(" 📊 THỐNG KÊ SỐ LƯỢNG MẪU THEO NHÃN (ID):")
    label_names = {0: "Đi thẳng (0)", 1: "Rẽ phải (1)", 2: "Rẽ trái (2)", 3: "Dừng/Khác (3)"}
    
    # In ra báo cáo theo thứ tự ID
    for label_id in sorted([k for k in class_counts.keys() if k is not None]):
        name = label_names.get(label_id, f"ID {label_id}")
        count = class_counts[label_id]
        print(f"   + {name:<17}: {count} mẫu")
        
    print("-" * 50)
    print(f" - File dữ liệu hoàn chỉnh : {output_json_path}")
    print("="*50)


if __name__ == "__main__":
    DIRECTION_JSON = r"./data/json/direction_trainset_train.json"
    TRACK_JSON = r"./data/json/train_clean.json"
    # Đổi tên file output để không ghi đè bản cũ nếu muốn so sánh
    OUTPUT_JSON = r"./data/json/intent_trainset_5frames_sliding.json" 
    
    extract_5_frames_per_label(DIRECTION_JSON, TRACK_JSON, OUTPUT_JSON)