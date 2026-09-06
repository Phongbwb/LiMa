import json
import numpy as np
import os
from post_process.motion.geometry import cosine_similarity 

DIRECTION_ID = {
    "go straight": 0,
    "turn right": 1,
    "turn left": 2
}

def moving_average_smoothing(points, window_size=5):
    """Lọc nhiễu tọa độ tâm Bounding Box."""
    if len(points) < window_size:
        return points
    smoothed = np.copy(points)
    for i in range(len(points)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(points), i + window_size // 2 + 1)
        smoothed[i] = np.mean(points[start_idx:end_idx], axis=0)
    return smoothed

def get_initial_action(list_boxes, min_dist=20.0, cosine_threshold=0.94):
    """
    Hàm chuẩn hóa "Initial Action" dựa trên tư duy code gốc của User.
    """
    if len(list_boxes) < 5:
        return DIRECTION_ID["go straight"]

    raw_points = np.array([
        [box[0] + box[2] / 2.0, box[1] + box[3] / 2.0] 
        for box in list_boxes
    ])
    points = moving_average_smoothing(raw_points, window_size=5)

    p_start = points[0]

    # 1. ĐOẠN CẮT (SLICE): Chỉ lấy 50% số frame đầu tiên để infer
    limit_idx = max(4, len(points) // 2)
    points_slice = points[:limit_idx]
    
    p_end_slice = points_slice[-1]
    v_slice_total = p_end_slice - p_start
    dist_slice = np.linalg.norm(v_slice_total)
    
    # Nếu trong 50% thời gian đầu mà xe đi chưa tới min_dist (20 pixel) -> Đi thẳng
    if dist_slice < min_dist:
        return DIRECTION_ID["go straight"]

    # 2. DYNAMIC INITIAL VECTOR: Né bẫy dừng đèn đỏ
    # Thay vì lấy điểm ở len//3, ta tìm điểm đầu tiên đi được 40% chiều dài của đoạn cắt
    idx_init = 1
    target_dist = dist_slice * 0.4
    
    for i in range(1, len(points_slice)):
        if np.linalg.norm(points_slice[i] - p_start) >= target_dist:
            idx_init = i
            break
            
    v_init = points_slice[idx_init] - p_start
    if np.linalg.norm(v_init) < 1e-5:
        return DIRECTION_ID["go straight"]

    # 3. TOÁN HỌC GỐC: So sánh hướng ban đầu với hướng tổng (của đoạn cắt 50%)
    cos_sim = cosine_similarity(v_init, v_slice_total)

    # Góc lệch trong quãng đường ngắn cần ngưỡng Cosine lớn (0.94) để nhận diện
    if cos_sim > cosine_threshold:
        return DIRECTION_ID["go straight"]

    cross_prod = np.cross(v_init, v_slice_total)
    
    if cross_prod < 0:
        return DIRECTION_ID["turn left"]
    
    return DIRECTION_ID["turn right"]

def generate_cur_result(tracks_path, output_path, min_dist=20.0, cosine_thresh=0.94):
    with open(tracks_path, 'r') as f:
        full_uuid = json.load(f)

    cur_result = {}
    for query, data in full_uuid.items():
        boxes = data.get("boxes", [])
        
        initial_action = get_initial_action(
            boxes, 
            min_dist=min_dist,
            cosine_threshold=cosine_thresh
        )
        
        cur_result[query] = {"id": initial_action}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(cur_result, f, indent=4)
        
if __name__ == "__main__":
    TRACKS_FILE = "./data/json/test-tracks.json"
    OUTPUT_FILE = "./data/json/test_tracks_direction.json"
    
    generate_cur_result(TRACKS_FILE, OUTPUT_FILE, min_dist=20.0, cosine_thresh=0.94)