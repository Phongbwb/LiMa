import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from tqdm import tqdm
import textwrap

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
DATASET_ROOT = "./data/cityflownl/data"
TRACKS_JSON = "./data/json/test-tracks.json"
BASELINE_FAILURE_JSON = "./data/json/retrieval/baseline_failure_cases.json"

# ==========================================
# CÁC HÀM XỬ LÝ DỮ LIỆU & ẢNH
# ==========================================
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def pad_to_square(img, fill_color=(255, 255, 255)):
    """Đệm thêm viền màu để ảnh crop trở thành hình vuông hoàn hảo."""
    x, y = img.size
    size = max(x, y)
    new_img = Image.new('RGB', (size, size), fill_color)
    new_img.paste(img, (int((size - x) / 2), int((size - y) / 2)))
    return new_img

def get_crop_image(track_id, tracks_db, dataset_root, frame_idx=0):
    """Load, crop và đệm ảnh xe từ một track dựa trên bounding box."""
    if track_id not in tracks_db:
        return None
    track_data = tracks_db[track_id]
    if not track_data["frames"] or frame_idx >= len(track_data["frames"]):
        return None
    frame_path = track_data["frames"][frame_idx]
    box = track_data["boxes"][frame_idx]
    full_path = os.path.join(dataset_root, frame_path)
    if not os.path.exists(full_path):
        return None
    try:
        with Image.open(full_path) as img:
            left, top = box[0], box[1]
            right, bottom = box[0] + box[2], box[1] + box[3]
            crop_img = img.crop((left, top, right, bottom)).convert('RGB')
            return pad_to_square(crop_img)
    except Exception as e:
        print(f"Error cropping {track_id}: {e}")
        return None

# ==========================================
# HÀM VISUALIZE CHÍNH
# ==========================================
def visualize_failures(failure_path, tracks_path, output_path, dataset_root, num_cases=3, top_k_fp=3):
    print(f"🚀 Đang bắt đầu visualize {num_cases} trường hợp lỗi...")
    failure_data = load_json(failure_path)
    tracks_db = load_json(tracks_path)
    
    # Sắp xếp theo thứ tự ưu tiên
    failure_data_sorted = sorted(failure_data, key=lambda x: x["ground_truth"]["rank"], reverse=True)
    cases_to_plot = failure_data_sorted[:num_cases]
    
    # Khởi tạo Figure với GridSpec chuyên nghiệp
    fig = plt.figure(figsize=(22, 5 * num_cases))
    # Tỉ lệ: Cột text (1.2) + GT (1) + FP_1 (1) + FP_2 (1) + FP_3 (1)
    col_ratios = [1.2] + [1] * (1 + top_k_fp)
    gs = gridspec.GridSpec(num_cases, 2 + top_k_fp, width_ratios=col_ratios, wspace=0.15, hspace=0.45)

    for i, case in enumerate(tqdm(cases_to_plot, desc="Processing cases")):
        q_uuid = case["query_uuid"]
        text_nl = case["text_query"]["nl"]
        gt_info = case["ground_truth"]
        fp_list = case.get(f"top_{len(case.get('top_5_false_positives', []))}_false_positives", [])[:top_k_fp]
        
        # --- CỘT 0: Text Requirement ---
        ax_text = fig.add_subplot(gs[i, 0])
        ax_text.axis('off')
        wrapped_text = textwrap.fill(text_nl, width=28)
        display_str = f"Query ID: {q_uuid[:8]}...\n\nText Requirement:\n'{wrapped_text}'"
        ax_text.text(0.5, 0.5, display_str, ha='center', va='center', fontsize=18, fontweight='bold')
        
        # --- CỘT 1: Ground Truth ---
        ax_gt = fig.add_subplot(gs[i, 1])
        gt_img = get_crop_image(gt_info["track_uuid"], tracks_db, dataset_root)
        if gt_img:
            ax_gt.imshow(gt_img)
            ax_gt.set_title(f"GROUND TRUTH\n(Rank: {gt_info['rank']}, Score: {gt_info['score']:.3f})", 
                            color='#2CA02C', fontsize=15, fontweight='bold', pad=10)
        
        # Style viền GT
        for spine in ax_gt.spines.values():
            spine.set_edgecolor('#2CA02C')
            spine.set_linewidth(5)
        ax_gt.set_xticks([]); ax_gt.set_yticks([])

        # --- CỘT 2 -> N: False Positives ---
        for j in range(top_k_fp):
            ax_fp = fig.add_subplot(gs[i, j+2])
            if j < len(fp_list):
                fp_img = get_crop_image(fp_list[j]["track_uuid"], tracks_db, dataset_root)
                if fp_img:
                    ax_fp.imshow(fp_img)
                    ax_fp.set_title(f"FALSE POSITIVE #{j+1}\n(Score: {fp_list[j]['score']:.3f})", 
                                    color='#D62728', fontsize=15, fontweight='bold', pad=10)
            
            # Style viền FP
            for spine in ax_fp.spines.values():
                spine.set_edgecolor('#D62728')
                spine.set_linewidth(5)
            ax_fp.set_xticks([]); ax_fp.set_yticks([])

    # Lưu file với chuẩn DPI cao
    output_dir = os.path.dirname(output_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã xuất báo cáo tại: {output_path}")

if __name__ == "__main__":
    visualize_failures(
        failure_path=BASELINE_FAILURE_JSON,
        tracks_path=TRACKS_JSON,
        output_path="./retrieval/baseline_qualitative_analysis_paper_ready.png",
        dataset_root=DATASET_ROOT,
        num_cases=3, 
        top_k_fp=3   
    )