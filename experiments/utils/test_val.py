import os
import json
from torch.utils.data import DataLoader

# Giả định bạn đã import dataset của mình
from experiments.utils.dataset import CityFlowNLDataset

def debug_validation_mapping(val_json_path, queries_json_path, data_root, text_emb_path):
    print("="*60)
    print("🔍 BẮT ĐẦU DEBUG LOGIC MAP UUID VÀ QUERY TẬP VALIDATION")
    print("="*60)

    # 1. Đọc dữ liệu thô từ file JSON
    print(f"📖 Đang đọc {val_json_path}...")
    with open(val_json_path, 'r') as f:
        full_track_ids = list(json.load(f).keys())
        
    print(f"📖 Đang đọc {queries_json_path}...")
    with open(queries_json_path, 'r') as f:
        queries_data = json.load(f)
    full_query_ids = list(queries_data.keys())

    # Kiểm tra số lượng
    print(f"📊 Tổng số Tracks trong file JSON: {len(full_track_ids)}")
    print(f"📊 Tổng số Queries trong file JSON: {len(full_query_ids)}")
    
    if len(full_track_ids) != len(full_query_ids):
        print("⚠️ CẢNH BÁO: Số lượng Track và Query không bằng nhau! Logic ánh xạ qua index có thể bị lệch.")
    else:
        print("✅ Số lượng Track và Query khớp nhau. Tiếp tục...")

    # 2. Khởi tạo Dataset và Dataloader
    print("\n📦 Đang khởi tạo Validation DataLoader...")
    val_dataset = CityFlowNLDataset(val_json_path, data_root, text_emb_path, max_frames=8) 
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False) # Không shuffle để dễ dò
    
    ordered_track_ids = val_dataset.track_ids
    print(f"📦 Dataloader đã nạp thành công {len(ordered_track_ids)} videos.\n")

    # 3. Chạy vòng lặp kiểm tra
    print("="*60)
    print(f"{'STT':<5} | {'TRACK UUID':<40} | {'CÂU QUERY TƯƠNG ỨNG'}")
    print("="*60)
    
    # Chỉ in ra 20 cặp đầu tiên để tránh trôi terminal
    max_print = 20 
    
    for i, tid in enumerate(ordered_track_ids):
        if i >= max_print:
            print(f"... (Còn {len(ordered_track_ids) - max_print} videos nữa)")
            break
            
        try:
            # Logic của bạn: Tìm vị trí gốc
            original_idx = full_track_ids.index(tid)
            
            # Bốc query tương ứng
            q_id = full_query_ids[original_idx]
            nl_text = queries_data[q_id]['nl'][0].strip()
            
            print(f"{i+1:<5} | {tid:<40} | {nl_text}")
            
        except ValueError:
            print(f"{i+1:<5} | {tid:<40} | ❌ LỖI: Không tìm thấy UUID trong JSON gốc!")
        except IndexError:
            print(f"{i+1:<5} | {tid:<40} | ❌ LỖI: Index bị lệch, không lấy được Query!")

    print("="*60)
    print("✅ HOÀN TẤT DEBUG!")

if __name__ == "__main__":
    # Cài đặt đường dẫn chuẩn của bạn ở đây
    DATA_ROOT = "./data/data"
    VAL_JSON = os.path.join(DATA_ROOT, "test-tracks.json")
    VAL_QUERIES = os.path.join(DATA_ROOT, "test-queries.json")
    TEXT_EMB_PATH = os.path.join(DATA_ROOT, "clip_text_tokens.pt")
    
    debug_validation_mapping(VAL_JSON, VAL_QUERIES, DATA_ROOT, TEXT_EMB_PATH)