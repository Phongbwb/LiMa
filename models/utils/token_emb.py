import json
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
import os

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Đang chạy trên {device}...")

    # 1. Tải mô hình CLIP Text Encoder
    print("📥 Đang tải mô hình CLIP...")
    model_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)
    text_encoder.eval()

    # 2. Danh sách các file JSON cần trích xuất
    # Thêm đường dẫn test-queries.json vào đây
    json_files = [
        './data/data/train-tracks.json', 
        './data/data/test-queries.json'
    ]
    
    text_to_emb = {}
    MAX_LENGTH = 32

    for json_path in json_files:
        if not os.path.exists(json_path):
            print(f"⚠️ Cảnh báo: Không tìm thấy file {json_path}, bỏ qua...")
            continue
            
        print(f"📖 Đang xử lý: {json_path}")
        with open(json_path, 'r') as f:
            raw_data = json.load(f)

        # Trích xuất tất cả các câu NL (Natural Language)
        for item_id, item_info in tqdm(raw_data.items()):
            # Lấy nl từ cả train (id xe) và test (id query)
            all_queries = item_info.get('nl', []) + item_info.get('nl_other_views', [])
            
            for nl_query in all_queries:
                clean_text = nl_query.strip().lower()
                
                if clean_text not in text_to_emb:
                    inputs = tokenizer(
                        nl_query, 
                        padding='max_length', 
                        truncation=True, 
                        max_length=MAX_LENGTH,
                        return_tensors="pt"
                    ).to(device)
                    
                    with torch.no_grad():
                        outputs = text_encoder(**inputs)
                        token_embeddings = outputs.last_hidden_state # [1, 32, 512]
                    
                    # Chuyển về CPU để tiết kiệm VRAM cho bước sau
                    text_to_emb[clean_text] = token_embeddings.squeeze(0).cpu()

    # 3. Lưu toàn bộ từ điển
    save_path = './data/data/clip_text_tokens.pt'
    torch.save(text_to_emb, save_path)
    print(f"\n✅ Hoàn tất! Đã lưu {len(text_to_emb)} chuỗi ngôn ngữ vào {save_path}")

if __name__ == "__main__":
    main()