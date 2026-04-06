import json
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
import os

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang chạy trên {device}...")

    # 1. Tải mô hình CLIP Text Encoder từ Hugging Face
    print("Đang tải mô hình CLIP...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_encoder.eval()

    # 2. Đọc file JSON chứa dữ liệu
    json_path = './data/data/train-tracks.json'
    print(f"Đang đọc {json_path}...")
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    # Dictionary lưu trữ: { "câu văn miêu tả": tensor_512 }
    text_to_emb = {}

    print("Đang trích xuất đặc trưng văn bản...")
    # Lặp qua tất cả các xe
    for track_id, track_info in tqdm(raw_data.items()):
        for nl_query in track_info['nl']:
            # Nếu câu này chưa được mã hóa thì tiến hành mã hóa
            if nl_query not in text_to_emb:
                # Tokenize và đưa lên GPU
                inputs = tokenizer(nl_query, padding=True, truncation=True, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    # pooler_output là vector 512 chiều đại diện cho toàn bộ câu
                    emb = text_encoder(**inputs).pooler_output 
                
                # Chuyển về CPU để tránh tràn RAM, và xóa chiều batch (squeeze)
                text_to_emb[nl_query] = emb.squeeze(0).cpu()

    # 3. Lưu toàn bộ từ điển ra file
    save_path = './data/data/clip_text_embeddings.pt'
    torch.save(text_to_emb, save_path)
    print(f"\nHoàn tất! Đã lưu {len(text_to_emb)} vector ngôn ngữ vào {save_path}")

if __name__ == "__main__":
    main()