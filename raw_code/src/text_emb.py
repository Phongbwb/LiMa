import json
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
import os
import re
import spacy

# ==============================================================================
# 0. KHỞI TẠO MÔ HÌNH NLP
# ==============================================================================
# Tải mô hình tiếng Anh của spacy (để bên ngoài hàm để không bị load lại nhiều lần)
print("📥 Đang tải mô hình spaCy (en_core_web_sm)...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ Không tìm thấy mô hình spaCy. Vui lòng chạy lệnh: python -m spacy download en_core_web_sm")
    exit()

# ==============================================================================
# 1. BỘ TỪ ĐIỂN CHUẨN HÓA
# ==============================================================================

SIZE_MAPPING = {
    "big": r"\b(huge|big|large|giant)\b",
    "medium": r"\b(midsized|midsize|medium(?!\s+(rate|speed))|semi)\b", 
    "small": r"\b(smaller|small|mini)\b"
}

COLOR_MAPPING = {
    "red": r"\b(dark-red|dark red|wine-red|reddish|burgundy|scarlet|rufous|res|red)\b",
    "silver": r"\b(dark silver|silver|sliver)\b",
    "grey": r"\b(dark gray|dark grey|taupe|grey|gray)\b",
    "blue": r"\b(dark-blue|dark blue|blue)\b",
    "maroon": r"\b(dark-maroon|maroon)\b",
    "white": r"\b(whitish|white|whit)\b",
    "black": r"\b(black|dark)\b",
    "purple": r"\b(purple)\b",
    "yellow": r"\b(metallic|yellow)\b",
    "orange": r"\b(orange)\b",
    "green": r"\b(green)\b",
    "gold": r"\b(golden|gold)\b",
    "brown": r"\b(brown-ish|brownish|sienna|brown|bay)\b",
    "tan": r"\b(tan)\b",
    "beige": r"\b(beige)\b",
    "bronze": r"\b(bronze)\b"
}

TYPE_MAPPING = {
    "vehicle": r"\b(ford mustang|chevrolet|mercedes|vehicle|chevvy|toyota|subaru|sports|honda|chevy|audi|car|can)\b",
    "pickup-truck": r"\b(cargo pickup truck|cargo truck|pickup truck|pick up truck|pickuptruck|pick up|pickup)\b",
    "truck": r"\b(tractortrailer|semitruck|truck|flatbed|18 wheeler|wheeler)\b",
    "crossover": r"\b(cross over|crossover)\b", 
    "coupe": r"\b(mini cooper|couple|coupe|coup)\b",
    "jeep": r"\b(cherokee|jeep)\b",
    "suv": r"\b(suv|spv|svu)\b",
    "sedan": r"\b(sedan|sede)\b",
    "hatchback": r"\b(hatchback|hatckback)\b",
    "van": r"\b(minivan|van)\b",
    "wagon": r"\b(wagon)\b",
    "mpv": r"\b(mpv)\b",
    "caravan": r"\b(caravan)\b",
    "bike": r"\b(bike)\b",
    "bus": r"\b(bus)\b",
    "taxi": r"\b(taxi)\b"
}

MOTION_MAPPING = {
    "go straight": r"\b(drive straight down|keep straight down|run straight down|continue straight|continue forward|drving straight|proceed straight|drive solo down|drive straight|keeps straight|keep straight|head straight|move straight|straight down|continue down|drive forward|drive past|drive down|go straight|goes straight|move foward|run straight|travel down|move ahead|accelerate|run across|go across|cross straight|drive up|continue|go forward|slow down|move down|when down|keep run|run down|go down|run up|travel|enters|drive|enter|cross|run through|run|speed|switch lane|change lane|change lanes|switch lanes|go trough|lead|without stop)\b",
    "turn left": r"\b(turn slightly left|turn street left|make a left turn|make left turn|left turning|turn left|tuns left|left turn|take left|make left|goes left|go left)\b",
    "turn right": r"\b(turn slightly right|make a right turn|make right turn|make slight right|travel right|turn right|right turn|take right|take righ|make right|curve right|goes right|go right)\b",
    "stop": r"\b(make stop|pause|wait|stop)\b",
    "special": r"\b(take uturn|turn corner)\b"
}

# ==============================================================================
# 2. HÀM BÓC TÁCH & LÀM SẠCH VĂN BẢN
# ==============================================================================

def clean_original_text(text):
    """
    1. Chuyển chữ thường.
    2. Xóa mạo từ (a, an, the) ở đầu câu.
    3. Dùng spaCy để nhận diện và đưa tất cả ĐỘNG TỪ về nguyên thể (Lemmatization).
    """
    text = text.lower().strip()
    
    # Xóa mạo từ ở đầu câu
    text = re.sub(r'^(a|an|the)\s+', '', text)
    
    # Đưa câu qua mô hình spaCy để phân tích ngữ pháp
    doc = nlp(text)
    
    clean_tokens = []
    for token in doc:
        # Nếu từ đó là Động từ (VERB) hoặc Trợ động từ (AUX), lấy dạng nguyên thể (lemma_)
        if token.pos_ in ["VERB", "AUX"]:
            clean_tokens.append(token.lemma_)
        else:
            # Các từ loại khác (Danh từ, tính từ...) giữ nguyên text gốc
            clean_tokens.append(token.text)
            
    # Ghép các từ lại thành câu hoàn chỉnh
    clean_sentence = " ".join(clean_tokens)
    
    # Loại bỏ khoảng trắng thừa trước dấu câu (nếu có do spacy tách token)
    clean_sentence = re.sub(r'\s+([.,!?])', r'\1', clean_sentence)
    
    return clean_sentence

def get_first_match(text, mapping):
    best_match_label = ""
    best_match_raw = ""
    best_idx = float('inf')
    for label, pattern in mapping.items():
        for match in re.finditer(pattern, text):
            if match.start() < best_idx:
                best_idx = match.start()
                best_match_label = label
                best_match_raw = match.group(0)
    return best_match_label, best_match_raw

def get_type_match(text, mapping):
    matches = []
    for label, pattern in mapping.items():
        for m in re.finditer(pattern, text):
            matches.append((m.start(), m.end(), m.group(0), label))
            
    if not matches:
        return "", ""
        
    matches.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    
    filtered_matches = []
    for m in matches:
        if not filtered_matches:
            filtered_matches.append(m)
        else:
            last_m = filtered_matches[-1]
            if m[1] <= last_m[1]:
                continue 
            filtered_matches.append(m)
            
    matches = filtered_matches
    
    merged_start = matches[0][0]
    merged_end = matches[0][1]
    final_label = matches[0][3]
    
    for i in range(1, len(matches)):
        curr_start = matches[i][0]
        curr_end = matches[i][1]
        curr_label = matches[i][3]
        
        gap = text[merged_end:curr_start].strip() if curr_start > merged_end else ""
        gap_words = set(gap.split())
        
        if curr_start <= merged_end or (len(gap) < 20 and not gap_words.intersection({"and", "by", "with", "behind", "pass", "follow", "cross", "after"})):
            merged_end = max(merged_end, curr_end)
            if final_label == "vehicle" or (curr_label != "vehicle" and final_label == curr_label):
                final_label = curr_label
            elif curr_label != "vehicle":
                final_label = curr_label
        else:
            break 
            
    full_raw_phrase = text[merged_start:merged_end]
    return final_label, full_raw_phrase

def get_motion_match(text, mapping):
    matches = []
    for label, pattern in mapping.items():
        for m in re.finditer(pattern, text):
            matches.append((m.start(), m.group(0), label))
    if not matches:
        return "", ""
        
    matches.sort(key=lambda x: x[0])
    directional_labels = {"turn left", "turn right", "go straight"}
    for m in matches:
        if m[2] in directional_labels:
            return m[2], m[1] 
    return matches[0][2], matches[0][1]

def extract_and_normalize(clean_text):
    text_lower = clean_text.lower()
    
    size, raw_size = get_first_match(text_lower, SIZE_MAPPING)
    color, raw_color = get_first_match(text_lower, COLOR_MAPPING)
    v_type, raw_type = get_type_match(text_lower, TYPE_MAPPING) 
    motion, raw_motion = get_motion_match(text_lower, MOTION_MAPPING)

    remaining_text = text_lower
    for raw_match in [raw_size, raw_color, raw_type, raw_motion]:
        if raw_match: 
            remaining_text = remaining_text.replace(raw_match, "", 1)
            
    ctx_label_type, ctx_type_raw = get_type_match(remaining_text, TYPE_MAPPING) 

    if ctx_type_raw:
        type_idx = remaining_text.find(ctx_type_raw)
        
        def get_closest_modifier_before(mapping, text, target_idx, max_dist=30):
            best_raw = ""
            best_start = -1
            for lbl, pat in mapping.items():
                for m in re.finditer(pat, text):
                    if m.start() < target_idx and (target_idx - m.end()) < max_dist:
                        if m.start() > best_start:
                            best_start = m.start()
                            best_raw = m.group(0)
            return best_raw

        ctx_raw_size = get_closest_modifier_before(SIZE_MAPPING, remaining_text, type_idx)
        ctx_raw_color = get_closest_modifier_before(COLOR_MAPPING, remaining_text, type_idx)
        
        ctx_raw_parts = [p for p in [ctx_raw_size, ctx_raw_color, ctx_type_raw] if p]
        context_text = " ".join(ctx_raw_parts)
    else:
        context_text = remaining_text
        
    stop_words_extra = r"\b(be|is|are|was|were|a|an|the|that|which|this|make|make a|at|on|in|with|of|by)\b"
    context_text = re.sub(stop_words_extra, ' ', context_text)
    context_text = re.sub(r'[^\w\s]', ' ', context_text) 
    context_text = re.sub(r'\s+', ' ', context_text).strip()

    if not context_text:
        context_text = "unknown context"

    return color, v_type, motion, context_text

def safe_string(text, dummy_val="unknown"):
    return text if text.strip() else dummy_val

# ==============================================================================
# 3. CHƯƠNG TRÌNH CHÍNH
# ==============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Đang chạy trên {device}...")

    print("📥 Đang tải mô hình CLIP...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_encoder.eval()

    clean_json_files = [
        './data/json/test-queries-clean.json', 
        './data/json/train_clean.json'
    ] 
    unique_sentences = {}
    
    print(f"📖 Đang đọc dữ liệu từ các file JSON...")
    for clean_json_path in clean_json_files:
        if os.path.exists(clean_json_path):
            print(f" -> Đang quét file: {clean_json_path}")
            with open(clean_json_path, 'r', encoding='utf-8') as f:
                clean_data = json.load(f)
            
            for track_id, track_info in clean_data.items():
                all_queries = track_info.get('nl', []) + track_info.get('nl_other_views', [])
                for query in all_queries:
                    # Chuyển đổi từ Original -> Clean text (bao gồm đưa động từ về nguyên thể)
                    cleaned_query = clean_original_text(query)
                    if cleaned_query:
                        unique_sentences[cleaned_query] = query 
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy file tại đường dẫn: {clean_json_path}")

    text_to_emb = {}

    def encode_text(text, max_len=16):
        inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt").to(device)
        outputs = text_encoder(**inputs)
        return inputs.input_ids.squeeze(0).cpu(), outputs.last_hidden_state.squeeze(0).cpu()
    
    with torch.no_grad():
        for clean_text, original_text in tqdm(unique_sentences.items(), desc="Tiến trình Encode CLIP"):
            
            color, v_type, mot, ctx = extract_and_normalize(clean_text)
            
            if not any([color, v_type, mot, ctx]):
                continue
            
            p_color_safe = safe_string(color, "unknown color")
            p_type_safe = safe_string(v_type, "unknown vehicle")
            p_mot_safe = safe_string(mot, "unknown motion")
            p_ctx_safe = safe_string(ctx, "unknown context")
            
            p_combined_safe = f"{p_color_safe} {p_type_safe} {p_mot_safe}".strip()
            
            c_ids, c_emb = encode_text(p_color_safe, max_len=8)      
            t_ids, t_emb = encode_text(p_type_safe, max_len=8)     
            m_ids, m_emb = encode_text(p_mot_safe, max_len=16) 
            ctx_ids, ctx_emb = encode_text(p_ctx_safe, max_len=16) 
            
            comb_ids, comb_emb = encode_text(p_combined_safe, max_len=32)
            
            text_to_emb[clean_text] = {
                "original_text": original_text,
                "clean_text": clean_text,
                "color_text": p_color_safe,
                "color_embedding": c_emb, 
                "color_input_ids": c_ids, 
                "type_text": p_type_safe,
                "type_embedding": t_emb, 
                "type_input_ids": t_ids, 
                "motion_text": p_mot_safe,
                "motion_embedding": m_emb, 
                "motion_input_ids": m_ids, 
                "context_text": p_ctx_safe,
                "context_embedding": ctx_emb, 
                "context_input_ids": ctx_ids, 
                
                "text_embeds_text": p_combined_safe,
                "text_embeds": comb_emb,         
                "text_embeds_ids": comb_ids,     
            }

    save_path = './data/data/clip_text_tokens_extracted.pt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(text_to_emb, save_path)
    
    print(f"\n✅ Thành công! Đã xử lý gộp dữ liệu và lưu {len(text_to_emb)} mẫu vào: {save_path}")

if __name__ == "__main__":
    main()