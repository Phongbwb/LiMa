import torch

def inspect_pt_file(file_path, num_samples=100):
    """
    Load tệp .pt và in ra màn hình số lượng câu được chỉ định để kiểm tra.
    """
    print(f"\n{'='*95}")
    print(f"KIỂM TRA {num_samples} CÂU TỪ TỆP: {file_path}")
    print(f"{'='*95}")

    try:
        data = torch.load(file_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp '{file_path}'. Vui lòng kiểm tra lại đường dẫn.")
        return

    count = 0
    
    # Cấu trúc mới: data là một dictionary với key là "clean_text", value là thông tin record
    for clean_text, record in data.items():
        if count >= num_samples:
            print(f"\n[!] Đã in đủ {num_samples} mẫu từ {file_path}.\n")
            return

        count += 1
        
        # Lấy thông tin từ các key tương ứng trong dict mới
        orig = record.get('original_text', 'N/A')
        color = record.get('color_text', 'N/A')
        v_type = record.get('type_text', 'N/A')
        motion = record.get('motion_text', 'N/A')
        context = record.get('context_text', 'N/A')
        
        # In định dạng gọn gàng để dễ quét mắt (scan)
        print(f"[{count:03d}]")
        print(f"      Gốc    : {orig}")
        print(f"      Clean  : {clean_text}")
        print(f"      Tách   : Color: {color:<13} | Type: {v_type:<12} | Motion: {motion:<18} | Context: {context}")
        print("-" * 95)
        
    # Trường hợp file có ít hơn num_samples câu
    print(f"\n[!] Đã in toàn bộ {count} mẫu có trong {file_path} (Tệp này có ít hơn {num_samples} câu).\n")

if __name__ == "__main__":
    train_file = "./data/data/clip_text_tokens_extracted.pt"
    
    # Kiểm tra 100 câu
    inspect_pt_file(train_file, 100)