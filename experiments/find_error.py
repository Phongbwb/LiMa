import os
import cv2
from PIL import Image

# Đổi lại đường dẫn nếu cần
FRAMES_DIR = "./data/data/validation/S05/c016/img1"

def find_killer_image():
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(valid_extensions)]
    image_files.sort()

    print(f"Bắt đầu quét {len(image_files)} ảnh để tìm Segmentation Fault...")
    
    for f in image_files:
        path = os.path.join(FRAMES_DIR, f)
        
        # [QUAN TRỌNG] In tên file ra TRƯỚC, ép buffer xả ngay lập tức (flush=True)
        print(f"Đang kiểm tra: {f}...", end=" ", flush=True)
        
        # Thử thực hiện các thao tác giải mã C++
        img = cv2.imread(path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
        print("OK") # Nếu code chạy qua được dòng trên, in ra OK

if __name__ == "__main__":
    find_killer_image()