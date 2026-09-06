import re
import matplotlib.pyplot as plt

# 1. Khai báo log_data (Hãy đảm bảo bạn dán chính xác log vào đây)
log_data = """
[DÁN LOG CỦA BẠN VÀO ĐÂY]
"""

# HOẶC BỎ COMMENT 2 DÒNG DƯỚI NẾU BẠN LƯU VÀO FILE TEXT RIÊNG:
with open('training_log.txt', 'r', encoding='utf-8') as f:
     log_data = f.read()

# Xóa các ký tự màu sắc/định dạng ẩn của terminal (ANSI codes) nếu có
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
log_data_clean = ansi_escape.sub('', log_data)

# 2. Khởi tạo mảng lưu trữ
epochs = []
train_losses = []
val_losses = []

# 3. Regex linh hoạt hơn: bỏ qua ký tự thừa ở giữa Train Loss và Val Loss
pattern = r"Train Loss:\s*([\d\.]+).*?Val Loss:\s*([\d\.]+)"
matches = re.findall(pattern, log_data_clean, re.DOTALL)

for i, match in enumerate(matches, 1):
    epochs.append(i)
    train_losses.append(float(match[0]))
    val_losses.append(float(match[1]))

# 4. Kiểm tra dữ liệu trước khi xử lý để tránh lỗi 'empty sequence'
if not val_losses:
    print(" LỖI: Không trích xuất được dữ liệu Loss.")
    print(" Hãy kiểm tra lại file log hoặc biến log_data xem đã chứa nội dung chưa.")
else:
    print(f" Đã trích xuất thành công {len(epochs)} epochs!")
    
    # 5. Vẽ biểu đồ
    plt.figure(figsize=(12, 6))

    plt.plot(epochs, train_losses, label='Train Loss', color='#1f77b4', linewidth=1.5)
    plt.plot(epochs, val_losses, label='Validation Loss', color='#ff7f0e', linewidth=1.5)


    # Định dạng
    plt.title('Li-Ma VLM: Training & Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Lưu và hiển thị
    plt.savefig('limavlm_loss_chart.png', dpi=300)
    plt.show()