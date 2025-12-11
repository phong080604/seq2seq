# app.py
import gradio as gr
import sys
import os

# Thêm print để biết script đã bắt đầu chạy
print(">>> BẮT ĐẦU CHẠY APP...")

try:
    from predictor import Translator
    print(">>> Đã import Translator thành công.")
except Exception as e:
    print(f"!!! LỖI IMPORT: {e}")
    sys.exit(1)

# Kiểm tra xem file model có tồn tại không
if not os.path.exists('assets/seq2seq_model.pth'):
    print("!!! LỖI: Không tìm thấy file 'assets/seq2seq_model.pth'")
    print("!!! Bạn đã chạy lệnh lưu model trong Notebook chưa?")
    sys.exit(1)

print(">>> Đang khởi tạo Model (Vui lòng đợi 10-20 giây)...")

try:
    # Khởi tạo Translator
    translator = Translator()
    print(">>> ĐÃ LOAD MODEL THÀNH CÔNG!")
except Exception as e:
    print(f"!!! LỖI KHI LOAD MODEL: {e}")
    sys.exit(1)

def translate_wrapper(text):
    if not text or not text.strip():
        return ""
    try:
        return translator.translate(text)
    except Exception as e:
        return f"Lỗi: {str(e)}"

# Giao diện
with gr.Blocks(title="seq2seq") as demo:
    gr.Markdown("# Dịch Máy Anh - Việt")
    
    with gr.Row():
        input_text = gr.Textbox(label="Tiếng Anh", placeholder="Hello")
        output_text = gr.Textbox(label="Tiếng Việt")
    
    btn = gr.Button("Dịch")
    btn.click(fn=translate_wrapper, inputs=input_text, outputs=output_text)

print(">>> Đang khởi chạy server Gradio...")

if __name__ == "__main__":
    # Bỏ server_name và share để chạy local ổn định nhất
    demo.launch()