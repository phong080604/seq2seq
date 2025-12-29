import sacrebleu
from tqdm import tqdm
from predictor import Translator  # Import lớp Translator bạn đã đóng gói

def evaluate_model_metrics(test_file):
    # 1. Khởi tạo bộ dịch
    print(">>> Đang nạp mô hình Transformer...")
    translator = Translator(
        model_path='assets/seq2seq_model.pth',
        src_tokenizer_path='assets/src_tokenizer.json',
        tgt_tokenizer_path='assets/tgt_tokenizer.json'
    )

    sources = []
    references = []
    predictions = []

    # 2. Đọc file test_5000_pairs.txt
    print(f">>> Đang đọc dữ liệu từ {test_file}...")
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                sources.append(parts[0])
                references.append(parts[1])

    # 3. Tiến hành dịch (Inference)
    print(f">>> Đang dịch {len(sources)} câu...")
    for src in tqdm(sources):
        try:
            pred = translator.translate(src)
            predictions.append(pred)
        except:
            predictions.append("") # Trường hợp lỗi thì bỏ trống để không lệch index

    # 4. Tính toán BLEU Score (Sử dụng chuẩn sacrebleu)
    # sacrebleu yêu cầu reference là một list của list
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    
    # 5. Tính toán chr-F Score 
    chrf = sacrebleu.corpus_chrf(predictions, [references])

    # 6. Hiển thị kết quả so sánh với Benchmark
    print(f"{'BLEU Score':<20} | {bleu.score:>15.2f}")
    print(f"{'chr-F Score':<20} | {chrf.score/100:>15.3f}") # Chia 100 để về hệ số 0-1 giống ảnh

if __name__ == "__main__":
    evaluate_model_metrics("test_balanced_5000.txt")