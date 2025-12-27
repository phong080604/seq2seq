def tokenize_sentence(text: str, tokenizer, num_steps: int = 32):
    """Sử dụng BPE Tokenizer để mã hóa câu."""
    # Bước tiền xử lý cơ bản (lower case)
    text = text.lower().strip()
    
    # 1. BPE Tokenizer tự động handle việc tách dấu câu và bẻ nhỏ subwords
    # encode() trả về một object chứa ids, tokens, attention_mask...
    encoded = tokenizer.encode(text)
    
    # 2. Lấy danh sách IDs và thêm <eos>
    # Lưu ý: Tokenizer đã có sẵn các token đặc biệt, ta lấy ID của chúng
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")
    
    ids = encoded.ids + [eos_id]

    # 3. Truncate (Cắt bớt nếu quá dài)
    if len(ids) > num_steps:
        ids = ids[:num_steps]
    
    valid_len = len(ids)

    # 4. Pad (Thêm pad_id nếu quá ngắn)
    if len(ids) < num_steps:
        ids += [pad_id] * (num_steps - len(ids))

    return ids, valid_len