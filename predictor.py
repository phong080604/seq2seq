import torch
from d2l import torch as d2l
from tokenizers import Tokenizer
# Đảm bảo bạn đã có file model_defs.py chứa TransformerDecoder trong cùng thư mục src
from src.model_defs import TransformerDecoder 

class Translator:
    def __init__(self, 
                 model_path='assets/seq2seq_model.pth', 
                 src_tokenizer_path='assets/src_tokenizer.json', 
                 tgt_tokenizer_path='assets/tgt_tokenizer.json'):
        
        self.device = d2l.try_gpu()
        print(f"Loading model on: {self.device}")

        # 1. Load BPE Tokenizers (Thay vì dùng pickle)
        self.src_vocab = Tokenizer.from_file(src_tokenizer_path)
        self.tgt_vocab = Tokenizer.from_file(tgt_tokenizer_path)

        # 2. Cấu hình Model (Sửa len() thành get_vocab_size())
        num_hiddens, num_blks, dropout = 256, 2, 0.3
        ffn_num_hiddens, num_heads = 1024, 4
        
        encoder = d2l.TransformerEncoder(
            self.src_vocab.get_vocab_size(), num_hiddens, ffn_num_hiddens, 
            num_heads, num_blks, dropout)
        
        decoder = TransformerDecoder(
            self.tgt_vocab.get_vocab_size(), num_hiddens, ffn_num_hiddens, 
            num_heads, num_blks, dropout)
        
        # Lấy ID của token <pad> bằng token_to_id()
        pad_id = self.tgt_vocab.token_to_id('<pad>')
        self.model = d2l.Seq2Seq(encoder, decoder, tgt_pad=pad_id, lr=0.0)
        
        # 3. Load Weights
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval() 

    def translate(self, input_text, num_steps=32):
        # Tiền xử lý văn bản
        input_text = input_text.lower()
        
        # Encode chuỗi bằng BPE Tokenizer
        # BPE tự động chia subwords, không cần split(' ') thủ công
        encoded = self.src_vocab.encode(input_text)
        ids = encoded.ids + [self.src_vocab.token_to_id("<eos>")]
        
        # Padding / Truncate
        if len(ids) < num_steps:
            ids += [self.src_vocab.token_to_id("<pad>")] * (num_steps - len(ids))
        else:
            ids = ids[:num_steps]
            
        enc_input = torch.tensor([ids], device=self.device) 
        # Tính valid length dựa trên Pad ID
        pad_id = self.src_vocab.token_to_id("<pad>")
        enc_valid_len = (enc_input != pad_id).type(torch.int32).sum(1)

        # Encoder forward
        enc_outputs = self.model.encoder(enc_input, enc_valid_len)
        dec_state = self.model.decoder.init_state(enc_outputs, enc_valid_len)
        
        # Decoder loop
        bos_id = self.tgt_vocab.token_to_id('<bos>')
        eos_id = self.tgt_vocab.token_to_id('<eos>')
        
        dec_input = torch.tensor([[bos_id]], device=self.device)
        output_tokens = []
        
        for _ in range(num_steps):
            Y, dec_state = self.model.decoder(dec_input, dec_state)
            
            # Lấy xác suất cao nhất cho token tiếp theo
            next_step_logits = Y[:, -1, :] 
            pred_idx = next_step_logits.argmax(dim=1).item()
            
            if pred_idx == eos_id:
                break
                
            # Chuyển ID thành chữ
            token = self.tgt_vocab.id_to_token(pred_idx)
            # Xử lý các token đặc biệt của BPE (ví dụ dấu '##' hoặc 'Ġ') nếu cần
            output_tokens.append(token)
            
            # Cập nhật dec_input cho bước tiếp theo (Transformer cần context cũ)
            next_id_tensor = torch.tensor([[pred_idx]], device=self.device)
            dec_input = torch.cat([dec_input, next_id_tensor], dim=1)
            
        # Nối các subwords lại và xóa ký tự đặc biệt của BPE (nếu có)
        full_translation = " ".join(output_tokens).replace('Ġ', ' ').replace(' ', ' ').strip()
        return full_translation