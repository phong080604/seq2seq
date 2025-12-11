# predictor.py
import torch
import pickle
from d2l import torch as d2l
from src.model_defs import Seq2Seq, Seq2SeqEncoder, Seq2SeqDecoder
from src.text_utils import tokenize_sentence

class Translator:
    def __init__(self, 
                 model_path='assets/seq2seq_model.pth', 
                 src_vocab_path='assets/src_vocab.pkl', 
                 tgt_vocab_path='assets/tgt_vocab.pkl'):
        
        self.device = d2l.try_gpu()
        print(f"Loading model on: {self.device}")

        # 1. Load Vocabularies
        with open(src_vocab_path, 'rb') as f:
            self.src_vocab = pickle.load(f)
        with open(tgt_vocab_path, 'rb') as f:
            self.tgt_vocab = pickle.load(f)

        # 2. Cấu hình Model (Hyperparams phải KHỚP với notebook lúc train)
        # Trong notebook của bạn: embed=512, hidden=512, layers=3
        embed_size, num_hiddens, num_layers, dropout = 512, 512, 3, 0.5
        
        encoder = Seq2SeqEncoder(len(self.src_vocab), embed_size, num_hiddens, num_layers, dropout)
        decoder = Seq2SeqDecoder(len(self.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
        
        # tgt_pad lấy từ vocab
        self.model = Seq2Seq(encoder, decoder, tgt_pad=self.tgt_vocab['<pad>'], lr=0.0)
        
        # 3. Load Weights
        # Lưu ý: LazyLinear cần chạy 1 lần forward dummy để khởi tạo shape trước khi load weight
        # Tuy nhiên, load_state_dict thường xử lý được. Nếu lỗi size mismatch, ta sẽ fix sau.
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() # Quan trọng: tắt Dropout

    def translate(self, input_text, num_steps=20):
        """
        Dịch một câu tiếng Anh sang tiếng Việt
        """
        # 1. Tokenize & Convert to Tensor
        src_indices, valid_len_val = tokenize_sentence(input_text, self.src_vocab, num_steps)
        
        # Tạo batch dimension (Batch size = 1)
        src_tensor = torch.tensor([src_indices], dtype=torch.long, device=self.device)
        src_valid_len = torch.tensor([valid_len_val], device=self.device)

        # 2. Encoder Forward
        enc_outputs = self.model.encoder(src_tensor, src_valid_len)
        dec_state = self.model.decoder.init_state(enc_outputs, src_valid_len)
        
        # 3. Decoder Loop (Greedy Search)
        # Bắt đầu với <bos>
        dec_input = torch.tensor([self.tgt_vocab['<bos>']], device=self.device).unsqueeze(0)
        
        output_tokens = []
        
        for _ in range(num_steps):
            # Forward decoder một bước
            Y, dec_state = self.model.decoder(dec_input, dec_state)
            
            # Lấy token có xác suất cao nhất (Argmax)
            dec_input = Y.argmax(dim=2)
            pred_idx = dec_input.squeeze().item()
            
            # Kiểm tra <eos>
            if pred_idx == self.tgt_vocab['<eos>']:
                break
                
            # Convert index -> string token
            token = self.tgt_vocab.to_tokens(pred_idx)
            output_tokens.append(token)
            
        return " ".join(output_tokens)