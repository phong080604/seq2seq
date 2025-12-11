import collections
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# class Encoder(nn.Module):
#     """The base encoder interface for the encoder--decoder architecture.

#     Defined in :numref:`sec_encoder-decoder`"""
#     def __init__(self):
#         super().__init__()

#     # Later there can be additional arguments (e.g., length excluding padding)
#     def forward(self, X, *args):
#         raise NotImplementedError


# class Decoder(nn.Module):
#     """The base decoder interface for the encoder--decoder architecture.

#     Defined in :numref:`sec_encoder-decoder`"""
#     def __init__(self):
#         super().__init__()

#     # Later there can be additional arguments (e.g., length excluding padding)
#     def init_state(self, enc_all_outputs, *args):
#         raise NotImplementedError


#     def forward(self, X, state):
#         raise NotImplementedError


# class EncoderDecoder(d2l.Classifier):
#     """The base class for the encoder--decoder architecture.

#     Defined in :numref:`sec_encoder-decoder`"""
#     def __init__(self, encoder, decoder):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#     def forward(self, enc_X, dec_X, *args):
#         enc_all_outputs = self.encoder(enc_X, *args)
#         dec_state = self.decoder.init_state(enc_all_outputs, *args)
#         # Return decoder output only
#         return self.decoder(dec_X, dec_state)[0]


#     def predict_step(self, batch, device, num_steps,
#                      save_attention_weights=False):
#         """Defined in :numref:`sec_seq2seq_training`"""
#         batch = [d2l.to(a, device) for a in batch]
#         src, tgt, src_valid_len, _ = batch
#         enc_all_outputs = self.encoder(src, src_valid_len)
#         dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
#         outputs, attention_weights = [d2l.expand_dims(tgt[:, 0], 1), ], []
#         for _ in range(num_steps):
#             Y, dec_state = self.decoder(outputs[-1], dec_state)
#             outputs.append(d2l.argmax(Y, 2))
#             # Save attention weights (to be covered later)
#             if save_attention_weights:
#                 attention_weights.append(self.decoder.attention_weights)
#         return d2l.concat(outputs[1:], 1), attention_weights


# def init_seq2seq(module):
#     """Initialize weights for sequence-to-sequence learning.

#     Defined in :numref:`sec_seq2seq`"""
#     if type(module) == nn.Linear:
#          nn.init.xavier_uniform_(module.weight)
#     if type(module) == nn.GRU:
#         for param in module._flat_weights_names:
#             if "weight" in param:
#                 nn.init.xavier_uniform_(module._parameters[param])


# class Seq2SeqEncoder(d2l.Encoder):
#     """The RNN encoder for sequence-to-sequence learning.

#     Defined in :numref:`sec_seq2seq`"""
#     def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
#                  dropout=0):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.rnn = d2l.GRU(embed_size, num_hiddens, num_layers, dropout)
#         self.apply(init_seq2seq)

#     def forward(self, X, *args):
#         # X shape: (batch_size, num_steps)
#         embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int64))
#         # embs shape: (num_steps, batch_size, embed_size)
#         outputs, state = self.rnn(embs)
#         # outputs shape: (num_steps, batch_size, num_hiddens)
#         # state shape: (num_layers, batch_size, num_hiddens)
#         return outputs, state

# class Seq2SeqDecoder(d2l.Decoder):
#     """The RNN decoder for sequence to sequence learning."""
#     def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
#                  dropout=0):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.rnn = d2l.GRU(embed_size+num_hiddens, num_hiddens,
#                            num_layers, dropout)
#         self.dense = nn.LazyLinear(vocab_size)
#         self.apply(init_seq2seq)

#     def init_state(self, enc_all_outputs, *args):
#         return enc_all_outputs

#     def forward(self, X, state):
#         # X shape: (batch_size, num_steps)
#         # embs shape: (num_steps, batch_size, embed_size)
#         embs = self.embedding(X.t().type(torch.int32))
#         enc_output, hidden_state = state
#         # context shape: (batch_size, num_hiddens)
#         context = enc_output[-1]
#         # Broadcast context to (num_steps, batch_size, num_hiddens)
#         context = context.repeat(embs.shape[0], 1, 1)
#         # Concat at the feature dimension
#         embs_and_context = torch.cat((embs, context), -1)
#         outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
#         outputs = self.dense(outputs).swapaxes(0, 1)
#         # outputs shape: (batch_size, num_steps, vocab_size)
#         # hidden_state shape: (num_layers, batch_size, num_hiddens)
#         return outputs, [enc_output, hidden_state]

# class Seq2Seq(d2l.EncoderDecoder):
#     """The RNN encoder--decoder for sequence to sequence learning.

#     Defined in :numref:`sec_seq2seq_decoder`"""
#     def __init__(self, encoder, decoder, tgt_pad, lr):
#         super().__init__(encoder, decoder)
#         self.save_hyperparameters()

#     def validation_step(self, batch):
#         Y_hat = self(*batch[:-1])
#         self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)


#     def configure_optimizers(self):
#         # Adam optimizer is used here
#         return torch.optim.Adam(self.parameters(), lr=self.lr)


# def bleu(pred_seq, label_seq, k):
#     """Compute the BLEU.

#     Defined in :numref:`sec_seq2seq_training`"""
#     pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
#     len_pred, len_label = len(pred_tokens), len(label_tokens)
#     score = math.exp(min(0, 1 - len_label / len_pred))
#     for n in range(1, min(k, len_pred) + 1):
#         num_matches, label_subs = 0, collections.defaultdict(int)
#         for i in range(len_label - n + 1):
#             label_subs[' '.join(label_tokens[i: i + n])] += 1
#         for i in range(len_pred - n + 1):
#             if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
#                 num_matches += 1
#                 label_subs[' '.join(pred_tokens[i: i + n])] -= 1
#         score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
#     return score


# def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
#                   cmap='Reds'):
#     """Show heatmaps of matrices.

#     Defined in :numref:`sec_queries-keys-values`"""
#     d2l.use_svg_display()
#     num_rows, num_cols, _, _ = matrices.shape
#     fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
#                                  sharex=True, sharey=True, squeeze=False)
#     for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
#         for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
#             pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
#             if i == num_rows - 1:
#                 ax.set_xlabel(xlabel)
#             if j == 0:
#                 ax.set_ylabel(ylabel)
#             if titles:
#                 ax.set_title(titles[j])
#     fig.colorbar(pcm, ax=axes, shrink=0.6);


# def masked_softmax(X, valid_lens):
#     """Perform softmax operation by masking elements on the last axis.

#     Defined in :numref:`sec_attention-scoring-functions`"""
#     # X: 3D tensor, valid_lens: 1D or 2D tensor
#     def _sequence_mask(X, valid_len, value=0):
#         maxlen = X.size(1)
#         mask = torch.arange((maxlen), dtype=torch.float32,
#                             device=X.device)[None, :] < valid_len[:, None]
#         X[~mask] = value
#         return X

#     if valid_lens is None:
#         return nn.functional.softmax(X, dim=-1)
#     else:
#         shape = X.shape
#         if valid_lens.dim() == 1:
#             valid_lens = torch.repeat_interleave(valid_lens, shape[1])
#         else:
#             valid_lens = valid_lens.reshape(-1)
#         # On the last axis, replace masked elements with a very large negative
#         # value, whose exponentiation outputs 0
#         X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
#         return nn.functional.softmax(X.reshape(shape), dim=-1)
