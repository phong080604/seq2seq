"""Utilities for preprocessing and tokenizing sentences for the seq2seq model.

This module provides a minimal `tokenize_sentence` function expected by
`predictor.py`. It intentionally avoids calling `vocab[...]` (which may
delegate to d2l internals) when checking membership and instead uses
`vocab.token_to_idx` and `vocab.unk` which are present on the pickled
`d2l.torch.Vocab` objects used in this project.
"""

def _preprocess(text: str) -> str:
    """Normalize spacing and punctuation for a single sentence."""
    if text is None:
        return ""
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Insert space before punctuation if missing (so punctuation becomes separate tokens)
    def no_space(char, prev_char):
        return char in ',.!?' and prev_char != ' '

    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())]
    return ''.join(out)


def tokenize_sentence(text: str, vocab, num_steps: int = 20):
    """Tokenize a single sentence and convert to fixed-length index list.

    Returns (indices, valid_len) where `indices` is a list of length `num_steps`
    containing token indices (padded with `<pad>` index if necessary) and
    `valid_len` is the number of non-pad tokens.
    """
    text = _preprocess(text)
    # Ensure <eos> is appended and split on spaces
    tokens = [t for t in (text + ' <eos>').split(' ') if t]

    # Truncate if longer than allowed steps
    if len(tokens) > num_steps:
        tokens = tokens[:num_steps]

    valid_len = len(tokens)

    # Access underlying mapping and unk index directly to avoid triggering
    # Vocab's __getitem__ special behavior.
    token_to_idx = getattr(vocab, 'token_to_idx', None)
    unk_idx = getattr(vocab, 'unk', None)
    pad_idx = token_to_idx.get('<pad>') if token_to_idx is not None else 0

    indices = []
    for tok in tokens:
        if token_to_idx is not None and tok in token_to_idx:
            indices.append(token_to_idx[tok])
        else:
            # Fall back to vocab.unk if available, otherwise 0
            indices.append(unk_idx if unk_idx is not None else 0)

    # Pad up to num_steps
    if len(indices) < num_steps:
        indices += [pad_idx] * (num_steps - len(indices))

    return indices, valid_len
