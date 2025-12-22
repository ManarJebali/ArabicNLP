import math
import random
from collections import Counter
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re

from src.preprocessing.tokenizer import tokenize_arabic


# =========================================
# Basic SGNS for your tokenize() outputs
# - final_list_train: List[List[int]] of 1-based token ids
# - onehot_dict: Dict[word -> id] with ids in [1..V]
# Adds robust validation/coercion for sequences to avoid TypeError.
# =========================================

def _coerce_int_token(tok) -> Optional[int]:
    # Accept Python/NumPy ints
    if isinstance(tok, (int, np.integer)):
        return int(tok)
    # Accept numeric strings like "123"
    if isinstance(tok, str):
        tok = tok.strip()
        if tok.isdigit():
            return int(tok)
        return None
    return None


def _coerce_and_validate_id_seqs(
    seqs_raw,
    vocab_size: int,
    max_report: int = 3,
) -> List[List[int]]:
    """
    Ensure we have List[List[int]] with ids in [1..vocab_size].
    - If a sequence is a list/array/tuple: keep only int-like tokens.json in range.
    - If a sequence is a single int-like token: wrap as a length-1 list.
    - If a sequence is a string of digits (e.g., '1 2 3' or '1,2,3'): parse digits.
    - Otherwise: raise with a helpful message.
    """
    cleaned: List[List[int]] = []
    bad_examples = []
    digit_re = re.compile(r"\d+")

    for s in seqs_raw:
        seq_ids: List[int] = []

        if isinstance(s, (list, tuple, np.ndarray)):
            for t in s:
                v = _coerce_int_token(t)
                if v is not None and 1 <= v <= vocab_size:
                    seq_ids.append(v)

        elif isinstance(s, (int, np.integer)):
            v = int(s)
            if 1 <= v <= vocab_size:
                seq_ids = [v]

        elif isinstance(s, str):
            # Try to parse numeric IDs from string
            nums = [int(m) for m in digit_re.findall(s)]
            seq_ids = [v for v in nums if 1 <= v <= vocab_size]

        else:
            bad_examples.append((type(s).__name__, s))
            continue

        if len(seq_ids) >= 2:
            cleaned.append(seq_ids)
        elif seq_ids:
            # Keep singletons? Usually not useful for skip-gram; drop to avoid empty batches.
            # You can keep them by changing this branch to cleaned.append(seq_ids).
            pass
        else:
            bad_examples.append((type(s).__name__, s))

    if not cleaned:
        sample = bad_examples[:max_report]
        raise TypeError(
            "final_list_train must be a list of integer-ID sequences (ids in [1..V]). "
            f"No valid sequences could be formed. Examples of invalid entries: {sample}"
        )

    if bad_examples:
        print(f"[warn] Skipped {len(bad_examples)} invalid/empty sequences while coercing to id lists.")

    return cleaned


def _build_token_freq(seqs: List[List[int]], vocab_size: int) -> np.ndarray:
    """Count token frequencies from 1-based index sequences."""
    freqs = np.zeros(vocab_size + 1, dtype=np.int64)  # +1 so index 0 remains unused
    for s in seqs:
        for t in s:
            if 1 <= t <= vocab_size:
                freqs[t] += 1
    return freqs


def _generate_skipgram_pairs(
    seqs: List[List[int]],
    window_size: int,
) -> List[Tuple[int, int]]:
    """Produce (center, context) positive pairs from 1-based token id sequences."""
    pairs: List[Tuple[int, int]] = []
    for seq in seqs:
        L = len(seq)
        if L < 2:
            continue
        for i, center in enumerate(seq):
            if center <= 0:
                continue
            left = max(0, i - window_size)
            right = min(L, i + window_size + 1)
            for j in range(left, right):
                if j == i:
                    continue
                ctx = seq[j]
                if ctx <= 0:
                    continue
                pairs.append((center, ctx))
    return pairs


class SkipGramPairsDS(Dataset):
    def __init__(self, pairs: List[Tuple[int, int]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        c, ctx = self.pairs[idx]
        return torch.tensor(c, dtype=torch.long), torch.tensor(ctx, dtype=torch.long)


def _unigram_noise(freqs: np.ndarray, power: float = 0.75) -> torch.Tensor:
    """
    Build noise distribution Pn(w) ∝ f(w)^power for indices [0..V], with prob[0]=0 (unused).
    """
    probs = freqs.astype(np.float64) ** power
    probs[0] = 0.0  # never sample index 0
    Z = probs.sum()
    if Z == 0:
        # fallback to uniform over 1..V
        probs[:] = 0.0
        probs[1:] = 1.0
        Z = probs.sum()
    probs /= Z
    return torch.tensor(probs, dtype=torch.float32)


class SGNS(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        # +1 so row 0 stays unused; your ids are 1..vocab_size
        self.in_embed = nn.Embedding(vocab_size + 1, embed_dim)
        self.out_embed = nn.Embedding(vocab_size + 1, embed_dim)
        self.embed_dim = embed_dim
        self._init_weights()

    def _init_weights(self):
        bound = 0.5 / self.embed_dim
        nn.init.uniform_(self.in_embed.weight, -bound, bound)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center: torch.Tensor, pos_ctx: torch.Tensor, neg_ctx: torch.Tensor) -> torch.Tensor:
        """
        center: [B] longs in [1..V]
        pos_ctx: [B] longs in [1..V]
        neg_ctx: [B, K] longs in [1..V]
        """
        v = self.in_embed(center)        # [B, D]
        u_pos = self.out_embed(pos_ctx)  # [B, D]
        u_neg = self.out_embed(neg_ctx)  # [B, K, D]

        # Positive term: log sigma(v · u_pos)
        pos_score = (v * u_pos).sum(dim=1)                  # [B]
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-12)

        # Negative term: sum log sigma(-v · u_neg)
        neg_score = torch.bmm(u_neg.neg(), v.unsqueeze(2)).squeeze(2)  # [B, K]
        neg_loss = torch.log(torch.sigmoid(neg_score) + 1e-12).sum(dim=1)

        loss = -(pos_loss + neg_loss).mean()
        return loss


@torch.no_grad()
def extract_embedding_matrix(model: SGNS) -> np.ndarray:
    """
    Return the input embedding matrix including the unused row 0.
    Shape: [vocab_size+1, embed_dim]; your words are at rows 1..V.
    """
    return model.in_embed.weight.detach().cpu().numpy()


def train_sgns_for_tokenize_outputs(
    final_list_train,  # allow any; we coerce inside
    onehot_dict: Dict[str, int],
    embed_dim: int = 100,
    window_size: int = 2,
    num_negatives: int = 5,
    min_count: int = 1,            # your tokenize already prunes; keep 1
    batch_size: int = 4096,
    epochs: int = 2,
    lr: float = 0.01,
    device: Optional[str] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, int]:
    """
    Train Word2Vec (Skip-gram with Negative Sampling) directly on your tokenize() outputs.

    Returns:
      - embedding_matrix: np.ndarray of shape [V+1, embed_dim] (row 0 unused)
      - vocab_size: V (so you can set nn.Embedding(V+1, embed_dim, padding_idx=0))
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    vocab_size = max(onehot_dict.values())  # ids are 1..V

    # Coerce and validate sequences → List[List[int]] with ids in [1..V]
    final_list_train = _coerce_and_validate_id_seqs(final_list_train, vocab_size)

    # Build frequencies
    freqs = _build_token_freq(final_list_train, vocab_size)
    if min_count > 1:
        mask = (np.arange(freqs.size) > 0) & (freqs < min_count)
        freqs[mask] = 0

    # Positive pairs
    pairs = _generate_skipgram_pairs(final_list_train, window_size=window_size)
    if len(pairs) == 0:
        raise ValueError("No skip-gram pairs generated after coercion. Check your inputs.")

    ds = SkipGramPairsDS(pairs)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    # Negative sampling distribution
    noise = _unigram_noise(freqs, power=0.75)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    noise = noise.to(device)

    # Model + optimizer
    model = SGNS(vocab_size=vocab_size, embed_dim=embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    model.train()
    for ep in range(1, epochs + 1):
        total, steps = 0.0, 0
        for center, pos_ctx in dl:
            center = center.to(device)
            pos_ctx = pos_ctx.to(device)

            with torch.no_grad():
                neg = torch.multinomial(noise, num_samples=num_negatives * center.size(0), replacement=True)
                neg = neg.view(center.size(0), num_negatives)  # [B, K]

            opt.zero_grad(set_to_none=True)
            loss = model(center, pos_ctx, neg)
            loss.backward()
            opt.step()

            total += loss.item()
            steps += 1
        print(f"Epoch {ep}/{epochs} - avg loss: {total / max(1, steps):.4f}")

    emb = extract_embedding_matrix(model)  # [V+1, D], row 0 unused
    return emb, vocab_size
