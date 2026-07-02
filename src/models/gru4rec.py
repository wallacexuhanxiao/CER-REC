import torch
from torch import nn


class GRU4Rec(nn.Module):
    def __init__(self, num_items, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def encode(self, seq):
        x = self.dropout(self.item_embedding(seq))
        x = x.masked_fill(seq.eq(0).unsqueeze(-1), 0.0)
        out, _ = self.gru(x)
        out = self.layer_norm(out)
        out = out.masked_fill(seq.eq(0).unsqueeze(-1), 0.0)
        return out

    def sequence_logits(self, seq, pos_items, neg_items):
        states = self.encode(seq)
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)
        pos_logits = (states * pos_emb).sum(dim=-1)
        neg_logits = (states * neg_emb).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, seq, candidates):
        states = self.encode(seq)
        final_state = states[:, -1, :]
        cand_emb = self.item_embedding(candidates)
        return (final_state.unsqueeze(1) * cand_emb).sum(dim=-1)
