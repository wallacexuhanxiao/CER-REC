import torch
from torch import nn


class SASRec(nn.Module):
    def __init__(self, num_items, hidden_dim=64, max_history_length=50, num_layers=2, num_heads=2, dropout=0.2):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_history_length = max_history_length
        self.num_heads = num_heads
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_history_length, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def encode(self, seq):
        batch_size, seq_len = seq.shape
        positions = seq.ne(0).long().cumsum(dim=1).sub(1).clamp_min(0)
        x = self.item_embedding(seq) + self.position_embedding(positions)
        x = x.masked_fill(seq.eq(0).unsqueeze(-1), 0.0)
        x = self.dropout(x)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=seq.device, dtype=torch.bool),
            diagonal=1,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = seq.eq(0).unsqueeze(1).expand(-1, seq_len, -1)
        attention_mask = causal_mask | key_padding_mask
        pad_queries = seq.eq(0)
        diagonal = torch.eye(seq_len, device=seq.device, dtype=torch.bool).unsqueeze(0)
        attention_mask = torch.where(pad_queries.unsqueeze(-1) & diagonal, False, attention_mask)
        attention_mask = attention_mask.repeat_interleave(self.num_heads, dim=0)
        out = self.encoder(x, mask=attention_mask)
        out = out.masked_fill(seq.eq(0).unsqueeze(-1), 0.0)
        out = self.layer_norm(out)
        out = out.masked_fill(seq.eq(0).unsqueeze(-1), 0.0)
        return out

    def forward(self, seq, pos_items, neg_items):
        states = self.encode(seq)
        final_state = states[:, -1, :]
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)
        pos_logits = (final_state * pos_emb).sum(dim=-1)
        neg_logits = (final_state.unsqueeze(1) * neg_emb).sum(dim=-1)
        return pos_logits, neg_logits

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
