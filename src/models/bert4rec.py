import torch
from torch import nn


class BERT4Rec(nn.Module):
    def __init__(self, num_items, hidden_dim=64, max_history_length=50, num_layers=2, num_heads=2, dropout=0.2):
        super().__init__()
        self.num_items = num_items
        self.mask_token_id = num_items + 1
        self.hidden_dim = hidden_dim
        self.max_history_length = max_history_length
        self.item_embedding = nn.Embedding(num_items + 2, hidden_dim, padding_idx=0)
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
        self.output_bias = nn.Parameter(torch.zeros(num_items + 2))

    def encode(self, seq):
        batch_size, seq_len = seq.shape
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        x = self.item_embedding(seq) + self.position_embedding(positions)
        x = x.masked_fill(seq.eq(0).unsqueeze(-1), 0.0)
        x = self.dropout(x)
        out = self.encoder(x, src_key_padding_mask=seq.eq(0))
        out = self.layer_norm(out)
        out = out.masked_fill(seq.eq(0).unsqueeze(-1), 0.0)
        return out

    def logits(self, seq):
        states = self.encode(seq)
        return states @ self.item_embedding.weight.t() + self.output_bias

    def predict(self, seq, candidates):
        batch_size, seq_len = seq.shape
        history = seq[:, -(seq_len - 1) :]
        mask_col = torch.full((batch_size, 1), self.mask_token_id, dtype=torch.long, device=seq.device)
        masked_seq = torch.cat([history, mask_col], dim=1)
        states = self.encode(masked_seq)
        final_state = states[:, -1, :]
        cand_emb = self.item_embedding(candidates)
        return (final_state.unsqueeze(1) * cand_emb).sum(dim=-1) + self.output_bias[candidates]
