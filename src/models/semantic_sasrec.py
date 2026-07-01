import numpy as np
import torch
from torch import nn


class SemanticSASRec(nn.Module):
    def __init__(
        self,
        semantic_embedding_path,
        hidden_dim=64,
        projection_hidden_dim=256,
        max_history_length=50,
        num_layers=2,
        num_heads=2,
        dropout=0.2,
    ):
        super().__init__()
        embeddings = np.load(semantic_embedding_path).astype(np.float32)
        self.semantic_input_dim = embeddings.shape[1]
        self.hidden_dim = hidden_dim
        self.max_history_length = max_history_length
        self.num_heads = num_heads
        self.register_buffer("semantic_embedding_table", torch.from_numpy(embeddings), persistent=False)
        self.projector = nn.Sequential(
            nn.LayerNorm(self.semantic_input_dim),
            nn.Linear(self.semantic_input_dim, projection_hidden_dim),
            nn.GELU(),
            nn.Linear(projection_hidden_dim, hidden_dim),
        )
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

    def project_items(self, item_ids):
        raw = self.semantic_embedding_table[item_ids]
        projected = self.projector(raw)
        return projected.masked_fill(item_ids.eq(0).unsqueeze(-1), 0.0)

    def _attention_mask(self, seq):
        batch_size, seq_len = seq.shape
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=seq.device, dtype=torch.bool),
            diagonal=1,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = seq.eq(0).unsqueeze(1).expand(-1, seq_len, -1)
        attention_mask = causal_mask | key_padding_mask
        pad_queries = seq.eq(0)
        diagonal = torch.eye(seq_len, device=seq.device, dtype=torch.bool).unsqueeze(0)
        attention_mask = torch.where(pad_queries.unsqueeze(-1) & diagonal, False, attention_mask)
        return attention_mask.repeat_interleave(self.num_heads, dim=0)

    def encode(self, seq):
        positions = seq.ne(0).long().cumsum(dim=1).sub(1).clamp_min(0)
        x = self.project_items(seq) + self.position_embedding(positions)
        x = x.masked_fill(seq.eq(0).unsqueeze(-1), 0.0)
        x = self.dropout(x)
        out = self.encoder(x, mask=self._attention_mask(seq))
        out = out.masked_fill(seq.eq(0).unsqueeze(-1), 0.0)
        out = self.layer_norm(out)
        return out.masked_fill(seq.eq(0).unsqueeze(-1), 0.0)

    def sequence_logits(self, seq, pos_items, neg_items):
        states = self.encode(seq)
        pos_emb = self.project_items(pos_items)
        neg_emb = self.project_items(neg_items)
        pos_logits = (states * pos_emb).sum(dim=-1)
        neg_logits = (states * neg_emb).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, seq, candidates):
        states = self.encode(seq)
        final_state = states[:, -1, :]
        cand_emb = self.project_items(candidates)
        return (final_state.unsqueeze(1) * cand_emb).sum(dim=-1)

