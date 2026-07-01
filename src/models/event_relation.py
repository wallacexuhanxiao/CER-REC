import torch
from torch import nn


class EventRelation(nn.Module):
    def __init__(self, hidden_dim=64, relation_hidden_dim=64, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, relation_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(relation_hidden_dim, relation_hidden_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Linear(relation_hidden_dim, 1)

    def forward(self, history_states, candidate_embeddings):
        hist = history_states.unsqueeze(1)
        cand = candidate_embeddings.unsqueeze(2)
        features = torch.cat(
            [
                hist.expand(-1, cand.shape[1], -1, -1),
                cand.expand(-1, -1, hist.shape[2], -1),
                hist * cand,
                hist - cand,
            ],
            dim=-1,
        )
        z = self.mlp(features)
        r = self.scorer(z).squeeze(-1)
        return z, r

