import torch
from torch import nn


class UserGate(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, 1))

    def forward(self, user_features, item_features=None):
        return torch.sigmoid(self.net(user_features)).squeeze(-1).unsqueeze(1)


class ItemGate(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, 1))

    def forward(self, user_features, item_features):
        return torch.sigmoid(self.net(item_features)).squeeze(-1)


class UserTargetGate(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, user_features, item_features):
        batch, candidates, _ = item_features.shape
        expanded_user = user_features.unsqueeze(1).expand(-1, candidates, -1)
        return torch.sigmoid(self.net(torch.cat([expanded_user, item_features], dim=-1))).squeeze(-1)

