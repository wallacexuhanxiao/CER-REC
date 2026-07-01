import torch

from src.models.sasrec import SASRec


def test_left_padding_forward_backward_without_nan():
    torch.manual_seed(2026)
    model = SASRec(
        num_items=20,
        hidden_dim=16,
        max_history_length=8,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
    )
    seq = torch.tensor(
        [
            [0, 0, 0, 0, 1, 2, 3, 4],
            [0, 0, 0, 5, 6, 7, 8, 9],
            [0, 0, 0, 0, 0, 0, 0, 10],
        ],
        dtype=torch.long,
    )
    pos = torch.tensor(
        [
            [0, 0, 0, 0, 2, 3, 4, 5],
            [0, 0, 0, 6, 7, 8, 9, 10],
            [0, 0, 0, 0, 0, 0, 0, 11],
        ],
        dtype=torch.long,
    )
    neg = torch.tensor(
        [
            [0, 0, 0, 0, 12, 13, 14, 15],
            [0, 0, 0, 11, 12, 13, 14, 15],
            [0, 0, 0, 0, 0, 0, 0, 12],
        ],
        dtype=torch.long,
    )

    states = model.encode(seq)
    assert torch.isfinite(states).all()
    assert torch.allclose(states[seq == 0], torch.zeros_like(states[seq == 0]))

    pos_logits, neg_logits = model.sequence_logits(seq, pos, neg)
    mask = pos.gt(0)
    assert torch.isfinite(pos_logits).all()
    assert torch.isfinite(neg_logits).all()

    loss = torch.nn.functional.binary_cross_entropy_with_logits(pos_logits[mask], torch.ones_like(pos_logits[mask]))
    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(neg_logits[mask], torch.zeros_like(neg_logits[mask]))
    loss.backward()

    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


def test_left_padding_amount_does_not_change_valid_outputs():
    torch.manual_seed(2026)
    padded_model = SASRec(
        num_items=20,
        hidden_dim=16,
        max_history_length=8,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
    )
    compact_model = SASRec(
        num_items=20,
        hidden_dim=16,
        max_history_length=4,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
    )
    compact_model.item_embedding.load_state_dict(padded_model.item_embedding.state_dict())
    compact_model.position_embedding.weight.data.copy_(padded_model.position_embedding.weight.data[:4])
    compact_model.encoder.load_state_dict(padded_model.encoder.state_dict())
    compact_model.layer_norm.load_state_dict(padded_model.layer_norm.state_dict())
    padded_model.eval()
    compact_model.eval()
    left_padded = torch.tensor([[0, 0, 0, 0, 1, 2, 3, 4]], dtype=torch.long)
    compact = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    with torch.no_grad():
        padded_states = padded_model.encode(left_padded)
        compact_states = compact_model.encode(compact)

    assert torch.isfinite(padded_states).all()
    assert torch.isfinite(compact_states).all()
    assert torch.allclose(padded_states[:, -4:, :], compact_states, atol=1e-6)
