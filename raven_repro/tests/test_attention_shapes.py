import math

import pytest

torch = pytest.importorskip("torch")

from raven.attention import ViewGuidedAttnProcessor


class DummyAttention:
    def __init__(self, dim=8, heads=2):
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = torch.nn.Linear(dim, dim, bias=False)
        self.to_k = torch.nn.Linear(dim, dim, bias=False)
        self.to_v = torch.nn.Linear(dim, dim, bias=False)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(dim, dim, bias=False), torch.nn.Dropout(0.0)])
        self.norm_cross = False
        self.residual_connection = False
        self.rescale_output_factor = 1.0
        self.spatial_norm = None
        self.group_norm = None

    def prepare_attention_mask(self, attention_mask, sequence_length, batch_size):
        return attention_mask

    def head_to_batch_dim(self, tensor):
        batch, tokens, dim = tensor.shape
        tensor = tensor.view(batch, tokens, self.heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3).reshape(batch * self.heads, tokens, self.head_dim)

    def batch_to_head_dim(self, tensor):
        batch_heads, tokens, head_dim = tensor.shape
        batch = batch_heads // self.heads
        tensor = tensor.view(batch, self.heads, tokens, head_dim)
        return tensor.permute(0, 2, 1, 3).reshape(batch, tokens, self.heads * head_dim)

    def get_attention_scores(self, query, key, attention_mask):
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(query.shape[-1])
        if attention_mask is not None:
            scores = scores + attention_mask
        return torch.softmax(scores, dim=-1)


def test_view_guided_attention_preserves_shape_for_cfg_batch():
    processor = ViewGuidedAttnProcessor()
    attn = DummyAttention()
    hidden_states = torch.randn(4, 16, 8)
    output = processor(attn, hidden_states)
    assert output.shape == hidden_states.shape
    assert processor.state.last_is_cross_attention is False


def test_cross_attention_path_preserves_shape():
    processor = ViewGuidedAttnProcessor()
    attn = DummyAttention()
    hidden_states = torch.randn(4, 16, 8)
    encoder_hidden_states = torch.randn(4, 5, 8)
    output = processor(attn, hidden_states, encoder_hidden_states=encoder_hidden_states)
    assert output.shape == hidden_states.shape
    assert processor.state.last_is_cross_attention is True


def test_cfg_pairs_use_reference_key_value_sources():
    processor = ViewGuidedAttnProcessor(debug=True)
    attn = DummyAttention()
    hidden_states = torch.zeros(4, 2, 8)
    hidden_states[0] = 1.0  # uncond reference
    hidden_states[1] = 2.0  # uncond view
    hidden_states[2] = 3.0  # cond reference
    hidden_states[3] = 4.0  # cond view
    processor(attn, hidden_states)
    key_checksums = processor.state.last_key_source_checksums
    value_checksums = processor.state.last_value_source_checksums
    assert key_checksums[0] == pytest.approx(key_checksums[1])
    assert key_checksums[2] == pytest.approx(key_checksums[3])
    assert value_checksums[0] == pytest.approx(value_checksums[1])
    assert value_checksums[2] == pytest.approx(value_checksums[3])
