"""View-guided correspondence attention for Diffusers UNets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AttentionDebugState:
    enabled: bool = False
    calls: int = 0
    last_shape: Optional[tuple[int, ...]] = None
    last_is_cross_attention: Optional[bool] = None
    last_batch_size: Optional[int] = None
    last_query_checksums: Optional[list[float]] = None
    last_key_source_checksums: Optional[list[float]] = None
    last_value_source_checksums: Optional[list[float]] = None


class ViewGuidedAttnProcessor:
    """Self-attention processor where view tokens attend to reference tokens.

    The UNet receives latents ordered as [reference, view]. With classifier-free
    guidance the effective order is [uncond reference, uncond view, cond reference,
    cond view]. Cross-attention is passed through unchanged.
    """

    def __init__(self, stream_batch_size: int = 1, debug: bool = False):
        self.stream_batch_size = stream_batch_size
        self.debug = debug
        self.state = AttentionDebugState(enabled=True)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("ViewGuidedAttnProcessor requires torch") from exc

        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape
        is_cross_attention = encoder_hidden_states is not None
        self.state.calls += 1
        self.state.last_shape = tuple(hidden_states.shape)
        self.state.last_is_cross_attention = is_cross_attention
        self.state.last_batch_size = batch_size

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            source_states = hidden_states
        else:
            source_states = encoder_hidden_states
            if getattr(attn, "norm_cross", False):
                source_states = attn.norm_encoder_hidden_states(source_states)

        key = attn.to_k(source_states)
        value = attn.to_v(source_states)

        if self.debug:
            self.state.last_query_checksums = query.detach().float().sum(dim=(1, 2)).cpu().tolist()

        if not is_cross_attention:
            if batch_size % 2 != 0:
                raise ValueError(
                    "View-guided self-attention expects paired reference/view batches; "
                    f"got batch size {batch_size}"
                )
            # Pairs are adjacent: 0=reference, 1=view, repeated for CFG halves.
            pair_count = batch_size // 2
            key_pairs = key.view(pair_count, 2, sequence_length, -1)
            value_pairs = value.view(pair_count, 2, sequence_length, -1)
            ref_key = key_pairs[:, 0:1].expand(-1, 2, -1, -1).reshape_as(key)
            ref_value = value_pairs[:, 0:1].expand(-1, 2, -1, -1).reshape_as(value)
            key = ref_key
            value = ref_value

        if self.debug:
            self.state.last_key_source_checksums = key.detach().float().sum(dim=(1, 2)).cpu().tolist()
            self.state.last_value_source_checksums = value.detach().float().sum(dim=(1, 2)).cpu().tolist()

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        rescale = getattr(attn, "rescale_output_factor", 1.0)
        return hidden_states / rescale


def make_view_guided_processors(unet, debug: bool = False) -> Dict[str, Any]:
    """Create processors only for self-attention layers, preserving cross-attn behavior."""
    try:
        from diffusers.models.attention_processor import AttnProcessor
    except ImportError as exc:
        raise ImportError("make_view_guided_processors requires diffusers") from exc

    processors: Dict[str, Any] = {}
    for name in unet.attn_processors.keys():
        if name.endswith("attn1.processor"):
            processors[name] = ViewGuidedAttnProcessor(debug=debug)
        else:
            processors[name] = AttnProcessor()
    return processors


def install_view_guided_attention(unet, debug: bool = False) -> Dict[str, Any]:
    processors = make_view_guided_processors(unet, debug=debug)
    unet.set_attn_processor(processors)
    return processors


def restore_default_attention(unet) -> None:
    try:
        from diffusers.models.attention_processor import AttnProcessor
    except ImportError as exc:
        raise ImportError("restore_default_attention requires diffusers") from exc

    unet.set_attn_processor(AttnProcessor())
