from __future__ import annotations

from typing import Any

import torch
from einops import rearrange


def index_first_axis(
    input_tensor: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Select rows from a tensor along its first axis."""

    return torch.index_select(
        input_tensor,
        dim=0,
        index=indices,
    )


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Remove padded tokens from a batched tensor.

    Compatible with flash_attn.bert_padding.unpad_input.
    """

    if attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask must have shape [batch_size, sequence_length], "
            f"got {tuple(attention_mask.shape)}"
        )

    if hidden_states.ndim < 2:
        raise ValueError(
            "hidden_states must have shape [batch_size, sequence_length, ...], "
            f"got {tuple(hidden_states.shape)}"
        )

    batch_size, sequence_length = attention_mask.shape

    if hidden_states.shape[0] != batch_size:
        raise ValueError(
            "hidden_states and attention_mask have different batch sizes: "
            f"{hidden_states.shape[0]} and {batch_size}"
        )

    if hidden_states.shape[1] != sequence_length:
        raise ValueError(
            "hidden_states and attention_mask have different sequence lengths: "
            f"{hidden_states.shape[1]} and {sequence_length}"
        )

    mask = attention_mask.to(dtype=torch.bool)

    sequence_lengths = mask.sum(
        dim=-1,
        dtype=torch.int32,
    )

    indices = torch.nonzero(
        mask.reshape(-1),
        as_tuple=False,
    ).reshape(-1)

    cumulative_sequence_lengths = torch.zeros(
        batch_size + 1,
        dtype=torch.int32,
        device=attention_mask.device,
    )

    cumulative_sequence_lengths[1:] = torch.cumsum(
        sequence_lengths,
        dim=0,
        dtype=torch.int32,
    )

    max_sequence_length = (
        int(sequence_lengths.max().item())
        if batch_size > 0
        else 0
    )

    flattened = rearrange(
        hidden_states,
        "b s ... -> (b s) ...",
    )

    unpadded = index_first_axis(
        flattened,
        indices,
    )

    return (
        unpadded,
        indices,
        cumulative_sequence_lengths,
        max_sequence_length,
    )


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    sequence_length: int,
) -> torch.Tensor:
    """Restore an unpadded tensor to its padded batch representation."""

    output_shape: tuple[Any, ...] = (
        batch_size * sequence_length,
        *hidden_states.shape[1:],
    )

    flattened = hidden_states.new_zeros(output_shape)

    flattened.index_copy_(
        0,
        indices,
        hidden_states,
    )

    return rearrange(
        flattened,
        "(b s) ... -> b s ...",
        b=batch_size,
        s=sequence_length,
    )
