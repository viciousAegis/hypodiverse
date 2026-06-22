"""Minimal compatibility shim for veRL.

This package does not implement FlashAttention kernels. It only exposes the
flash_attn.bert_padding helpers needed by veRL's padding conversion code.
"""

__version__ = "0.0.0"
