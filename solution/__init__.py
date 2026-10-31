"""Решение команды: Triton-кернелы RMSNorm, RoPE, log_softmax (+ пример SwiGLU из шаблона)."""

from .swiglu import patch_qwen2_mlp


def apply():
    from transformers.models.qwen2 import modeling_qwen2 as m
    from . import kernels
    from .logsoftmax import patch_log_softmax

    kernels._orig_rms = m.Qwen2RMSNorm.forward
    kernels._orig_rope = m.apply_rotary_pos_emb
    m.Qwen2RMSNorm.forward = kernels.rmsnorm_forward
    m.apply_rotary_pos_emb = kernels.apply_rotary_pos_emb
    patch_log_softmax()
    patch_qwen2_mlp()  # пример из шаблона: ускоряет, но в зачёт своих кернелов не идёт
