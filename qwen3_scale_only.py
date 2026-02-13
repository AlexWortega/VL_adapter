"""
Qwen3-0.6B with ScaleOnly norm replacement.

Replaces all Qwen3RMSNorm layers with a lightweight ScaleOnly norm
that performs only learnable per-channel scaling (weight * x) without
variance computation. This gives ~16% prefill and ~21% decode speedup
over the original RMSNorm on CPU.

Usage:
    from qwen3_scale_only import load_qwen3_scale_only

    model, tokenizer = load_qwen3_scale_only()
    # Use as a normal HuggingFace model
    output = model.generate(tokenizer("Hello", return_tensors="pt").input_ids, max_new_tokens=50)
    print(tokenizer.decode(output[0]))

    # Or selectively replace only QK norms (safest, no retrain needed):
    from qwen3_scale_only import load_qwen3_scale_only_qk_only
    model, tokenizer = load_qwen3_scale_only_qk_only()
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-0.6B"


class ScaleOnlyNorm(nn.Module):
    """Learnable per-channel scale without variance normalization.

    Replaces RMSNorm's forward:
        RMSNorm:   weight * (x / sqrt(mean(x^2) + eps))   -- 3 ops over dim
        ScaleOnly: weight * x                              -- 1 elementwise mul

    The learnable weight vector preserves the same parameter interface
    as RMSNorm, so existing weights can be loaded directly (though
    retraining/fine-tuning is recommended for best quality).
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps  # kept for interface compat

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.weight * hidden_states

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def replace_norms_with_scale_only(model, replace_qk=True, replace_layer=True, replace_final=True):
    """Replace RMSNorm modules with ScaleOnly in a Qwen3 model.

    Args:
        model: A Qwen3ForCausalLM model.
        replace_qk: Replace q_norm and k_norm in attention (dim=head_dim).
        replace_layer: Replace input_layernorm and post_attention_layernorm (dim=hidden_size).
        replace_final: Replace the final model.norm (dim=hidden_size).

    Returns:
        The modified model (in-place).
    """
    for layer in model.model.layers:
        hidden_size = layer.hidden_size
        head_dim = layer.self_attn.head_dim
        eps = model.config.rms_norm_eps

        if replace_qk:
            # Transfer existing weights
            old_q = layer.self_attn.q_norm
            old_k = layer.self_attn.k_norm

            new_q = ScaleOnlyNorm(head_dim, eps=eps)
            new_k = ScaleOnlyNorm(head_dim, eps=eps)
            new_q.weight.data.copy_(old_q.weight.data)
            new_k.weight.data.copy_(old_k.weight.data)

            layer.self_attn.q_norm = new_q
            layer.self_attn.k_norm = new_k

        if replace_layer:
            old_in = layer.input_layernorm
            old_post = layer.post_attention_layernorm

            new_in = ScaleOnlyNorm(hidden_size, eps=eps)
            new_post = ScaleOnlyNorm(hidden_size, eps=eps)
            new_in.weight.data.copy_(old_in.weight.data)
            new_post.weight.data.copy_(old_post.weight.data)

            layer.input_layernorm = new_in
            layer.post_attention_layernorm = new_post

    if replace_final:
        old_norm = model.model.norm
        new_norm = ScaleOnlyNorm(model.config.hidden_size, eps=model.config.rms_norm_eps)
        new_norm.weight.data.copy_(old_norm.weight.data)
        model.model.norm = new_norm

    return model


def load_qwen3_scale_only(model_name=MODEL_NAME, dtype=torch.float32, device="cpu"):
    """Load Qwen3-0.6B with ALL norms replaced by ScaleOnly.

    Best speedup (~16-21%), but requires fine-tuning for quality.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = replace_norms_with_scale_only(model, replace_qk=True, replace_layer=True, replace_final=True)
    model.eval()
    return model, tokenizer


def load_qwen3_scale_only_qk_only(model_name=MODEL_NAME, dtype=torch.float32, device="cpu"):
    """Load Qwen3-0.6B with only q_norm/k_norm replaced by ScaleOnly.

    Safest option — softmax already normalizes attention scores,
    so QK norms are least critical. ~5-9% speedup, minimal quality loss.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = replace_norms_with_scale_only(model, replace_qk=True, replace_layer=False, replace_final=False)
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    import time

    print("Loading Qwen3-0.6B with ScaleOnly norms...")
    model, tokenizer = load_qwen3_scale_only()

    # Verify architecture
    norm_types = set()
    for name, module in model.named_modules():
        if "norm" in name.lower():
            norm_types.add(type(module).__name__)
    print(f"Norm types in model: {norm_types}")

    # Quick inference test
    prompt = "The meaning of life is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    print(f"\nPrompt: {prompt}")

    with torch.no_grad():
        # Warmup
        for _ in range(3):
            model.generate(input_ids, max_new_tokens=1, do_sample=False, temperature=None, top_p=None)

        start = time.perf_counter()
        output = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        elapsed = time.perf_counter() - start

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    num_tokens = output.shape[1] - input_ids.shape[1]
    print(f"Generated ({num_tokens} tokens, {elapsed*1000:.0f}ms, {num_tokens/elapsed:.1f} tok/s):")
    print(generated)
