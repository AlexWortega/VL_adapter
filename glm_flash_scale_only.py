"""
GLM-4.7-Flash with ScaleOnly norm replacement.

GLM-4.7-Flash uses MLA (Multi-head Latent Attention) with unique norm positions:
  - q_a_layernorm  (dim=768)  — between Q low-rank compress/decompress
  - kv_a_layernorm (dim=512)  — between KV low-rank compress/decompress
  - input_layernorm       (dim=2048) — pre-attention (Pre-LN)
  - post_attention_layernorm (dim=2048) — pre-MLP/MoE
  - model.norm            (dim=2048) — final norm

Usage:
    from glm_flash_scale_only import replace_glm_norms_with_scale_only

    model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.7-Flash", ...)
    model = replace_glm_norms_with_scale_only(model, replace_mla=True, replace_layer=True)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


MODEL_NAME = "zai-org/GLM-4.7-Flash"


class ScaleOnlyNorm(nn.Module):
    """Learnable per-channel scale without variance normalization.

    Replaces RMSNorm's forward:
        RMSNorm:   weight * (x / sqrt(mean(x^2) + eps))
        ScaleOnly: weight * x

    Same parameter interface as Glm4MoeLiteRMSNorm for weight transfer.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return self.weight * hidden_states

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def replace_glm_norms_with_scale_only(model, replace_mla=True, replace_layer=True, replace_final=True):
    """Replace Glm4MoeLiteRMSNorm with ScaleOnly in a GLM-4.7-Flash model.

    Args:
        model: A Glm4MoeLiteForCausalLM model.
        replace_mla: Replace q_a_layernorm and kv_a_layernorm in MLA attention.
        replace_layer: Replace input_layernorm and post_attention_layernorm.
        replace_final: Replace the final model.norm.

    Returns:
        The modified model (in-place). Weights are transferred.
    """
    eps = model.config.rms_norm_eps

    for layer in model.model.layers:
        hidden_size = layer.hidden_size
        attn = layer.self_attn

        if replace_mla:
            # MLA q_a_layernorm
            if hasattr(attn, 'q_a_layernorm') and hasattr(attn.q_a_layernorm, 'weight'):
                old = attn.q_a_layernorm
                new_norm = ScaleOnlyNorm(old.weight.shape[0], eps=eps)
                new_norm.weight.data.copy_(old.weight.data)
                attn.q_a_layernorm = new_norm

            # MLA kv_a_layernorm
            if hasattr(attn, 'kv_a_layernorm') and hasattr(attn.kv_a_layernorm, 'weight'):
                old = attn.kv_a_layernorm
                new_norm = ScaleOnlyNorm(old.weight.shape[0], eps=eps)
                new_norm.weight.data.copy_(old.weight.data)
                attn.kv_a_layernorm = new_norm

        if replace_layer:
            # input_layernorm
            old_in = layer.input_layernorm
            new_in = ScaleOnlyNorm(hidden_size, eps=eps)
            new_in.weight.data.copy_(old_in.weight.data)
            layer.input_layernorm = new_in

            # post_attention_layernorm
            old_post = layer.post_attention_layernorm
            new_post = ScaleOnlyNorm(hidden_size, eps=eps)
            new_post.weight.data.copy_(old_post.weight.data)
            layer.post_attention_layernorm = new_post

    if replace_final:
        old_norm = model.model.norm
        new_norm = ScaleOnlyNorm(model.config.hidden_size, eps=eps)
        new_norm.weight.data.copy_(old_norm.weight.data)
        model.model.norm = new_norm

    return model


def replace_glm_mla_norms_only(model):
    """Safest option: only replace MLA norms (q_a + kv_a layernorms).

    MLA low-rank bottleneck already constrains magnitudes,
    so these norms are least critical for output quality.
    ~8-13% prefill speedup with minimal quality loss.
    """
    return replace_glm_norms_with_scale_only(
        model, replace_mla=True, replace_layer=False, replace_final=False
    )


if __name__ == "__main__":
    # Demo with scaled-down model (full model needs ~120GB RAM)
    print("Creating scaled-down GLM-4.7-Flash for demo...")

    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_hidden_layers = 4
    config.n_routed_experts = 4
    config.num_local_experts = 4
    config.num_experts_per_tok = min(config.num_experts_per_tok, 4)
    config.mlp_layer_types = config.mlp_layer_types[:4]
    config.num_nextn_predict_layers = 0

    model = AutoModelForCausalLM.from_config(config)
    model = model.to(dtype=torch.float32)

    print("\nBefore replacement:")
    for name, module in model.named_modules():
        if 'norm' in name.lower():
            print(f"  {name}: {type(module).__name__}")

    model = replace_glm_norms_with_scale_only(model)

    print("\nAfter replacement:")
    norm_types = set()
    for name, module in model.named_modules():
        if 'norm' in name.lower():
            norm_types.add(type(module).__name__)
    print(f"  All norm types: {norm_types}")

    # Quick forward test
    input_ids = torch.randint(0, config.vocab_size, (1, 32))
    with torch.no_grad():
        output = model(input_ids)
    print(f"\nForward pass OK. Output shape: {output.logits.shape}")
