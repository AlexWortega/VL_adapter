"""
Benchmark: GLM-4.7-Flash (30B-A3B MoE) with different norm removal strategies.

Since the full model (30B params, ~120GB fp32) doesn't fit in RAM,
we create the model from config with random weights. This gives
accurate latency measurements for the architecture since norm overhead
is independent of weight values.

GLM-4.7-Flash uses MLA (Multi-head Latent Attention) with low-rank
projections, making its norm placement different from standard models:

Norm positions:
1. q_a_layernorm  (dim=768)  x47  — after Q low-rank projection
2. kv_a_layernorm (dim=512)  x47  — after KV low-rank projection
3. input_layernorm       (dim=2048) x47  — pre-attention
4. post_attention_layernorm (dim=2048) x47  — pre-MLP/MoE
5. model.norm            (dim=2048) x1   — final norm

We benchmark a SCALED-DOWN version (fewer layers, fewer experts)
that fits in RAM while preserving the exact same norm structure.
"""

import copy
import gc
import json
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM


MODEL_NAME = "zai-org/GLM-4.7-Flash"
DEVICE = "cpu"
DTYPE = torch.float32
WARMUP_ITERS = 3
BENCH_ITERS = 10
SEQ_LENGTHS = [128, 512]


class ScaleOnlyNorm(nn.Module):
    """Learnable per-channel scale without variance normalization."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return self.weight * hidden_states

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class IdentityNorm(nn.Module):
    """No-op replacement."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, hidden_states):
        return hidden_states


class ConstantScaleNorm(nn.Module):
    """Fixed scalar normalization: divide by sqrt(hidden_size)."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.scale = 1.0 / (hidden_size ** 0.5)

    def forward(self, hidden_states):
        return hidden_states * self.scale


@dataclass
class BenchResult:
    name: str
    prefill_ms: dict = field(default_factory=dict)
    decode_ms: float = 0.0
    total_params: int = 0
    norm_params_removed: int = 0


def count_norm_params(model):
    total = 0
    for name, module in model.named_modules():
        if 'norm' in name.lower() and hasattr(module, 'weight') and module.weight is not None:
            total += module.weight.numel()
    return total


def remove_mla_norms(model):
    """Remove q_a_layernorm and kv_a_layernorm (MLA low-rank norms)."""
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, 'q_a_layernorm'):
            attn.q_a_layernorm = IdentityNorm()
        if hasattr(attn, 'kv_a_layernorm'):
            attn.kv_a_layernorm = IdentityNorm()
    return model


def remove_input_layernorms(model):
    for layer in model.model.layers:
        layer.input_layernorm = IdentityNorm()
    return model


def remove_post_attention_layernorms(model):
    for layer in model.model.layers:
        layer.post_attention_layernorm = IdentityNorm()
    return model


def remove_final_norm(model):
    model.model.norm = IdentityNorm()
    return model


def remove_all_norms(model):
    model = remove_mla_norms(model)
    model = remove_input_layernorms(model)
    model = remove_post_attention_layernorms(model)
    model = remove_final_norm(model)
    return model


def replace_with_scale_only(model):
    """Replace all RMSNorm with ScaleOnly."""
    eps = model.config.rms_norm_eps
    for layer in model.model.layers:
        hidden_size = layer.hidden_size
        attn = layer.self_attn

        if hasattr(attn, 'q_a_layernorm'):
            dim = attn.q_a_layernorm.weight.shape[0] if hasattr(attn.q_a_layernorm, 'weight') else model.config.q_lora_rank
            attn.q_a_layernorm = ScaleOnlyNorm(dim, eps=eps)
        if hasattr(attn, 'kv_a_layernorm'):
            dim = attn.kv_a_layernorm.weight.shape[0] if hasattr(attn.kv_a_layernorm, 'weight') else model.config.kv_lora_rank
            attn.kv_a_layernorm = ScaleOnlyNorm(dim, eps=eps)

        layer.input_layernorm = ScaleOnlyNorm(hidden_size, eps=eps)
        layer.post_attention_layernorm = ScaleOnlyNorm(hidden_size, eps=eps)

    model.model.norm = ScaleOnlyNorm(model.config.hidden_size, eps=eps)
    return model


def replace_with_constant_scale(model):
    """Replace all RMSNorm with ConstantScale."""
    eps = model.config.rms_norm_eps
    for layer in model.model.layers:
        hidden_size = layer.hidden_size
        attn = layer.self_attn

        if hasattr(attn, 'q_a_layernorm'):
            dim = attn.q_a_layernorm.weight.shape[0] if hasattr(attn.q_a_layernorm, 'weight') else model.config.q_lora_rank
            attn.q_a_layernorm = ConstantScaleNorm(dim, eps=eps)
        if hasattr(attn, 'kv_a_layernorm'):
            dim = attn.kv_a_layernorm.weight.shape[0] if hasattr(attn.kv_a_layernorm, 'weight') else model.config.kv_lora_rank
            attn.kv_a_layernorm = ConstantScaleNorm(dim, eps=eps)

        layer.input_layernorm = ConstantScaleNorm(hidden_size, eps=eps)
        layer.post_attention_layernorm = ConstantScaleNorm(hidden_size, eps=eps)

    model.model.norm = ConstantScaleNorm(model.config.hidden_size, eps=eps)
    return model


def benchmark_prefill(model, input_ids, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)

        times = []
        for _ in range(iters):
            start = time.perf_counter()
            _ = model(input_ids)
            end = time.perf_counter()
            times.append((end - start) * 1000)

    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def benchmark_decode(model, input_ids, num_tokens=20, warmup=2, iters=5):
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        times = []
        for _ in range(iters):
            start = time.perf_counter()
            output = model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            end = time.perf_counter()
            total_ms = (end - start) * 1000
            generated_tokens = output.shape[1] - input_ids.shape[1]
            times.append(total_ms / max(generated_tokens, 1))

    return {
        "per_token_ms": sum(times) / len(times),
        "min_per_token_ms": min(times),
        "max_per_token_ms": max(times),
    }


def create_scaled_model(num_layers=8, n_routed_experts=8):
    """Create a scaled-down GLM-4.7-Flash that fits in RAM.

    Preserves the exact same layer structure (MLA + MoE + norms),
    just with fewer layers and experts for benchmarking norm overhead.
    """
    config = AutoConfig.from_pretrained(MODEL_NAME)

    # Scale down to fit in ~16GB RAM
    config.num_hidden_layers = num_layers
    config.n_routed_experts = n_routed_experts
    config.num_local_experts = n_routed_experts
    config.num_experts_per_tok = min(config.num_experts_per_tok, n_routed_experts)
    config.mlp_layer_types = config.mlp_layer_types[:num_layers]
    if hasattr(config, 'layer_types') and config.layer_types:
        config.layer_types = config.layer_types[:num_layers]
    # Remove next-token prediction layers to simplify
    config.num_nextn_predict_layers = 0

    print(f"  Config: {num_layers} layers, {n_routed_experts} experts, "
          f"hidden={config.hidden_size}, MLA q_lora_rank={config.q_lora_rank}, "
          f"kv_lora_rank={config.kv_lora_rank}")

    model = AutoModelForCausalLM.from_config(config)
    model = model.to(dtype=DTYPE)
    model.eval()
    return model


def run_experiment(name, modify_fn=None, num_layers=8, n_routed_experts=8):
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    model = create_scaled_model(num_layers=num_layers, n_routed_experts=n_routed_experts)
    original_norm_params = count_norm_params(model)

    if modify_fn is not None:
        model = modify_fn(model)

    current_norm_params = count_norm_params(model)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  Total params: {total_params:,}")
    print(f"  Norm params remaining: {current_norm_params:,}")
    print(f"  Norm params removed: {original_norm_params - current_norm_params:,}")

    # List norms
    if modify_fn is None:
        print(f"\n  Norm layers:")
        seen = {}
        for n, m in model.named_modules():
            if 'norm' in n.lower():
                key = n.split('.')[-1]
                tname = type(m).__name__
                if key not in seen:
                    seen[key] = {'type': tname, 'count': 0,
                                 'params': sum(p.numel() for p in m.parameters())}
                seen[key]['count'] += 1
        for key, info in seen.items():
            print(f"    {key}: {info['type']} x{info['count']}, {info['params']} params each")

    result = BenchResult(
        name=name,
        total_params=total_params,
        norm_params_removed=original_norm_params - current_norm_params,
    )

    # Prefill benchmarks
    for seq_len in SEQ_LENGTHS:
        input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=DEVICE)
        stats = benchmark_prefill(model, input_ids)
        result.prefill_ms[seq_len] = stats
        print(f"  Prefill seq_len={seq_len}: {stats['mean_ms']:.1f}ms (±{stats['std_ms']:.1f}ms)")

    # Decode benchmark
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8), device=DEVICE)
    decode_stats = benchmark_decode(model, input_ids, num_tokens=20)
    result.decode_ms = decode_stats
    print(f"  Decode per token: {decode_stats['per_token_ms']:.1f}ms")

    del model
    gc.collect()
    return result


def format_results_table(results):
    baseline = results[0]
    lines = []
    lines.append("\n" + "=" * 110)
    lines.append("BENCHMARK RESULTS: GLM-4.7-Flash (scaled-down) Norm Removal")
    lines.append("=" * 110)

    for seq_len in SEQ_LENGTHS:
        lines.append(f"\n--- Prefill (seq_len={seq_len}) ---")
        lines.append(f"{'Variant':<45} {'Mean(ms)':>10} {'Std(ms)':>10} {'Speedup':>10} {'Params Removed':>15}")
        lines.append("-" * 95)

        base_ms = baseline.prefill_ms[seq_len]["mean_ms"]
        for r in results:
            ms = r.prefill_ms[seq_len]["mean_ms"]
            std = r.prefill_ms[seq_len]["std_ms"]
            speedup = base_ms / ms if ms > 0 else 0
            lines.append(
                f"{r.name:<45} {ms:>10.1f} {std:>10.1f} {speedup:>9.3f}x {r.norm_params_removed:>15,}"
            )

    lines.append(f"\n--- Decode (per token) ---")
    lines.append(f"{'Variant':<45} {'Mean(ms)':>10} {'Speedup':>10}")
    lines.append("-" * 70)

    base_decode = baseline.decode_ms["per_token_ms"]
    for r in results:
        ms = r.decode_ms["per_token_ms"]
        speedup = base_decode / ms if ms > 0 else 0
        lines.append(f"{r.name:<45} {ms:>10.1f} {speedup:>9.3f}x")

    return "\n".join(lines)


def format_analysis():
    lines = []
    lines.append("\n" + "=" * 110)
    lines.append("ANALYSIS: GLM-4.7-Flash Norm Architecture")
    lines.append("=" * 110)
    lines.append("""
## GLM-4.7-Flash Architecture (30B-A3B MoE with MLA)

Key differences from standard transformers:

1. **MLA (Multi-head Latent Attention)** uses low-rank projections:
   - Q: hidden(2048) → q_a_proj(768) → **q_a_layernorm** → q_b_proj(heads*qk_dim)
   - KV: hidden(2048) → kv_a_proj(512+64) → **kv_a_layernorm** → kv_b_proj(...)
   These norms sit BETWEEN the low-rank compression and decompression.

2. **MoE (Mixture of Experts)**: 64 routed + 1 shared expert per layer
   - The post_attention_layernorm normalizes BEFORE the MoE router
   - Important for router stability (expert selection)

3. **47 layers** with Pre-LN pattern

## Norm inventory (full model):

| Norm                     | Dim  | Count | Params each | Total     |
|--------------------------|------|-------|-------------|-----------|
| q_a_layernorm            | 768  | 47    | 768         | 36,096    |
| kv_a_layernorm           | 512  | 47    | 512         | 24,064    |
| input_layernorm          | 2048 | 47    | 2,048       | 96,256    |
| post_attention_layernorm | 2048 | 47    | 2,048       | 96,256    |
| model.norm               | 2048 | 1     | 2,048       | 2,048     |
| **Total**                |      | **189** |           | **254,720** (0.0008% of 30B) |

## Cost of RMSNorm per call:
   - Cast to fp32: memory bandwidth
   - x.pow(2).mean(-1): reduction over dim
   - rsqrt + multiply: O(1) + O(dim)
   - weight * result: O(dim)
   For dim=2048: ~6K FLOPs per token per norm
   For dim=768: ~2.3K FLOPs
   For dim=512: ~1.5K FLOPs

## Replacement recommendations for GLM-4.7-Flash:

### MLA norms (q_a_layernorm, kv_a_layernorm) — MOST IMPACTFUL TO REMOVE
   - These norms are called on EVERY token for EVERY layer
   - They operate on compressed representations (768, 512 dims)
   - The low-rank bottleneck already constrains magnitudes
   - **Recommendation**: Replace with ScaleOnly or remove entirely
   - Expected speedup: ~3-8% (architecture-dependent)

### input_layernorm / post_attention_layernorm — RISKY TO REMOVE
   - Critical for Pre-LN training stability
   - post_attention_layernorm is especially important for MoE:
     it normalizes inputs to the router, affecting expert selection
   - **Recommendation**: Replace with ScaleOnly for training from scratch
   - Keep as RMSNorm for fine-tuning existing weights

### model.norm (final) — KEEP
   - Only 1 instance, negligible cost
   - Important for LM head logit scale

### Practical strategy:
   1. **Inference speedup (no retrain)**: Remove q_a_layernorm + kv_a_layernorm
   2. **Train from scratch**: ScaleOnly for all MLA norms, RMSNorm for layer norms
   3. **Maximum speed**: ScaleOnly everywhere except model.norm
""")
    return "\n".join(lines)


def main():
    NUM_LAYERS = 8
    N_EXPERTS = 8

    print("GLM-4.7-Flash Norm Removal Benchmark (scaled-down)")
    print(f"Scaled config: {NUM_LAYERS} layers, {N_EXPERTS} experts")
    print(f"Device: {DEVICE}, Dtype: {DTYPE}")
    print(f"Warmup: {WARMUP_ITERS}, Bench iters: {BENCH_ITERS}")

    experiments = [
        ("1. Original (baseline)", None),
        ("2. Remove MLA norms (q_a + kv_a)", remove_mla_norms),
        ("3. Remove input_layernorm", remove_input_layernorms),
        ("4. Remove post_attention_layernorm", remove_post_attention_layernorms),
        ("5. Remove final model.norm", remove_final_norm),
        ("6. Remove ALL norms", remove_all_norms),
        ("7. Replace ALL with ScaleOnly", replace_with_scale_only),
        ("8. Replace ALL with ConstantScale", replace_with_constant_scale),
    ]

    results = []
    for name, modify_fn in experiments:
        result = run_experiment(name, modify_fn,
                                num_layers=NUM_LAYERS,
                                n_routed_experts=N_EXPERTS)
        results.append(result)

    table = format_results_table(results)
    print(table)

    analysis = format_analysis()
    print(analysis)

    # Save results
    output = {
        "model": MODEL_NAME,
        "scaled_config": {"num_layers": NUM_LAYERS, "n_routed_experts": N_EXPERTS},
        "device": DEVICE,
        "dtype": str(DTYPE),
        "results": []
    }
    for r in results:
        output["results"].append({
            "name": r.name,
            "total_params": r.total_params,
            "norm_params_removed": r.norm_params_removed,
            "prefill_ms": {str(k): v for k, v in r.prefill_ms.items()},
            "decode_ms": r.decode_ms,
        })

    report_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(report_dir, "benchmark_glm_flash_results.json"), "w") as f:
        json.dump(output, f, indent=2)

    with open(os.path.join(report_dir, "benchmark_glm_flash_report.txt"), "w") as f:
        f.write(table + "\n" + analysis)

    print(f"\nResults saved.")


if __name__ == "__main__":
    main()
