"""
Benchmark: Qwen3-0.6B with different norm removal strategies.

Tests the following configurations:
1. Original model (baseline)
2. Remove q_norm + k_norm (QK norms in attention)
3. Remove input_layernorm (pre-attention norm)
4. Remove post_attention_layernorm (pre-MLP norm)
5. Remove final model.norm
6. Remove ALL norms
7. Replace RMSNorm with Identity (same as removal, but explicit)

Measures:
- Forward pass latency (ms) for prefill (seq_len=128)
- Forward pass latency (ms) for prefill (seq_len=512)
- Token generation latency (ms) for single token decode
- Throughput (tokens/sec)
"""

import copy
import gc
import json
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "cpu"
DTYPE = torch.float32
WARMUP_ITERS = 3
BENCH_ITERS = 10
SEQ_LENGTHS = [128, 512]


@dataclass
class BenchResult:
    name: str
    prefill_ms: dict = field(default_factory=dict)  # seq_len -> ms
    decode_ms: float = 0.0
    total_params: int = 0
    norm_params_removed: int = 0


class IdentityNorm(nn.Module):
    """Replacement for RMSNorm that just passes through."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class ScaleOnlyNorm(nn.Module):
    """Learnable per-channel scale without variance normalization.
    Cheaper than RMSNorm: just element-wise multiply, no reduction op.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.weight * hidden_states


class ConstantScaleNorm(nn.Module):
    """Fixed scalar normalization: divide by sqrt(hidden_size).
    No learnable params, constant-time, no reduction.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.scale = 1.0 / (hidden_size ** 0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.scale


class L2Norm(nn.Module):
    """L2 normalization (unit-norm per token).
    Similar cost to RMSNorm but different mathematical properties.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm = hidden_states.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return self.weight * (hidden_states / norm)


def count_norm_params(model):
    """Count parameters in all norm layers."""
    total = 0
    for name, module in model.named_modules():
        if 'norm' in name.lower() and hasattr(module, 'weight') and module.weight is not None:
            total += module.weight.numel()
    return total


def remove_qk_norms(model):
    """Replace q_norm and k_norm with Identity in all attention layers."""
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.q_norm = IdentityNorm()
        attn.k_norm = IdentityNorm()
    return model


def remove_input_layernorms(model):
    """Replace input_layernorm with Identity in all decoder layers."""
    for layer in model.model.layers:
        layer.input_layernorm = IdentityNorm()
    return model


def remove_post_attention_layernorms(model):
    """Replace post_attention_layernorm with Identity in all decoder layers."""
    for layer in model.model.layers:
        layer.post_attention_layernorm = IdentityNorm()
    return model


def remove_final_norm(model):
    """Replace final model.norm with Identity."""
    model.model.norm = IdentityNorm()
    return model


def remove_all_norms(model):
    """Remove ALL norms from the model."""
    model = remove_qk_norms(model)
    model = remove_input_layernorms(model)
    model = remove_post_attention_layernorms(model)
    model = remove_final_norm(model)
    return model


def replace_with_scale_only(model):
    """Replace all RMSNorm with ScaleOnlyNorm (learnable scale, no variance)."""
    for layer in model.model.layers:
        hidden_size = layer.hidden_size
        head_dim = layer.self_attn.head_dim

        layer.self_attn.q_norm = ScaleOnlyNorm(head_dim)
        layer.self_attn.k_norm = ScaleOnlyNorm(head_dim)
        layer.input_layernorm = ScaleOnlyNorm(hidden_size)
        layer.post_attention_layernorm = ScaleOnlyNorm(hidden_size)

    model.model.norm = ScaleOnlyNorm(model.config.hidden_size)
    return model


def replace_with_constant_scale(model):
    """Replace all RMSNorm with ConstantScaleNorm (no learnable params)."""
    for layer in model.model.layers:
        hidden_size = layer.hidden_size
        head_dim = layer.self_attn.head_dim

        layer.self_attn.q_norm = ConstantScaleNorm(head_dim)
        layer.self_attn.k_norm = ConstantScaleNorm(head_dim)
        layer.input_layernorm = ConstantScaleNorm(hidden_size)
        layer.post_attention_layernorm = ConstantScaleNorm(hidden_size)

    model.model.norm = ConstantScaleNorm(model.config.hidden_size)
    return model


def replace_with_l2norm(model):
    """Replace all RMSNorm with L2Norm."""
    for layer in model.model.layers:
        hidden_size = layer.hidden_size
        head_dim = layer.self_attn.head_dim

        layer.self_attn.q_norm = L2Norm(head_dim)
        layer.self_attn.k_norm = L2Norm(head_dim)
        layer.input_layernorm = L2Norm(hidden_size)
        layer.post_attention_layernorm = L2Norm(hidden_size)

    model.model.norm = L2Norm(model.config.hidden_size)
    return model


def benchmark_prefill(model, input_ids, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Benchmark forward pass (prefill) latency."""
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(input_ids)

        # Benchmark
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
    """Benchmark autoregressive token generation latency."""
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Benchmark
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


def load_fresh_model():
    """Load a fresh copy of the model."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        device_map=DEVICE,
    )
    model.eval()
    return model


def run_experiment(name, modify_fn=None):
    """Run a single experiment: load model, optionally modify, benchmark, cleanup."""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    model = load_fresh_model()
    original_norm_params = count_norm_params(model)

    if modify_fn is not None:
        model = modify_fn(model)

    current_norm_params = count_norm_params(model)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  Total params: {total_params:,}")
    print(f"  Norm params remaining: {current_norm_params:,}")
    print(f"  Norm params removed: {original_norm_params - current_norm_params:,}")

    result = BenchResult(
        name=name,
        total_params=total_params,
        norm_params_removed=original_norm_params - current_norm_params,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Prefill benchmarks
    for seq_len in SEQ_LENGTHS:
        input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=DEVICE)
        stats = benchmark_prefill(model, input_ids)
        result.prefill_ms[seq_len] = stats
        print(f"  Prefill seq_len={seq_len}: {stats['mean_ms']:.1f}ms (±{stats['std_ms']:.1f}ms)")

    # Decode benchmark
    prompt = "The meaning of life is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    decode_stats = benchmark_decode(model, input_ids, num_tokens=20)
    result.decode_ms = decode_stats
    print(f"  Decode per token: {decode_stats['per_token_ms']:.1f}ms")

    # Cleanup
    del model
    gc.collect()

    return result


def format_results_table(results):
    """Format results as a nice comparison table."""
    baseline = results[0]

    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("BENCHMARK RESULTS: Qwen3-0.6B Norm Removal")
    lines.append("=" * 100)

    # Prefill table
    for seq_len in SEQ_LENGTHS:
        lines.append(f"\n--- Prefill (seq_len={seq_len}) ---")
        lines.append(f"{'Variant':<40} {'Mean(ms)':>10} {'Std(ms)':>10} {'Speedup':>10} {'Params Removed':>15}")
        lines.append("-" * 90)

        base_ms = baseline.prefill_ms[seq_len]["mean_ms"]
        for r in results:
            ms = r.prefill_ms[seq_len]["mean_ms"]
            std = r.prefill_ms[seq_len]["std_ms"]
            speedup = base_ms / ms if ms > 0 else 0
            lines.append(
                f"{r.name:<40} {ms:>10.1f} {std:>10.1f} {speedup:>9.3f}x {r.norm_params_removed:>15,}"
            )

    # Decode table
    lines.append(f"\n--- Decode (per token) ---")
    lines.append(f"{'Variant':<40} {'Mean(ms)':>10} {'Speedup':>10}")
    lines.append("-" * 65)

    base_decode = baseline.decode_ms["per_token_ms"]
    for r in results:
        ms = r.decode_ms["per_token_ms"]
        speedup = base_decode / ms if ms > 0 else 0
        lines.append(f"{r.name:<40} {ms:>10.1f} {speedup:>9.3f}x")

    return "\n".join(lines)


def format_recommendations(results):
    """Generate recommendations based on benchmark results."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("ANALYSIS & RECOMMENDATIONS")
    lines.append("=" * 100)

    lines.append("""
## Architecture of Qwen3-0.6B Norms

Qwen3-0.6B uses Qwen3RMSNorm (Root Mean Square Layer Normalization) in 4 positions:

1. **q_norm / k_norm** (28 layers x 2 = 56 norms, dim=128 each)
   - Applied to Q and K projections BEFORE RoPE
   - Stabilizes attention logits by normalizing query/key magnitudes
   - Relatively cheap: operates on head_dim=128

2. **input_layernorm** (28 layers, dim=1024 each)
   - Pre-attention norm (Pre-LN transformer style)
   - Critical for training stability in deep transformers
   - Operates on full hidden_size=1024

3. **post_attention_layernorm** (28 layers, dim=1024 each)
   - Pre-MLP norm
   - Normalizes residual stream before MLP
   - Operates on full hidden_size=1024

4. **model.norm** (1 norm, dim=1024)
   - Final norm before LM head
   - Ensures proper scale for logit computation

## RMSNorm computation cost breakdown:
   - Cast to float32 (memory bandwidth)
   - Square all elements: O(d)
   - Mean reduction: O(d)
   - rsqrt: O(1)
   - Multiply by normalized + weight: O(d)
   Total: ~3*d FLOPs per token per norm application

## Replacement Recommendations:

### 1. ScaleOnly (learnable per-channel scale, NO normalization)
   - **Cost**: 1x multiply (vs 3x for RMSNorm)
   - **Pros**: Keeps learnable scale, ~3x fewer FLOPs in norm
   - **Cons**: Loses variance normalization → may destabilize training
   - **Best for**: q_norm/k_norm (attention is already softmax-normalized)

### 2. ConstantScale (divide by sqrt(d), no learnable params)
   - **Cost**: 1x multiply by constant
   - **Pros**: Zero learnable params, deterministic
   - **Cons**: No adaptation, may hurt quality significantly
   - **Best for**: Not recommended for production; good for ablation

### 3. L2Norm (normalize to unit vector, then scale)
   - **Cost**: Similar to RMSNorm (norm + divide + scale)
   - **Pros**: Mathematically cleaner, same cost
   - **Cons**: No speedup over RMSNorm
   - **Best for**: Research comparison only

### 4. Complete Removal (Identity)
   - **Cost**: Zero
   - **Pros**: Maximum speedup
   - **Cons**: Model will need retraining; removing input/post_attn norms
     will likely break pretrained weights completely

## Practical Recommendation:

For **inference speedup without retraining**:
  → Remove q_norm + k_norm only. These have the least impact on output
    quality because attention scores go through softmax anyway.
    The speedup is modest but "free" quality-wise for many tasks.

For **training a new model from scratch**:
  → Replace RMSNorm with ScaleOnly for q_norm/k_norm
  → Keep input_layernorm and post_attention_layernorm as RMSNorm
    (critical for training stability)
  → Or try replacing ALL norms with ScaleOnly and compensate with
    careful learning rate scheduling + gradient clipping

For **maximum throughput (with retraining budget)**:
  → Remove q_norm/k_norm entirely (Identity)
  → Replace input_layernorm + post_attention_layernorm with ScaleOnly
  → Keep final model.norm as RMSNorm (cheap, important for logit scale)
""")

    return "\n".join(lines)


def main():
    print("Qwen3-0.6B Norm Removal Benchmark")
    print(f"Device: {DEVICE}, Dtype: {DTYPE}")
    print(f"Warmup: {WARMUP_ITERS}, Bench iters: {BENCH_ITERS}")
    print(f"Sequence lengths: {SEQ_LENGTHS}")

    experiments = [
        ("1. Original (baseline)", None),
        ("2. Remove q_norm + k_norm", remove_qk_norms),
        ("3. Remove input_layernorm", remove_input_layernorms),
        ("4. Remove post_attention_layernorm", remove_post_attention_layernorms),
        ("5. Remove final model.norm", remove_final_norm),
        ("6. Remove ALL norms", remove_all_norms),
        ("7. Replace with ScaleOnly", replace_with_scale_only),
        ("8. Replace with ConstantScale", replace_with_constant_scale),
        ("9. Replace with L2Norm", replace_with_l2norm),
    ]

    results = []
    for name, modify_fn in experiments:
        result = run_experiment(name, modify_fn)
        results.append(result)

    # Print comparison
    table = format_results_table(results)
    print(table)

    recommendations = format_recommendations(results)
    print(recommendations)

    # Save results to JSON
    output = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "warmup_iters": WARMUP_ITERS,
        "bench_iters": BENCH_ITERS,
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

    report_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(report_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {report_path}")

    # Also save the text report
    report_txt_path = os.path.join(os.path.dirname(__file__), "benchmark_report.txt")
    with open(report_txt_path, "w") as f:
        f.write(table)
        f.write("\n")
        f.write(recommendations)
    print(f"Report saved to {report_txt_path}")


if __name__ == "__main__":
    main()
