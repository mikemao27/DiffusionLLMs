import torch
import triton
import triton.testing

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import stats

from custom_kernel import (
    check_cosine_similarity,
    apply_update_hidden_cache,
    kernel_integration,
)

# Global config

DEVICE = triton.runtime.driver.active.get_active_torch_device()
torch.manual_seed(42)

# Correctness thresholds
MASK_AGREEMENT_THRESHOLD = 0.95 # kernel vs. PyTorch mask must agree on ≥95% of tokens
ACTIVE_UPDATE_ATOL = 1e-3 # tolerance for checking active-token cache writes
FROZEN_UPDATE_ATOL = 1e-6 # tolerance for checking frozen-token cache is unchanged

# Default cosine-similarity freeze threshold (front/mid of block)
DEFAULT_THRESHOLD = 0.99

# Benchmark display
SPEEDUP_HEATMAP_VMIN = 0.5 # colormap lower bound (below 1.0 = regression)
SPEEDUP_HEATMAP_VMAX = 2.0 # colormap upper bound (clipped; real max shown if higher)
P_VALUE_SIGNIFICANCE = 0.05 # threshold for marking a speedup as statistically significant

# Positional-regime colours (used consistently across all plots)
REGIME_COLOURS = {"front": "#2196F3", "mid": "#FF9800", "end": "#E53935"}

# Helpers

def make_random_hidden(seq_length: int, dimension_size: int, dtype=torch.float16) -> torch.Tensor:
    return torch.randn(seq_length, dimension_size, dtype=dtype, device=DEVICE)

def pytorch_cosine_similarity_reference(
    previous: torch.Tensor, current: torch.Tensor
) -> torch.Tensor:
    a = previous.float()
    b = current.float()
    dot = (a * b).sum(dim=-1)
    norm_sum = (a * a).sum(dim=-1) + (b * b).sum(dim=-1)
    return dot / (norm_sum + 1e-8).sqrt()

def simulate_hidden_drift(
    base: torch.Tensor,
    target: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    alpha = step / total_steps
    noise_scale = max(0.0, 1.0 - alpha * 2) * 0.05 # 0.05 caps the absolute noise level
    noise = torch.randn_like(base) * noise_scale
    return (1 - alpha) * base + alpha * target + noise

# Part 1 — Correctness

def test_cosine_similarity_correctness(
    seq_length: int = 256,
    dimension_size: int = 256,
    block_length: int = 64,
) -> bool:
    print("-" * 64)
    print("Part 1A: Correctness — cosine similarity kernel vs. PyTorch")
    print("-" * 64)

    previous = make_random_hidden(seq_length, dimension_size)
    current = make_random_hidden(seq_length, dimension_size)
    mask = torch.zeros(seq_length, dtype=torch.int32, device=DEVICE)
    counter = torch.zeros(seq_length, dtype=torch.int32, device=DEVICE)

    check_cosine_similarity(previous, current, mask, counter, block_length=block_length)
    torch.cuda.synchronize()
    triton_mask = mask.clone()

    similarities = pytorch_cosine_similarity_reference(previous, current)
    reference_mask = (similarities <= DEFAULT_THRESHOLD).to(torch.int32)

    agreement = (triton_mask == reference_mask).float().mean().item()
    passed = agreement >= MASK_AGREEMENT_THRESHOLD

    print(f"Agreement with PyTorch reference: {agreement * 100:.1f}%")
    print(f"Result: {'PASSED' if passed else 'FAILED'}")
    print()
    return passed

def test_cache_update_correctness(
    seq_length: int = 128,
    dimension_size: int = 256,
) -> bool:
    print("-" * 64)
    print("Part 1B: Correctness — cache update kernel")
    print("-" * 64)

    new_hidden = make_random_hidden(seq_length, dimension_size)
    cached_hidden = make_random_hidden(seq_length, dimension_size)
    original = cached_hidden.clone()

    active_indices = torch.arange(0, seq_length, 2, dtype=torch.int32, device=DEVICE)
    frozen_indices = torch.arange(1, seq_length, 2, device=DEVICE)

    apply_update_hidden_cache(new_hidden, cached_hidden, active_indices)
    torch.cuda.synchronize()

    active_ok = torch.allclose(
        cached_hidden[active_indices.long()].float(),
        new_hidden[active_indices.long()].float(),
        atol=ACTIVE_UPDATE_ATOL,
    )
    frozen_ok = torch.allclose(
        cached_hidden[frozen_indices].float(),
        original[frozen_indices].float(),
        atol=FROZEN_UPDATE_ATOL,
    )

    print(f"Active positions updated correctly: {'PASSED' if active_ok else 'FAILED'}")
    print(f"Frozen positions left unchanged: {'PASSED' if frozen_ok else 'FAILED'}")
    print()
    return active_ok and frozen_ok

# Part 2 — Freeze-rate characterization

def run_freeze_characterization(
    num_steps: int = 32,
    seq_length: int = 64,
    dimension_size: int = 256,
    block_length: int = 32,
) -> dict[str, list[float]]:
    base = make_random_hidden(seq_length, dimension_size)
    target = make_random_hidden(seq_length, dimension_size)
    cache = base.clone()
    mask = torch.ones(seq_length, dtype=torch.int32, device=DEVICE)
    counter = torch.zeros(seq_length, dtype=torch.int32, device=DEVICE)

    third = block_length // 3
    positions = torch.arange(seq_length, device=DEVICE) % block_length
    region = {
        "front": positions < third,
        "mid": (positions >= third) & (positions < 2 * third),
        "end": positions >= 2 * third,
    }

    freeze_rates: dict[str, list[float]] = {"front": [], "mid": [], "end": []}

    for step in range(num_steps):
        current = simulate_hidden_drift(base, target, step, num_steps)
        kernel_integration(current, cache, mask, counter, block_length=block_length)
        torch.cuda.synchronize()

        for regime, selector in region.items():
            if selector.any():
                rate = (mask[selector] == 0).float().mean().item()
            else:
                rate = 0.0
            freeze_rates[regime].append(rate)

    return freeze_rates

# Part 3 — FLOPs saved vs. hidden-state quality sweep

def run_flops_quality_sweep(
    num_steps: int = 32,
    seq_length: int = 64,
    dimension_size: int = 256,
    block_length: int = 32,
) -> list[dict]:
    base = make_random_hidden(seq_length, dimension_size)
    target = make_random_hidden(seq_length, dimension_size)
    results = []

    for steps_front in range(1, 7):
        steps_mid = max(1, steps_front - 1)
        steps_end = 1

        cache = base.clone()
        mask = torch.ones(seq_length, dtype=torch.int32, device=DEVICE)
        counter = torch.zeros(seq_length, dtype=torch.int32, device=DEVICE)

        active_fractions: list[float] = []
        l2_distances: list[float] = []

        for step in range(num_steps):
            current = simulate_hidden_drift(base, target, step, num_steps)
            kernel_integration(
                current, cache, mask, counter,
                block_length=block_length,
                freeze_steps_front=steps_front,
                freeze_steps_mid=steps_mid,
                freeze_steps_end=steps_end,
            )
            torch.cuda.synchronize()

            active_fractions.append(mask.float().mean().item())

            # Per-token L2: how far is the (possibly stale) cache from the dense output?
            l2 = (cache.float() - current.float()).norm().item() / seq_length
            l2_distances.append(l2)

        results.append({
            "freeze_steps_front": steps_front,
            "freeze_steps_mid": steps_mid,
            "freeze_steps_end": steps_end,
            "mean_active": float(np.mean(active_fractions)),
            "flops_saved": 1.0 - float(np.mean(active_fractions)),
            "mean_l2": float(np.mean(l2_distances)),
        })

    return results

# Part 4 — Wall-clock benchmark

def run_benchmark(
    seq_lengths: tuple[int, ...] = (256, 512, 1024, 2048),
    active_fractions: tuple[float, ...] = (0.9, 0.7, 0.5, 0.3),
    dimension_size: int = 256,
    block_length: int = 32,
    n_warmup: int = 10,
    n_trials: int = 100,
) -> list[dict]:
    print("-" * 72)
    print("Part 4: Wall-clock benchmark — Triton vs. PyTorch")
    print("-" * 72)
    print(f"{'seq_length':>10} {'active%':>8} {'pytorch_ms':>12} "
          f"{'triton_ms':>12} {'speedup':>9} {'p':>8} {'sig':>5}")
    print(" " + "-" * 68)

    all_results = []

    for seq_length in seq_lengths:
        for active_fraction in active_fractions:
            previous = make_random_hidden(seq_length, dimension_size)
            current = make_random_hidden(seq_length, dimension_size)
            cache = previous.clone()
            mask = torch.ones(seq_length, dtype=torch.int32, device=DEVICE)
            counter = torch.zeros(seq_length, dtype=torch.int32, device=DEVICE)

            # Pre-fill freeze counter so the kernel skips (1 - active_fraction) of tokens
            n_frozen = int(seq_length * (1 - active_fraction))
            counter[:n_frozen] = 2

            # Warmup — ensures JIT compilation and GPU warm state don't skew timing
            for _ in range(n_warmup):
                pytorch_cosine_similarity_reference(previous, current)
                kernel_integration(
                    current, cache, mask.clone(), counter.clone(),
                    block_length=block_length,
                )
            torch.cuda.synchronize()

            pytorch_times: list[float] = []
            triton_times: list[float] = []

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for _ in range(n_trials):
                torch.cuda.synchronize()

                start_event.record()
                pytorch_cosine_similarity_reference(previous, current)
                end_event.record()
                torch.cuda.synchronize()
                pytorch_times.append(start_event.elapsed_time(end_event))

                start_event.record()
                kernel_integration(
                    current, cache, mask.clone(), counter.clone(),
                    block_length=block_length,
                )
                end_event.record()
                torch.cuda.synchronize()
                triton_times.append(start_event.elapsed_time(end_event))

            pt = np.array(pytorch_times)
            tr = np.array(triton_times)
            speedup = pt.mean() / tr.mean()
            _, p_val = stats.ttest_ind(pt, tr)
            significant = p_val < P_VALUE_SIGNIFICANCE

            print(f"{seq_length:>10} {active_fraction*100:>7.0f}%"
                  f"{pt.mean():>11.3f}ms {tr.mean():>11.3f}ms"
                  f"{speedup:>8.2f}x {p_val:>8.4f} {'*' if significant else ' ':>5}")

            all_results.append({
                "seq_length": seq_length,
                "active_fraction": active_fraction,
                "pytorch_ms": float(pt.mean()),
                "pytorch_std": float(pt.std()),
                "triton_ms": float(tr.mean()),
                "triton_std": float(tr.std()),
                "speedup": speedup,
                "p_value": p_val,
                "significant": significant,
            })

    return all_results

# Part 5 — Plots

def plot_all(
    freeze_rates: dict[str, list[float]],
    sweep_results: list[dict],
    bench_results: list[dict],
    num_steps: int = 32,
) -> None:
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    steps = np.arange(num_steps)
    seq_lengths = sorted({r["seq_length"]      for r in bench_results})
    afs = sorted({r["active_fraction"] for r in bench_results}, reverse=True)
    fs_vals = [r["freeze_steps_front"] for r in sweep_results]
    flops_saved = [r["flops_saved"] * 100   for r in sweep_results]
    l2_vals = [r["mean_l2"]             for r in sweep_results]

    # --- [0,0] Freeze rate per regime ---
    ax1 = fig.add_subplot(gs[0, 0])
    for regime, colour in REGIME_COLOURS.items():
        ax1.plot(steps, freeze_rates[regime], label=regime.capitalize(),
                 color=colour, linewidth=2)
    ax1.set_xlabel("Decoding step")
    ax1.set_ylabel("Freeze rate")
    ax1.set_title("Freeze rate per positional regime\n(mirrors Fig. 1 of policy report)")
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # --- [0,1] FLOPs saved vs. freeze_steps_front ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(fs_vals, flops_saved, color="#4CAF50", alpha=0.8)
    ax2.set_xlabel("freeze_steps_front")
    ax2.set_ylabel("FLOPs saved (%)")
    ax2.set_title("FLOPs saved vs. freeze step\n(mirrors Fig. 4 upper of policy report)")
    ax2.set_xticks(fs_vals)
    ax2.grid(True, alpha=0.3, axis="y")

    # --- [0,2] Quality vs. efficiency tradeoff ---
    ax3 = fig.add_subplot(gs[0, 2])
    sc = ax3.scatter(flops_saved, l2_vals, c=fs_vals, cmap="plasma", s=80, zorder=3)
    for r, l2 in zip(sweep_results, l2_vals):
        ax3.annotate(
            f"fs={r['freeze_steps_front']}",
            (r["flops_saved"] * 100, l2),
            textcoords="offset points", xytext=(4, 4), fontsize=8,
        )
    ax3.set_xlabel("FLOPs saved (%)")
    ax3.set_ylabel("Mean L2 distance (cache vs. dense)")
    ax3.set_title("Quality vs. efficiency tradeoff\n(mirrors Fig. 4 of policy report)")
    ax3.grid(True, alpha=0.3)

    # --- [1,0] Speedup bar chart at 50% active ---
    FIXED_ACTIVE_FRACTION = 0.5
    ax4 = fig.add_subplot(gs[1, 0])
    sub = [r for r in bench_results if r["active_fraction"] == FIXED_ACTIVE_FRACTION]
    bar_colours = ["#4CAF50" if r["significant"] else "#9E9E9E" for r in sub]
    ax4.bar([str(s) for s in seq_lengths], [r["speedup"] for r in sub],
            color=bar_colours, alpha=0.85)
    ax4.axhline(1.0, color="black", linewidth=1, linestyle="--")  # 1.0 = no speedup
    ax4.set_xlabel("Sequence length")
    ax4.set_ylabel("Speedup vs. PyTorch baseline")
    ax4.set_title(f"Wall-clock speedup @ {int(FIXED_ACTIVE_FRACTION*100)}% active\n"
                  f"(green = p < {P_VALUE_SIGNIFICANCE})")
    ax4.grid(True, alpha=0.3, axis="y")

    # --- [1,1] Speedup heatmap ---
    ax5 = fig.add_subplot(gs[1, 1])
    matrix = np.array([
        [next(r["speedup"] for r in bench_results
              if r["seq_length"] == s and r["active_fraction"] == a)
         for s in seq_lengths]
        for a in afs
    ])
    im = ax5.imshow(
        matrix, aspect="auto", cmap="RdYlGn",
        vmin=SPEEDUP_HEATMAP_VMIN,
        vmax=max(SPEEDUP_HEATMAP_VMAX, matrix.max()),  # extend ceiling if data exceeds 2x
    )
    plt.colorbar(im, ax=ax5, label="Speedup")
    ax5.set_xticks(range(len(seq_lengths)))
    ax5.set_xticklabels([str(s) for s in seq_lengths])
    ax5.set_yticks(range(len(afs)))
    ax5.set_yticklabels([f"{int(a*100)}%" for a in afs])
    ax5.set_xlabel("Sequence length")
    ax5.set_ylabel("Active fraction")
    ax5.set_title("Speedup heatmap\n(seq length × active fraction)")

    # --- [1,2] Speedup + significance table ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    table_data = []
    cell_colours  = []
    for a in afs:
        row, crow = [], []
        for s in seq_lengths:
            r = next(x for x in bench_results
                     if x["seq_length"] == s and x["active_fraction"] == a)
            row.append(f"{r['speedup']:.2f}x\np={r['p_value']:.3f}")
            crow.append("#C8E6C9" if r["significant"] else "#FFCDD2")
        table_data.append(row)
        cell_colours.append(crow)

    tbl = ax6.table(
        cellText=table_data,
        rowLabels=[f"{int(a*100)}% active" for a in afs],
        colLabels=[str(s) for s in seq_lengths],
        cellColours=cell_colours,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.8)
    ax6.set_title(
        f"Speedup & significance\n(green = p < {P_VALUE_SIGNIFICANCE}, red = not significant)",
        pad=12,
    )

    plt.suptitle(
        "Fast-dLLM V2 — Token Skipping Policy: Kernel Evaluation",
        fontsize=14, fontweight="bold",
        y=1.01, # push above the subplot grid to avoid overlap with top row titles
    )
    plt.savefig("policy_results.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved → policy_results.png")
    plt.show()

# Entry point

if __name__ == "__main__":
    torch.manual_seed(42)

    # 1 — Correctness (must pass before any characterization is meaningful)
    correctness_sim = test_cosine_similarity_correctness()
    correctness_cache = test_cache_update_correctness()

    if not (correctness_sim and correctness_cache):
        print("Correctness tests FAILED — fix kernels before proceeding.")
        raise SystemExit(1)

    # 2 — Freeze-rate characterization
    print("=" * 64)
    print("Part 2: Freeze-rate characterization across decoding steps")
    print("=" * 64)
    freeze_rates = run_freeze_characterization(
        num_steps=32, seq_length=64, dimension_size=256, block_length=32
    )
    for regime, rates in freeze_rates.items():
        print(f"  {regime:6s} — mean freeze rate: {np.mean(rates) * 100:.1f}%")
    print()

    # 3 — FLOPs / quality sweep
    print("=" * 64)
    print("Part 3: FLOPs saved vs. quality (freeze_steps sweep)")
    print("=" * 64)
    print(f"{'fs_front':>8} {'fs_mid':>7} {'flops_saved%':>13} {'mean_L2':>10}")
    print(" " + "-" * 42)
    sweep_results = run_flops_quality_sweep(
        num_steps=32, seq_length=64, dimension_size=256, block_length=32
    )
    for r in sweep_results:
        print(f"{r['freeze_steps_front']:>8} {r['freeze_steps_mid']:>7} "
              f"{r['flops_saved'] * 100:>12.1f}% {r['mean_l2']:>10.4f}")
    print()

    # 4 — Wall-clock benchmark
    bench_results = run_benchmark(
        seq_lengths = (256, 512, 1024, 2048),
        active_fractions = (0.9, 0.7, 0.5, 0.3),
        dimension_size = 256,
        block_length = 32,
    )
    print()

    # 5 — Plots
    plot_all(freeze_rates, sweep_results, bench_results, num_steps=32)