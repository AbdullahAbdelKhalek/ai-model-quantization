"""
make_chart.py - plot the perplexity-vs-memory figure from the results the steps recorded.

As you run step1 through step6, each one saves its measured memory and perplexity to
results/results.json. This script just reads that file and plots it, so it's instant and never
re-runs the models. The numbers are exactly what you saw on screen while running the steps.

    pip install matplotlib
    python make_chart.py

Output: results/perplexity_vs_memory.png
"""

import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# Presentation only (color + label offset in points), keyed by step number. The measured numbers
# come from results/results.json, which the steps write.
STYLE = {
    1: ("#1f77b4", (8, 8)),
    2: ("#1f77b4", (8, 8)),
    3: ("#1f77b4", (8, -14)),
    4: ("#d62728", (10, 0)),
    5: ("#2ca02c", (10, 14)),
    6: ("#2ca02c", (-30, -22)),
}


def main():
    path = os.path.join(HERE, "results", "results.json")
    if not os.path.exists(path):
        print("No results yet. Run the six steps first (each one records its numbers), then re-run this:")
        print("  python Baseline-and-measuring-memory/step1.py   (and step2 ... step6)")
        return

    with open(path) as f:
        data = json.load(f)

    missing = [s for s in range(1, 7) if str(s) not in data]
    if missing:
        print(f"Missing results for step(s): {', '.join(map(str, missing))}.")
        print("Run those step scripts first (each records its numbers), then re-run this.")
        return

    # Assemble points in step order from the recorded data.
    points = []
    for step in range(1, 7):
        entry = data[str(step)]
        color, offset = STYLE[step]
        points.append((entry["label"], entry["memory_mb"], entry["perplexity"], color, offset))

    baseline_ppl = points[0][2]  # fp32 perplexity = the "full quality" reference line

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(baseline_ppl, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(478, baseline_ppl + 2, "full-precision quality", color="gray", fontsize=9, ha="right")

    for label, memory, perplexity, color, (dx, dy) in points:
        ax.scatter(memory, perplexity, s=90, color=color, zorder=3)
        ax.annotate(label, (memory, perplexity), textcoords="offset points", xytext=(dx, dy), fontsize=9)

    # Callouts point at the actual measured 4-bit-uniform (step 4) and NF4 (step 5) points.
    uniform = points[3]
    nf4 = points[4]
    ax.annotate(
        "same size, quality restored",
        xy=(nf4[1], nf4[2]), xytext=(250, 70),
        arrowprops=dict(arrowstyle="->", color="#2ca02c"), color="#2ca02c", fontsize=10,
    )
    ax.annotate(
        "naive 4-bit breaks the model",
        xy=(uniform[1], uniform[2]), xytext=(250, 105),
        arrowprops=dict(arrowstyle="->", color="#d62728"), color="#d62728", fontsize=10,
    )

    ax.set_xlabel("Model memory (MB), smaller is better")
    ax.set_ylabel("Perplexity, lower is better")
    ax.set_title("Compressing AI model: memory vs. quality")
    ax.invert_xaxis()  # so the model gets "smaller" as you read left to right
    ax.grid(True, alpha=0.3)

    out = os.path.join(HERE, "results", "perplexity_vs_memory.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
