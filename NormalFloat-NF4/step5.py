"""
Step 5 - NormalFloat (NF4): make 4-bit actually work.

Step 4 showed the problem: naive uniform 4-bit spreads its 16 levels EVENLY, but neural network
weights are not spread evenly. They cluster tightly around zero in a bell curve, with a few
large outliers. Uniform levels waste most of their resolution on the empty tails and leave the
crowded middle too coarse, so the model's quality collapses (perplexity ~113).

The QLoRA paper's fix is NF4 ("NormalFloat"): instead of evenly spaced levels, place the 16
levels at the QUANTILES of a normal distribution (dense near zero, sparse in the tails), so
resolution lands where the weights actually are. Same 4 bits, same memory. The only change is
WHICH 16 values the codes map to.

The result is striking: 4-bit quality jumps back to essentially the full-precision baseline.

Run:  python step5.py
"""

import logging
import os
import warnings

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D

# Quiet the Hugging Face / transformers startup messages so the output matches this README.
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

MODEL_NAME = "gpt2"
GROUP_SIZE = 64
EVAL_TEXT = (
    "Memory has quietly become one of the most important resources in modern computing. "
    "Every time a large model is trained or run, its weights must be held somewhere fast "
    "enough to reach, and that space is neither free nor unlimited. As models have grown, "
    "the gap between what they need and what a single machine can offer has widened. "
    "Engineers now spend a great deal of effort finding ways to store the same knowledge in "
    "less space, so that more can be done with the hardware that already exists."
)

# The NF4 codebook from the QLoRA paper: 16 levels in [-1, 1], placed at normal-distribution
# quantiles (note how tightly they bunch near 0 compared to evenly spaced uniform levels).
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.0910500, 0.0,
    0.0795803, 0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.5626170, 0.7229568, 1.0])
# Midpoints between consecutive levels = decision boundaries for "snap to nearest level".
NF4_BOUNDARIES = (NF4_LEVELS[:-1] + NF4_LEVELS[1:]) / 2


def quantize_nf4(weight: torch.Tensor, group_size: int):
    """Block-quantize a weight to 4-bit NF4 codes. Returns (packed_uint8, block_scales_fp32, n)."""
    flat = weight.flatten()
    n = flat.numel()
    pad = (-n) % group_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, group_size)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    normalized = (blocks / scale).clamp(-1, 1)
    # Snap each normalized value to the nearest NF4 level -> an index 0..15.
    codes = torch.bucketize(normalized, NF4_BOUNDARIES).to(torch.uint8)
    # Pack two 4-bit codes per byte.
    codes = codes.reshape(-1, 2)
    packed = (codes[:, 0] | (codes[:, 1] << 4)).to(torch.uint8)
    return packed, scale.squeeze(1).to(torch.float32), n


def dequantize_nf4(packed, scale, n, group_size):
    """Reverse: unpack codes, look each up in the NF4 table, rescale. Returns 1D float32."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.stack([low, high], dim=1).reshape(-1).long()
    values = NF4_LEVELS.to(packed.device)[codes].view(-1, group_size)  # index -> level value
    x = values * scale.unsqueeze(1)
    return x.reshape(-1)[:n]


class NF4Conv1D(torch.nn.Module):
    """GPT-2 Conv1D with its weight stored as 4-bit NF4 codes + float32 block scales."""

    def __init__(self, conv1d: Conv1D, group_size: int = GROUP_SIZE):
        super().__init__()
        self.nf = conv1d.nf
        self.shape = tuple(conv1d.weight.shape)
        self.group_size = group_size
        self.n = conv1d.weight.numel()
        packed, scale, _ = quantize_nf4(conv1d.weight.data, group_size)
        self.register_buffer("packed", packed)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", conv1d.bias.data.to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_nf4(self.packed, self.scale, self.n, self.group_size).view(self.shape)
        size_out = x.size()[:-1] + (self.nf,)
        out = torch.addmm(self.bias, x.reshape(-1, x.size(-1)), weight.to(x.dtype))
        return out.view(size_out)


def quantize_block_weights(model: torch.nn.Module):
    for module in model.modules():
        for child_name, child in module.named_children():
            if isinstance(child, Conv1D):
                setattr(module, child_name, NF4Conv1D(child))
    return model


def model_memory_mb(model: torch.nn.Module) -> float:
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / (1024**2)


def perplexity(model, tokenizer, text, device) -> float:
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss
    return float(torch.exp(loss))


def record_result(step: int, label: str, memory_mb: float, perplexity_value: float):
    """Save this step's numbers to results/results.json so make_chart.py can plot the whole series
    without re-running anything. The steps produce the data; the chart just reads it."""
    import json

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "results.json")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data[str(step)] = {"label": label, "memory_mb": round(memory_mb, 1), "perplexity": round(perplexity_value, 2)}
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)

    before = model_memory_mb(model)
    quantize_block_weights(model)
    after = model_memory_mb(model)
    ppl = perplexity(model, tokenizer, EVAL_TEXT, device)

    print(f"Technique:    4-bit NF4 / NormalFloat (block size {GROUP_SIZE})")
    print(f"Memory:       {before:7.1f} MB  ->  {after:7.1f} MB   ({before / after:.2f}x smaller)")
    print(f"Perplexity:   {ppl:7.2f}   (uniform 4-bit was 112.74; baseline 28.92)")
    record_result(5, "NF4 (paper)", after, ppl)


if __name__ == "__main__":
    main()
