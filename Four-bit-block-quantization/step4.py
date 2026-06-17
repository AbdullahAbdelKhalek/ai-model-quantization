"""
Step 4 - 4-bit block quantization: ~8x smaller weights.

We push from 8 bits to 4 bits per weight. 4 bits can only name 16 levels (0-15), so the
approximation is coarser, and now a single shared scale per channel is no longer enough.
The fix is BLOCK quantization: chop each weight matrix into small blocks (here 64 values), and
give every block its own scale, fitted to that block's largest value. Small blocks keep the 16
levels well-used, so quality stays reasonable even at 4 bits.

Two 4-bit values are packed into a single byte (a byte holds two "nibbles"), which is what makes
the storage actually drop, since PyTorch has no native 4-bit type.

The memory drops nicely, but watch the perplexity: it EXPLODES (from ~29 to ~113). Uniform 4-bit
is simply too coarse for an LLM. This is the cliff the QLoRA paper was built to fix. Step 5
(NF4 / NormalFloat) shows the fix, which costs no extra memory at all.

Run:  python step4.py
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


def quantize_4bit(weight: torch.Tensor, group_size: int):
    """Flatten the weight, quantize each block of `group_size` values to 4 bits, pack 2 per byte.

    Returns (packed_uint8, block_scales_float32, num_real_values).
    """
    flat = weight.flatten()
    n = flat.numel()
    pad = (-n) % group_size  # pad up to a whole number of blocks
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])

    blocks = flat.view(-1, group_size)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)  # one scale per block
    # Map [-scale, +scale] -> [0, 15].
    codes = ((blocks + scale) / (2 * scale) * 15).round().clamp(0, 15).to(torch.uint8)

    # Pack two consecutive 4-bit codes into one byte: low nibble | (high nibble << 4).
    codes = codes.reshape(-1, 2)
    packed = (codes[:, 0] | (codes[:, 1] << 4)).to(torch.uint8)
    return packed, scale.squeeze(1).to(torch.float32), n


def dequantize_4bit(packed, scale, n, group_size):
    """Reverse of quantize_4bit -> a 1D float32 tensor of length n."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.stack([low, high], dim=1).reshape(-1)  # unpack the two nibbles
    codes = codes.view(-1, group_size).float()
    s = scale.unsqueeze(1)
    x = codes / 15 * 2 * s - s  # invert the [-scale, +scale] -> [0, 15] mapping
    return x.reshape(-1)[:n]


class Quant4BitConv1D(torch.nn.Module):
    """GPT-2 Conv1D with its weight stored as packed 4-bit blocks + float32 block scales."""

    def __init__(self, conv1d: Conv1D, group_size: int = GROUP_SIZE):
        super().__init__()
        self.nf = conv1d.nf
        self.shape = tuple(conv1d.weight.shape)  # (in_features, out_features)
        self.group_size = group_size
        self.n = conv1d.weight.numel()

        packed, scale, _ = quantize_4bit(conv1d.weight.data, group_size)
        self.register_buffer("packed", packed)          # 0.5 bytes per weight
        self.register_buffer("scale", scale)            # float32, one per 64-value block
        self.register_buffer("bias", conv1d.bias.data.to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_4bit(self.packed, self.scale, self.n, self.group_size).view(self.shape)
        size_out = x.size()[:-1] + (self.nf,)
        out = torch.addmm(self.bias, x.reshape(-1, x.size(-1)), weight.to(x.dtype))
        return out.view(size_out)


def quantize_block_weights(model: torch.nn.Module):
    for module in model.modules():
        for child_name, child in module.named_children():
            if isinstance(child, Conv1D):
                setattr(module, child_name, Quant4BitConv1D(child))
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

    print(f"Technique:    4-bit block quantization (block size {GROUP_SIZE}), float32 scales")
    print(f"Memory:       {before:7.1f} MB  ->  {after:7.1f} MB   ({before / after:.2f}x smaller)")
    print(f"Perplexity:   {ppl:7.2f}   (baseline was 28.92, lower is better)")
    record_result(4, "4-bit uniform (naive)", after, ppl)


if __name__ == "__main__":
    main()
