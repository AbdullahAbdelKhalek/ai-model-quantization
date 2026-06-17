"""
Step 6 - Double quantization: quantize the scales too (the second QLoRA trick).

After step 5 the weights are 4-bit NF4 and quality is back to baseline. But every block of 64
weights still carries its own float32 scale: 32 bits per 64 weights = 0.5 bits per weight, spent
purely on scales. On a small model that is a few megabytes; on a 65B model it is gigabytes.

The QLoRA paper's "double quantization" applies quantization a SECOND time, to the scales
themselves. We group the block scales (256 at a time) and store each as an 8-bit value relative
to a second-level float32 scale. Each scale shrinks from 32 bits to ~8 bits.

It is quantization on top of quantization: first compress the weights, then compress the
quantizer's own bookkeeping. The weights are untouched, so quality stays the same; the model
just gets a little smaller. (Modest here, large at scale.)

Run:  python step6.py
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
SCALE_BLOCK = 256
EVAL_TEXT = (
    "Memory has quietly become one of the most important resources in modern computing. "
    "Every time a large model is trained or run, its weights must be held somewhere fast "
    "enough to reach, and that space is neither free nor unlimited. As models have grown, "
    "the gap between what they need and what a single machine can offer has widened. "
    "Engineers now spend a great deal of effort finding ways to store the same knowledge in "
    "less space, so that more can be done with the hardware that already exists."
)

NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.0910500, 0.0,
    0.0795803, 0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.5626170, 0.7229568, 1.0])
NF4_BOUNDARIES = (NF4_LEVELS[:-1] + NF4_LEVELS[1:]) / 2


def quantize_nf4(weight: torch.Tensor, group_size: int):
    """4-bit NF4 block quantization (same as step 5). Returns (packed, scales_fp32, n)."""
    flat = weight.flatten()
    n = flat.numel()
    pad = (-n) % group_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, group_size)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    codes = torch.bucketize((blocks / scale).clamp(-1, 1), NF4_BOUNDARIES).to(torch.uint8)
    codes = codes.reshape(-1, 2)
    packed = (codes[:, 0] | (codes[:, 1] << 4)).to(torch.uint8)
    return packed, scale.squeeze(1).to(torch.float32), n


def dequantize_nf4(packed, scale, n, group_size):
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.stack([low, high], dim=1).reshape(-1).long()
    values = NF4_LEVELS.to(packed.device)[codes].view(-1, group_size)
    return (values * scale.unsqueeze(1)).reshape(-1)[:n]


def quantize_scales_8bit(scales: torch.Tensor, scale_block: int):
    """The 'double' part: store each block scale as 8 bits, per second-level block of 256."""
    m = scales.numel()
    pad = (-m) % scale_block
    s = torch.cat([scales, scales.new_zeros(pad)]) if pad else scales
    sb = s.view(-1, scale_block)
    second = sb.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)  # one fp32 per 256 scales
    codes = (sb / second * 255).round().clamp(0, 255).to(torch.uint8)
    return codes.reshape(-1), second.squeeze(1).to(torch.float32), m


def dequantize_scales_8bit(codes, second, m, scale_block):
    s = codes.view(-1, scale_block).float() / 255 * second.unsqueeze(1)
    return s.reshape(-1)[:m]


class DoubleQuantNF4Conv1D(torch.nn.Module):
    """GPT-2 Conv1D: 4-bit NF4 weights + 8-bit (double-quantized) block scales."""

    def __init__(self, conv1d: Conv1D, group_size: int = GROUP_SIZE, scale_block: int = SCALE_BLOCK):
        super().__init__()
        self.nf = conv1d.nf
        self.shape = tuple(conv1d.weight.shape)
        self.group_size = group_size
        self.scale_block = scale_block
        self.n = conv1d.weight.numel()

        packed, scale_fp32, _ = quantize_nf4(conv1d.weight.data, group_size)
        scale_codes, second, self.num_scales = quantize_scales_8bit(scale_fp32, scale_block)

        self.register_buffer("packed", packed)            # 4-bit NF4 weights
        self.register_buffer("scale_codes", scale_codes)  # 8-bit scales (was float32)
        self.register_buffer("second", second)            # tiny: one float32 per 256 scales
        self.register_buffer("bias", conv1d.bias.data.to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = dequantize_scales_8bit(self.scale_codes, self.second, self.num_scales, self.scale_block)
        weight = dequantize_nf4(self.packed, scale, self.n, self.group_size).view(self.shape)
        size_out = x.size()[:-1] + (self.nf,)
        out = torch.addmm(self.bias, x.reshape(-1, x.size(-1)), weight.to(x.dtype))
        return out.view(size_out)


def quantize_block_weights(model: torch.nn.Module):
    for module in model.modules():
        for child_name, child in module.named_children():
            if isinstance(child, Conv1D):
                setattr(module, child_name, DoubleQuantNF4Conv1D(child))
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

    print(f"Technique:    NF4 weights + 8-bit double-quantized scales (block {GROUP_SIZE}/{SCALE_BLOCK})")
    print(f"Memory:       {before:7.1f} MB  ->  {after:7.1f} MB   ({before / after:.2f}x smaller)")
    print(f"Perplexity:   {ppl:7.2f}   (baseline 28.92, NF4 without DQ ~28.94)")
    record_result(6, "NF4 + double-quant", after, ppl)


if __name__ == "__main__":
    main()
