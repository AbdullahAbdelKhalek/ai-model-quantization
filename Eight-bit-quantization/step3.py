"""
Step 3 - 8-bit quantization (int8): 4x smaller.

float16 just shortened each number. int8 goes further: instead of storing a real number, we
store a small integer from -127 to 127 (one byte) that names one of 256 evenly spaced "levels",
plus a single scale factor that says how big one level is.

  stored_int = round(weight / scale)        # an integer in [-127, 127]
  weight ~= stored_int * scale              # reconstructed on the way out

If one scale covered the whole matrix, a single large weight would stretch the levels and
everything else would round to zero. So we use ONE SCALE PER OUTPUT CHANNEL (per column of the
weight), and each column gets a scale fitted to its own largest value. This is "per-channel"
quantization, and it is what makes int8 work well in practice.

Run:  python step3.py
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
EVAL_TEXT = (
    "Memory has quietly become one of the most important resources in modern computing. "
    "Every time a large model is trained or run, its weights must be held somewhere fast "
    "enough to reach, and that space is neither free nor unlimited. As models have grown, "
    "the gap between what they need and what a single machine can offer has widened. "
    "Engineers now spend a great deal of effort finding ways to store the same knowledge in "
    "less space, so that more can be done with the hardware that already exists."
)


class Int8Conv1D(torch.nn.Module):
    """GPT-2 Conv1D with its weight stored as int8 + one float16 scale per output channel."""

    def __init__(self, conv1d: Conv1D):
        super().__init__()
        self.nf = conv1d.nf
        weight = conv1d.weight.data  # shape (in_features, out_features)

        # One scale per output channel (per column). amax over the input dimension (dim=0).
        scale = weight.abs().amax(dim=0, keepdim=True) / 127.0  # (1, out_features)
        scale = scale.clamp_min(1e-8)
        weight_int8 = (weight / scale).round().clamp(-127, 127).to(torch.int8)

        self.register_buffer("weight_int8", weight_int8)            # 1 byte per weight
        self.register_buffer("scale", scale.to(torch.float16))      # tiny: one per column
        self.register_buffer("bias", conv1d.bias.data.to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reconstruct the float weight: integer level * its column's scale.
        weight = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype)
        size_out = x.size()[:-1] + (self.nf,)
        out = torch.addmm(self.bias, x.reshape(-1, x.size(-1)), weight)
        return out.view(size_out)


def quantize_block_weights(model: torch.nn.Module):
    for module in model.modules():
        for child_name, child in module.named_children():
            if isinstance(child, Conv1D):
                setattr(module, child_name, Int8Conv1D(child))
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

    print(f"Technique:    int8 (8-bit), one scale per output channel")
    print(f"Memory:       {before:7.1f} MB  ->  {after:7.1f} MB   ({before / after:.2f}x smaller)")
    print(f"Perplexity:   {ppl:7.2f}   (baseline was 28.92, lower is better)")
    record_result(3, "int8", after, ppl)


if __name__ == "__main__":
    main()
