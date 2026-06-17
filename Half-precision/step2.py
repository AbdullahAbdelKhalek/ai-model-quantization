"""
Step 2 - Half precision (float16): the easiest 2x.

A float32 weight uses 32 bits (4 bytes). A float16 weight uses 16 bits (2 bytes), the same
number written with fewer digits. Storing every transformer weight in float16 instead of float32
halves the memory those layers use, with almost no change in quality.

We only touch the transformer block weights (GPT-2's `Conv1D` layers). The token embedding and
the normalization layers are left in float32 on purpose, because quantizing them costs quality for
little memory gain, which is what production quantizers (bitsandbytes, QLoRA) also do.

One practical detail: CPUs are slow at float16 math, so we STORE the weight in float16 but cast
it back up to float32 just for the matmul. The saving is in storage, not compute.

Run:  python step2.py
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


class HalfConv1D(torch.nn.Module):
    """A drop-in replacement for GPT-2's Conv1D that stores its weight in float16.

    GPT-2's Conv1D computes `x @ weight + bias`, with weight shaped (in_features, out_features).
    We keep that exact math, but hold the weight as float16 to halve its storage.
    """

    def __init__(self, conv1d: Conv1D):
        super().__init__()
        self.nf = conv1d.nf  # number of output features
        # Store the weight in float16 (the 2x saving). Keep bias in float32 (tiny, helps quality).
        self.register_buffer("weight", conv1d.weight.data.to(torch.float16))
        self.register_buffer("bias", conv1d.bias.data.to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(x.dtype)  # cast 16-bit storage back up to 32-bit for the matmul
        size_out = x.size()[:-1] + (self.nf,)
        out = torch.addmm(self.bias, x.reshape(-1, x.size(-1)), weight)
        return out.view(size_out)


def quantize_block_weights(model: torch.nn.Module):
    """Replace every Conv1D in the transformer blocks with our HalfConv1D."""
    for module in model.modules():
        for child_name, child in module.named_children():
            if isinstance(child, Conv1D):
                setattr(module, child_name, HalfConv1D(child))
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

    print(f"Technique:    float16 (half precision) on transformer block weights")
    print(f"Memory:       {before:7.1f} MB  ->  {after:7.1f} MB   ({before / after:.2f}x smaller)")
    print(f"Perplexity:   {ppl:7.2f}   (baseline was 28.92, lower is better)")
    record_result(2, "float16", after, ppl)


if __name__ == "__main__":
    main()
