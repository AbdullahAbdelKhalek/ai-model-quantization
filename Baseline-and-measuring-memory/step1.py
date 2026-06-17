"""
Step 1 - Baseline: how big is an LLM, and how do we measure "memory" and "quality"?

Before we shrink anything, we need a starting line. This script loads a small open LLM
(GPT-2, 124M parameters) at its normal full precision (float32) and reports the two numbers
that every later step will try to improve:

  - MEMORY  : the total bytes used by the model's weights (parameters + buffers).
  - QUALITY : perplexity on a fixed passage of text. Perplexity is "how surprised the model
              is by real language"; LOWER is better. A model that has been damaged by
              over-aggressive compression will become more surprised, and perplexity rises.

Run:  python step1.py
(The first run downloads GPT-2, ~500 MB, from Hugging Face. No API key needed.)
"""

import logging
import os
import warnings

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Quiet the Hugging Face / transformers startup messages so the output matches this README.
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

MODEL_NAME = "gpt2"  # the classic 124M-parameter model

# A fixed passage we score every model on, so the quality number is comparable across all steps.
# (Original text written for this project, public domain.)
EVAL_TEXT = (
    "Memory has quietly become one of the most important resources in modern computing. "
    "Every time a large model is trained or run, its weights must be held somewhere fast "
    "enough to reach, and that space is neither free nor unlimited. As models have grown, "
    "the gap between what they need and what a single machine can offer has widened. "
    "Engineers now spend a great deal of effort finding ways to store the same knowledge in "
    "less space, so that more can be done with the hardware that already exists."
)


def model_memory_mb(model: torch.nn.Module) -> float:
    """Total size of all weights (parameters + buffers) in megabytes.

    element_size() is the number of bytes one number takes (4 for float32, 2 for float16,
    1 for int8). numel() is how many numbers a tensor holds. Multiply and sum over everything.
    """
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_bytes / (1024**2)


def perplexity(model: torch.nn.Module, tokenizer, text: str, device: str) -> float:
    """exp(average cross-entropy loss) over the text. Lower is better.

    Passing labels=input_ids tells the model to score how well it predicts each next token;
    it returns the average cross-entropy. Exponentiating turns that into perplexity.
    """
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

    # Load the model and its tokenizer (the tool that turns text into token ids and back).
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)

    num_params = sum(p.numel() for p in model.parameters())
    memory = model_memory_mb(model)
    ppl = perplexity(model, tokenizer, EVAL_TEXT, device)

    print(f"Model:        {MODEL_NAME}")
    print(f"Device:       {device}")
    print(f"Parameters:   {num_params / 1e6:.1f} M")
    print(f"Memory (fp32):{memory:8.1f} MB")
    print(f"Perplexity:   {ppl:8.2f}  (lower is better)")
    record_result(1, "fp32 baseline", memory, ppl)

    # A quick greedy generation, so you can see it is a real, working language model.
    prompt = "In the future, artificial intelligence will"
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,  # passes both input_ids and attention_mask
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(f"\nPrompt:  {prompt}")
    print(f"Sample:  {tokenizer.decode(generated[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
