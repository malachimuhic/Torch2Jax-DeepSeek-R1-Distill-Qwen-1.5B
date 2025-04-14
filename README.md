
<div align="center">

  <img src="assets/deepseekjax.png" alt="A jax logo style image of a whale." width="200" height="auto" />
  <h1>Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B</h1>
  
  <p>
    Flax (JAX) implementation of DeepSeek-R1-Distill-Qwen-1.5B with weights ported from Hugging Face.
  </p>
  
  
<!-- Badges -->
<p>
  <a href="https://github.com/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B/contributors">
    <img src="https://img.shields.io/github/contributors/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B" alt="last update" />
  </a>
  <a href="https://github.com/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B/network/members">
    <img src="https://img.shields.io/github/forks/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B" alt="forks" />
  </a>
  <a href="https://github.com/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B/stargazers">
    <img src="https://img.shields.io/github/stars/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B" alt="stars" />
  </a>
  <a href="https://github.com/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B/issues/">
    <img src="https://img.shields.io/github/issues/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B" alt="open issues" />
  </a>
  <a href="https://github.com/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B.svg" alt="license" />
  </a>
</p>
   
<h4>
    <a href="https://github.com/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B/issues/">Report Bug</a>
  <span> · </span>
    <a href="https://github.com/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B/issues/">Request Feature</a>
    <span> · </span>
    <a href="https://colab.research.google.com/drive/1jJaAARwbsFeV5hZoffrNwFhc8i2I-7ji?usp=sharing">Colab</a>
  </h4>
</div>



## Overview

Colab: https://colab.research.google.com/drive/1amicLNKmS-gRxhhg4sTbSCO_sKwgnllZ?usp=sharing

This repository provides both Flax (JAX) and PyTorch implementations of the DeepSeek-R1-Distill-Qwen-1.5B model. It includes:

- **Inference [QUICKSTART]**:
    - `inference.ipynb`: Contains a quickstart script to download and convert params from torch to flax, load model and perform text generation.

- **Flax Implementations**:  
    - `model_flax.py`: The Flax implementation.  

- **PyTorch Implementation**:  
    - `model_torch.py`: A reference implementation in PyTorch.

- **Conversion Script**:  
    - `torch_to_flax.py`: A utility to convert a PyTorch checkpoint (state dictionary) into Flax parameters.

## System Requirements
### Single GPU
16GB VRAM on the GPU + 64GB RAM (this can be swap)

### Multi-Device
Runs sharded on v2-8 TPU on Google Colab. 

---

# Malachi's Changes
### Project Objective
To build infrastructure that allows flexible extension and fine-tuning of transformer models through LoRA, a lightweight and parameter-efficient method for training.

### What's Included
*  A working transformer model (Qwen2-based).
*  LoRA support added to key components (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj).
*  Global LoRA toggle (ENABLE_LORA) and per-instance toggles for fine control.
*  Torch-native
*  Ready to run in Google Colab for testing and experimentation.

Google Colab: https://drive.google.com/file/d/1p3zRZDR38YMAVMShNtO4kAf0bkOKEsgd/view?usp=sharing
## LoRA: Low-Rank Adaptation Overview

LoRA replaces large weight updates with small, trainable low-rank matrices (`A` and `B`) that are added to existing model layers. This drastically reduces the number of parameters needed during fine-tuning while maintaining performance.

> Learn more in the original paper: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

---

## Where LoRA Was Applied

LoRA was integrated into **linear projection layers** within:

| Module         | Layers Modified                         | Reason                            |
|----------------|------------------------------------------|-----------------------------------|
| Self-Attention | `q_proj`, `k_proj`, `v_proj`, `o_proj`   | High impact, core to attention mechanism |
| MLP Block      | `gate_proj`, `up_proj`                   | Common for adaptation in literature |
| Output Head    | `lm_head` (LoRA optional, currently off) | Can be optionally enabled if needed |

The integration wraps these layers in a `LoRALinear` module.

---

**Global Toggle**  
   Located at the top of `model_torch.py`:
   ```python
   ENABLE_LORA = True  # Set to False to disable LoRA globally
```

---

# LoRA Integration: Qwen2 Transformer with DeepSeek 1.5B Weights

This document explains how and why **LoRA (Low-Rank Adaptation)** was integrated into a custom Qwen2 transformer model. It covers background concepts, implementation design, code modifications, trade-offs, and next steps.

---

## What is LoRA?

**Low-Rank Adaptation (LoRA)** is a parameter-efficient fine-tuning technique that:

- **Freezes pretrained weights** and injects small, trainable low-rank matrices.
- **Reduces trainable parameters** significantly (e.g., <1% vs. full fine-tuning).
- **Preserves model performance** while lowering memory, compute, and deployment cost.

### Why Not Traditional Fine-Tuning?

Traditional fine-tuning updates every parameter in the model. This is:

- Expensive (memory/GPU usage).
- Slow to train.
- Challenging to deploy.

LoRA offers a more efficient alternative, training only small, targeted updates.

### Compared to Adapters

Adapters add extra layers, introducing inference latency. LoRA avoids this by:

- Modifying existing linear layers via additive updates.
- Incurring **no runtime penalty**.
- Remaining modular and swappable like adapters.

---

## How LoRA Works

Consider a linear layer:

```
y = W x
```

Instead of fine-tuning the full weight matrix `W`, LoRA uses:

```
W' = W + ΔW
ΔW ≈ B A
```

Where:

- `A` ∈ ℝ<sup>r×d</sup> and `B` ∈ ℝ<sup>d×r</sup>
- `r` is the **rank**, typically ≪ `d`
- Only `A` and `B` are trained (e.g., 16,384 parameters vs. 1M+)

At inference:

```
y = (W + B A) x
```

LoRA doesn't change output shapes or add layers. The low-rank matrices can be **merged** into `W` for zero-latency inference.

---

## Benefits of Low-Rank Matrices

- **Reduced memory and compute** during training.
- **Faster backpropagation** and gradient updates.
- **Preserves pretrained knowledge** by freezing original weights.
- **No added inference cost** — LoRA merges into base weights.

---

## Implementation Overview

You were given a raw Qwen2-based transformer loading DeepSeek 1.5B weights. LoRA was integrated into this model by:

- Wrapping attention and MLP projections with `LoRALinear`
- Preserving pretrained weight compatibility
- Allowing toggleable LoRA support via a `use_lora` argument

---

## Integration Points & Justification

### `Qwen2Attention`
LoRA applied to:
- `q_proj`, `k_proj`, `v_proj` (standard in literature)
- Optionally `o_proj`

**Why?**
- These are core to attention mechanisms and highly expressive.
- Minimal updates here have outsized impact on model behavior.

### `Qwen2MLP`
LoRA applied to:
- `gate_proj`, `up_proj`

**Why?**
- These expand the hidden state.
- Common LoRA targets due to parameter size and importance.

**Not applied to `down_proj`** — it's smaller, less expressive.

### Toggle Support
Passed `use_lora` flag through:

- `Qwen2ForCausalLM` → `Qwen2Model` → `Qwen2DecoderLayer` → `Qwen2Attention` & `Qwen2MLP`

---

## How to Use

```python
from model_torch import Qwen2ForCausalLM, Qwen2Config

config = Qwen2Config(use_lora=True)
model = Qwen2ForCausalLM(config)
```

---

## Trade-offs and Limitations

### Benefits

- 10–100× fewer trainable parameters
- No inference latency
- Task-specific modularity
- Easy toggling and compatibility with existing weights

### Limitations

- Lower expressiveness if rank is too small
- May not work for tasks requiring full model adaptation
- Requires careful handling of pretrained weight loading
- Less confident output
- Longer inference time due to Extra Matrix Multiplication

## When to Avoid LoRA

- Tasks needing broad weight updates
- Small models (benefits are marginal)
- High-precision applications
- Real-time systems where any compute cost matters
- Poor pretrained model quality

## Use Cases

- Swap LoRA modules for task-specific behavior
- Efficient multi-task training
- Deploy modular upgrades without retraining full models
- Combine with other methods (e.g., prefix-tuning)

---

## Implementation Checklist

1. **Understand the Model**
   - Inspect `model_torch.py` and `model_flax.py`
   - Identify Q/K/V and FFN weights
   - Understand how DeepSeek weights are loaded

2. **Decide LoRA Scope**
   - Apply LoRA to attention and MLP projection layers
   - Preserve forward pass behavior and shapes

3. **Implement**
   - Create `LoRALinear` (done)
   - Inject into model with toggle support

4. **Test**
   - Confirm pretrained weights load correctly
   - Validate model with LoRA enabled/disabled
   - Compare performance

5. **Document**
   - Justify choices
   - Note trade-offs
   - Flag open questions

---

## Drawbacks of my implementation:
  - LoRA adds time at inference
  - Extra memory usage when LoRA is disabled
  - LoRA in lm_head can be used for more efficiency (although it seems like lm_head can destabilize training)
  - Lacking layer-to-layer LoRA controls
  - Could use regularization for more efficiency

## Question(s):
In your instructions you said that:

"You’ve been given a working, raw implementation of a transformer model based on the Qwen2 architecture. It loads and runs using pretrained DeepSeek 1.5B weights, and it does not rely on high-level ML libraries like Hugging Face Transformers."

In my current implementation, I load the pretrained weights using Hugging Face’s from_pretrained() method. My understanding is that this is necessary because training a model like DeepSeek 1.5B from scratch would require enormous computational resources and data — far beyond the scope of this assignment. 

Could you clarify whether using Hugging Face to load pretrained weights only (without relying on it for modeling, training, or inference logic) still aligns with the spirit of the instructions? Or would I need to pull weights from another source or format to fully comply?

## Resources

- [LoRA Paper (arXiv)](https://arxiv.org/pdf/2106.09685)
- [LoRA GitHub (Microsoft)](https://github.com/microsoft/LoRA)
- [LoRA YouTube Explanation](https://www.youtube.com/watch?v=DhRoTONcyZE)

