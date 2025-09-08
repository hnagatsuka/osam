# OSAM — One‑Shot Algebraic Map

**Constant‑Memory Sequence Modeling via Cross TensorSketch and Value Retrieval**

[![License: Apache‑2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](#license)

> **Status (v0.1):** Experiments are **TBD**. A formal evaluation suite will land in **v0.2**. Early feedback, ports, and benchmarks are very welcome.

---

## Overview

OSAM is a single‑file reference implementation (`osam_minidemo.py`) of a constant‑memory sequence model. Instead of attention or recurrent SSM state, OSAM uses:

* **D=2 cross TensorSketch features** of (query × prefix) with a **learnable relative phase**.
* A **constant‑memory value path** that linearly binds values to sketched keys and reads them back by contracting with the query sketch.
* Optional **frequency‑domain cache** for the prefix sketch (FFT/IFFT), **compute thinning** (correlation every *k* tokens), and **telemetry** for stability.

The model runs on CPU or GPU (PyTorch), supports **AMP** (bf16/fp16), external corpora, checkpoints, and an **interactive REPL**.

> If OSAM resonates with you, please use it and spread the word. Benchmarks and ports are very welcome via Issues/PRs; great examples will be highlighted in the README.

---

## Why OSAM?

* **Constant memory in sequence length**: persistent state size is $K:2m$ and $H:2dm$ per batch element—independent of $T$.
* **Algebraic readout**: cross‑correlation on sketches via FFT provides a compact retrieval signal; value memory adds a direct content path.
* **Small, auditable code**: single file, explicit state handling (detach on read, fresh tensors on write), and detailed logs.

---

## Quick Start

### Requirements

* Python **3.10+**
* **PyTorch 2.x** (CUDA optional). Install per the [official instructions](https://pytorch.org/get-started/locally/).

### Clone

```bash
git clone https://github.com/hnagatsuka/osam
cd osam
# (optional) python -m venv .venv && source .venv/bin/activate
```

### Train on built‑in toy corpus

```bash
python osam_minidemo.py \
  --steps 3000 --batch 64 --seq 128 \
  --d 128 --m 4096 --kupdate 4 \
  --freq_cache --dropout 0.1 --label_smoothing 0.05 \
  --save osam_demo.pt
```

### Train on a periodic “needle” corpus

```bash
python osam_minidemo.py --dataset needle \
  --steps 3000 --batch 64 --seq 128 --d 128 --m 4096
```

### Train on your own text

```bash
python osam_minidemo.py --text_path ./corpus.txt \
  --steps 3000 --batch 64 --seq 128 --d 128 --m 4096 \
  --save osam_corpus.pt
```

### Inference from checkpoint

```bash
python osam_minidemo.py --no_train --load osam_demo.pt \
  --prompt "hello " --max_new_tokens 128 --temp 0.9 --topk 40 --topp 0.9
```

### Interactive REPL

```bash
python osam_minidemo.py --no_train --load osam_demo.pt --interactive \
  --temp 0.9 --topk 40 --topp 0.95
```

---

## Concept in 30 Seconds

* **Sketching.** CountSketch maps $\mathbb{R}^d\to\mathbb{R}^m$ with fixed index/sign hashes. We sketch the **phased** query/key into **Re/Im** channels.
* **Cross feature.** $\phi_t = \operatorname{IFFT}(\operatorname{FFT}(Q_t)\odot\operatorname{FFT}(K_{t-1})) / \sqrt{t+1}$. `--kupdate k` computes this every $k$ tokens.
* **Value memory.** $H_t = H_{t-1} + v_t\otimes S(\tilde{k}_t)$ (Re/Im stacks). Read by contracting $H$ with $Q_t$, then a small MLP.
* **Output.** Gated residual + LayerNorm combine cross‑feature and value‑read to produce the next hidden.

---

## CLI Reference (selected)

| Flag                                  |       Default | Meaning                                          |
| ------------------------------------- | ------------: | ------------------------------------------------ |
| `--steps`                             |           600 | Training steps                                   |
| `--batch`                             |            32 | Batch size                                       |
| `--seq`                               |            64 | Sequence length (chars)                          |
| `--d`                                 |            64 | Model width $d$                                  |
| `--m`                                 |          2048 | Sketch size $m$                                  |
| `--hidden`                            |           128 | Readout MLP hidden size                          |
| `--lr` / `--wd`                       |   3e-3 / 1e-2 | AdamW LR / weight decay                          |
| `--clip`                              |           1.0 | Grad‑norm clip                                   |
| `--seed`                              |          1337 | RNG seed                                         |
| `--no_valpath`                        |           off | Disable value‑retrieval path                     |
| `--freeze_phase`                      |           off | Freeze phase params ($\theta_q,\theta_k,\omega$) |
| `--corrupt`                           |           0.0 | Random input corruption prob                     |
| `--dropout`                           |           0.0 | Dropout in readout/value MLP                     |
| `--label_smoothing`                   |           0.0 | Cross‑entropy label smoothing                    |
| `--dataset`                           |     `default` | `default` or `needle`                            |
| `--text_path`                         |             — | External UTF‑8 text file path                    |
| `--kupdate`                           |             1 | Compute cross‑corr every *k* tokens              |
| `--freq_cache`                        |           off | Cache K in frequency domain                      |
| `--amp`                               |        `none` | `none`, `bf16`, or `fp16`                        |
| `--telemetry_stride`                  |             1 | Collect telemetry every N computed steps         |
| `--cufft_max_plans`                   |             4 | cuFFT plan cache capacity (CUDA)                 |
| `--no_warmup_plans`                   |           off | Skip cuFFT plan warmup                           |
| `--save` / `--load`                   |             — | Checkpoint path (train / infer)                  |
| `--no_train`                          |           off | Skip training; run demo/inference                |
| `--prompt`                            |             — | Prompt for generation                            |
| `--max_new_tokens`                    |            64 | Generation length                                |
| `--temp`/`--topk`/`--topp`/`--greedy` | 1.0/0/0.0/off | Sampling controls                                |
| `--interactive`                       |           off | REPL mode                                        |

> **LR buckets.** Phase/gate parameters use **10× LR** relative to the base params.

---

## Telemetry & Logs

* Prints: `loss`, `ema`, `ppl≈`, `gnorm`, `toks/s`, and telemetry (`phi_norm`, `s1_norm`, gates, phase params) plus a short greedy sample at intervals.
* Summary reports averaged telemetry and **avg\_toks/s** at the end of training.
* `s1_norm` is defined as `||K|| + ||H||` aggregated over Re/Im channels.

---

## Reproducibility Notes

* **Determinism:** set `--seed`. Exact reproducibility can still vary across hardware/CUDA drivers.
* **AMP:** `--amp bf16|fp16`. For fp16, gradient scaling is enabled automatically.
* **FFT planning:** tune `--cufft_max_plans`, optionally `--no_warmup_plans` (CUDA only).

---

## Roadmap

* v0.2: evaluation suite (PPL/BPC, toks/s), ablations (`kupdate`, `freq_cache`, value path, phase freeze, corruption), and subword tokenizer experiments.
* Multi‑block stacks and lightweight stacking strategies.

---

## FAQ

**Is OSAM a drop‑in replacement for attention?**  Not directly; it’s an algebraic mechanism with different inductive biases.

**Does it work on subword tokens?**  The demo is character‑level. Subword is planned; PRs welcome.

**How big can `m` be?**  Memory scales with `d*m`. Start small, increase cautiously.

**Why constant memory if `H` is `d×m`?**  `H` is constant in sequence length `T` (per batch element), unlike attention caches.

---

## Citation

If you use OSAM in academic work, please cite the preprint:

> Hideaki Nagatsuka. *One‑Shot Algebraic Map (OSAM): Constant‑Memory Sequence Modeling via Cross TensorSketch and Value Retrieval*. GitHub preprint, 2025. Repository: [https://github.com/hnagatsuka/osam](https://github.com/hnagatsuka/osam)

---

## Contributing

Issues and PRs are welcome. Please include environment details (Python/PyTorch/CUDA), commands, and logs. Ports to tokenized datasets or alternative sketch backends are especially appreciated.

---

## License

Licensed under the **Apache License, Version 2.0**. See `LICENSE` for details.
