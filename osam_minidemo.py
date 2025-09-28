#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -----------------------------------------------------------------------------
"""
One-Shot Algebraic Map (OSAM) — Cross TensorSketch + Constant-Memory Value Readout

Features (no regressions from prior versions + additions):
  • D=2 polynomial cross feature via CountSketch of (query × prefix) with relative phase
  • Constant-memory Value readout path ("V-path"; toggleable)
  • Learnable phase (theta_q, theta_k, omega) with optional freezing
  • Input corruption (--corrupt)
  • Telemetry + SUMMARY (avg_toks/s computed as total_tokens / total_time)
  • Dropout (--dropout), Label Smoothing (--label_smoothing)
  • Dataset switch (--dataset {default, needle})
  • Compute thinning every k steps (--kupdate)
  • Frequency-domain cache for K (--freq_cache)
  • Checkpoint save/load (--save / --load)
  • Inference demo (REPL / one-shot) (--interactive / --prompt --temp --topk --topp --max_new_tokens)
  • External corpus loader (--text_path <file.txt>)
  • Memory-friendly: small cuFFT plan cache, optional warmup, safe views/contiguous.
    Autograd-safe: prefix states are treated as buffers (no grad through K/H/FK).

Deps: Python 3.10+, PyTorch (CPU/GPU)
"""
from __future__ import annotations
import math
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

# ------------------------------ Utilities ------------------------------ #

def get_device() -> torch.device:
    """Choose CUDA if available, else CPU."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def grad_global_norm(model: nn.Module) -> float:
    """Compute global grad-norm for logging."""
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(p.grad.detach().float().norm())
    if not norms:
        return 0.0
    return torch.stack(norms).norm().item()

@torch.no_grad()
def cuda_mem_telemetry() -> str:
    """Report CUDA allocated and reserved memory (MB)."""
    if not torch.cuda.is_available():
        return "cpu"
    mem_alloc = torch.cuda.memory_allocated() / (1024**2)
    mem_resvd = torch.cuda.memory_reserved() / (1024**2)
    return f"cuda mem: alloc={mem_alloc:.1f}MB resv={mem_resvd:.1f}MB"

# ------------------------------ Toy corpora ------------------------------ #

def build_dataset_default(repeat:int=20000) -> Tuple[str, Dict[str,int], Dict[int,str]]:
    """Simple repetitive English-like corpus for quick sanity tests."""
    base = "hello world, it's test corpus. "
    text = base * repeat
    vocab = sorted(set(text))
    stoi = {ch:i for i,ch in enumerate(vocab)}
    itos = {i:ch for ch,i in stoi.items()}
    return text, stoi, itos

def build_dataset_needle(repeat:int=6000, needle_period:int=200, needle:str=" target phrase ") -> Tuple[str, Dict[str,int], Dict[int,str]]:
    """Insert a periodic 'needle' substring into a long haystack. Space-separated lowercase only."""
    hay = "needle test string "
    chunks = []
    acc = 0
    for _ in range(repeat):
        s = hay
        acc += len(s)
        if acc >= needle_period:
            s += needle
            acc = 0
        chunks.append(s)
    text = ''.join(chunks)
    vocab = sorted(set(text))
    stoi = {ch:i for i,ch in enumerate(vocab)}
    itos = {i:ch for ch,i in stoi.items()}
    return text, stoi, itos

def build_dataset_from_text(text:str) -> Tuple[str, Dict[str,int], Dict[int,str]]:
    """Construct (text, stoi, itos) from an external raw text string."""
    if not isinstance(text, str) or len(text) < 8:
        raise ValueError("External text is too short; provide a longer file.")
    vocab = sorted(set(text))
    stoi = {ch:i for i,ch in enumerate(vocab)}
    itos = {i:ch for ch,i in stoi.items()}
    return text, stoi, itos

# ------------------------------ Mini dataloader ------------------------------ #

@dataclass
class Batch:
    x: torch.Tensor  # (B, T) int64
    y: torch.Tensor  # (B, T) int64

class CharLoader:
    """Uniformly sample overlapping char sequences from a long text."""
    def __init__(self, text:str, stoi:Dict[str,int], seq_len:int=64, batch_size:int=32, seed:int=1337, corrupt:float=0.0):
        self.text = text
        self.stoi = stoi
        self.vocab_size = len(stoi)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = torch.Generator().manual_seed(seed)
        self.as_ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
        if len(self.as_ids) < (seq_len + 2):
            raise ValueError(f"Text too short for seq_len={seq_len}. Need at least {seq_len+2} chars.")
        self.max_start = len(self.as_ids) - (seq_len + 1)
        self.corrupt = float(corrupt)

    def sample(self, device:torch.device) -> Batch:
        """Sample a batch of (x,y) char sequences with optional input corruption."""
        idx = torch.randint(low=0, high=self.max_start, size=(self.batch_size,), generator=self.rng)
        seqs = [self.as_ids[i:i+self.seq_len+1] for i in idx]
        chunk = torch.stack(seqs, dim=0).to(device)
        x = chunk[:, :-1]
        y = chunk[:, 1:]
        if self.corrupt > 0.0:
            mask = torch.rand_like(x.float()) < self.corrupt
            rand_tok = torch.randint(low=0, high=self.vocab_size, size=x.shape, device=device)
            x = torch.where(mask, rand_tok, x)
        return Batch(x,y)

# ------------------------------ CountSketch ------------------------------ #

class CountSketch(nn.Module):
    """
    CountSketch: linear map R^d -> R^m using a fixed hash (index + sign).
    Enables convolution-like properties under polynomial kernel sketches.
    """
    def __init__(self, d:int, m:int, seed:int=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        h = torch.randint(low=0, high=m, size=(d,), generator=g)
        s = torch.randint(low=0, high=2, size=(d,), generator=g) * 2 - 1  # ±1
        self.register_buffer("h", h.long())
        self.register_buffer("sign", s.float())
        self.m = m
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *B, d = x.shape
        assert d == self.d, f"CountSketch expected last-dim={self.d}, got {d}"
        out = x.new_zeros(*B, self.m)
        idx = self.h.expand(*B, d)
        val = (self.sign.expand_as(x) * x)
        out.scatter_add_(-1, idx, val)
        return out

# ------------------------------ OSAM core (cross + V readout) ------------------------------ #

class OSAMCrossBlock(nn.Module):
    """
    D=2 cross TensorSketch OSAM (prefix is treated as constant buffer):
      • Maintain prefix K = Σ S(k_i · e^{+i ω pos_i}) as 2 channels (Re/Im)
      • Compute query Q = S(q_t · e^{-i ω pos_t}) on the fly
      • φ_t = IFFT(FFT(Q) ⊙ FFT(K)) (norm='ortho') —> readout features
      • Constant-memory Value readout with H = Σ v_i ⊗ S(k_i · e^{+i ω pos_i})
      • Optional frequency cache FK for K (differential updates)
    Notes:
      We block gradients through historical prefix states (K/H/FK). This avoids in-place
      version bumps while keeping learning signals local to current projections and readout.
    """
    def __init__(
        self,
        d_model:int,
        m:int=2048,
        readout_hidden:int=128,
        gate_cap_feat:float=0.25,
        gate_cap_val:float=0.25,
        enable_val:bool=True,
        dropout:float=0.0,
        use_freq_cache:bool=False
    ):
        super().__init__()
        self.d_model = d_model
        self.m = m
        self.gate_cap_feat = gate_cap_feat
        self.gate_cap_val  = gate_cap_val
        self.enable_val = enable_val
        self.use_freq_cache = use_freq_cache

        # Projections
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        nn.init.orthogonal_(self.Wq.weight)
        nn.init.orthogonal_(self.Wk.weight)
        nn.init.orthogonal_(self.Wv.weight)

        # CountSketch
        self.cs = CountSketch(d_model, m, seed=42)

        # Phase params (bounded via tanh)
        self.theta_q_raw = nn.Parameter(torch.tensor(0.05))
        self.theta_k_raw = nn.Parameter(torch.tensor(0.05))
        self.omega_raw   = nn.Parameter(torch.tensor(0.02))

        # Readout (φ -> d)
        self.readout = nn.Sequential(
            nn.Linear(m, readout_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(readout_hidden, d_model)
        )

        # V-path projection
        self.v_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # Residual gates
        self.gate_feat_raw = nn.Parameter(torch.tensor(-2.5))  # sigmoid(-2.5)≈0.075
        self.gate_val_raw  = nn.Parameter(torch.tensor(-2.5))
        self.ln = nn.LayerNorm(d_model)

    @property
    def gate_feat(self) -> torch.Tensor:
        return self.gate_cap_feat * torch.sigmoid(self.gate_feat_raw)
    @property
    def gate_val(self) -> torch.Tensor:
        if not self.enable_val:
            return torch.tensor(0.0, device=self.gate_feat_raw.device)
        return self.gate_cap_val * torch.sigmoid(self.gate_val_raw)
    @property
    def theta_q(self) -> torch.Tensor:
        return 0.5 * torch.tanh(self.theta_q_raw)
    @property
    def theta_k(self) -> torch.Tensor:
        return 0.5 * torch.tanh(self.theta_k_raw)
    @property
    def omega(self) -> torch.Tensor:
        return 0.5 * torch.tanh(self.omega_raw)

    def zero_state(self, batch_size:int, device:torch.device):
        """Allocate zeroed states: K:(B,2,m), H:(B,2,d,m), FK:(B,m) or None."""
        K = torch.zeros(batch_size, 2, self.m, device=device)
        H = torch.zeros(batch_size, 2, self.d_model, self.m, device=device)
        FK = torch.zeros(batch_size, self.m, dtype=torch.complex64, device=device) if self.use_freq_cache else None
        return K, H, FK

    def _cs_complex(self, re:torch.Tensor, im:torch.Tensor):
        return self.cs(re), self.cs(im)

    def step(
        self,
        e_t:torch.Tensor,
        pos_t:torch.Tensor,
        K:torch.Tensor,
        H:torch.Tensor,
        FK:Optional[torch.Tensor],
        compute:bool=True
    ):
        """
        Single-timestep update.
        We treat K/H/FK as constant buffers when *reading* (detach). Updates write new
        buffers for the next step (also detached) to avoid autograd in-place issues.
        """
        B, d = e_t.shape
        # q,k,v
        q = self.Wq(e_t)
        k = self.Wk(e_t)
        v = self.Wv(e_t)

        # Relative phase
        phase_q = self.theta_q * q - self.omega * pos_t.view(-1,1)
        phase_k = self.theta_k * k + self.omega * pos_t.view(-1,1)
        q_re = q * torch.cos(phase_q); q_im = q * torch.sin(phase_q)
        k_re = k * torch.cos(phase_k); k_im = k * torch.sin(phase_k)

        # Q sketch (grad flows to current q)
        Q_re, Q_im = self._cs_complex(q_re, q_im)

        # φ via FFT using detached K
        if compute:
            FQ = fft.fft(torch.complex(Q_re, Q_im), dim=-1, norm="ortho")
            if self.use_freq_cache and FK is not None:
                psi = fft.ifft(FQ * FK, dim=-1, norm="ortho").real
            else:
                with torch.no_grad():
                    Kc = torch.complex(K.detach()[:,0,:].float(), K.detach()[:,1,:].float())
                    FK_now = fft.fft(Kc, dim=-1, norm="ortho")
                psi = fft.ifft(FQ * FK_now, dim=-1, norm="ortho").real
        else:
            psi = e_t.new_zeros(B, self.m)

        # Length normalization
        tscale = (pos_t.float() + 1.0).sqrt().view(-1,1)
        psi = psi / tscale

        # Feature readout
        h_feat = self.readout(psi)

        # V-path: update H (buffer semantics)
        dK_re, dK_im = self._cs_complex(k_re, k_im)  # depends on current k
        H_prev = H.detach()
        H_re_prev, H_im_prev = H_prev[:,0,:,:], H_prev[:,1,:,:]
        H_re_next = H_re_prev + v.unsqueeze(-1) * dK_re.detach().unsqueeze(1)
        H_im_next = H_im_prev + v.unsqueeze(-1) * dK_im.detach().unsqueeze(1)
        H_next = torch.stack([H_re_next, H_im_next], dim=1).detach()

        # FK cache differential update
        if self.use_freq_cache:
            with torch.no_grad():
                dFK = fft.fft(torch.complex(dK_re.detach(), dK_im.detach()), dim=-1, norm="ortho")
                FK = (FK + dFK) if FK is not None else dFK
        FK_next = FK

        # Constant-memory value read (no grad through H_prev or Q sketch)
        vread = e_t.new_zeros(B, d)
        if compute and self.enable_val:
            vread = (torch.matmul(H_re_prev, Q_re.detach().unsqueeze(-1)).squeeze(-1)
                     - torch.matmul(H_im_prev, Q_im.detach().unsqueeze(-1)).squeeze(-1))
            vread = (vread / tscale)
            vread = self.v_proj(vread)

        # Residual + LN
        y = self.ln(e_t + self.gate_feat * h_feat + self.gate_val * vread)

        # K update (buffer semantics)
        K_prev = K.detach()
        K_next = torch.stack([K_prev[:,0,:] + dK_re.detach(),
                              K_prev[:,1,:] + dK_im.detach()], dim=1).detach()

        return y, K_next, H_next, FK_next, psi, vread

# ------------------------------ Language model wrapper ------------------------------ #

class OSAMLM(nn.Module):
    """Character-level LM using a single OSAM cross+value block."""
    def __init__(
        self,
        vocab_size:int,
        d_model:int=64,
        m:int=2048,
        readout_hidden:int=128,
        enable_val:bool=True,
        dropout:float=0.0,
        use_freq_cache:bool=False,
        kupdate:int=1,
        telemetry_stride:int=1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.kupdate = max(1, int(kupdate))
        self.telemetry_stride = max(1, int(telemetry_stride))
        self.embed = nn.Embedding(vocab_size, d_model)
        self.osam = OSAMCrossBlock(
            d_model, m=m, readout_hidden=readout_hidden, enable_val=enable_val,
            dropout=dropout, use_freq_cache=use_freq_cache
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x:torch.Tensor, return_telemetry:bool=False):
        B, T = x.shape
        device = x.device
        e = self.embed(x)  # (B,T,d)
        K, H, FK = self.osam.zero_state(B, device)
        ys = []
        psi_norms, K_norms, H_norms = [], [], []
        pos = torch.arange(T, device=device).view(1,T).expand(B,T)
        for t in range(T):
            compute = ((t % self.kupdate) == 0)
            y_t, K, H, FK, psi_t, _v = self.osam.step(e[:,t,:], pos[:,t], K, H, FK, compute=compute)
            ys.append(y_t)
            if return_telemetry and compute and ((t % self.telemetry_stride) == 0):
                psi_norms.append(psi_t.detach().float().norm(dim=-1))
                K_norms.append(K.detach().float().norm(dim=(-2,-1)))
                H_norms.append(H.detach().float().norm(dim=(-3,-1)))
        Y = torch.stack(ys, dim=1)         # (B,T,d)
        logits = self.head(Y)              # (B,T,V)
        tel = {}
        if return_telemetry and psi_norms:
            psi_norm = torch.stack(psi_norms, dim=1).mean().item()
            K_norm   = torch.stack(K_norms,   dim=0).mean().item()
            H_norm   = torch.stack(H_norms,   dim=0).mean().item()
            tel = {
                "phi_norm": psi_norm,
                "s1_norm": (K_norm + H_norm),
                "gate_feat": self.osam.gate_feat.detach().item(),
                "gate_val":  self.osam.gate_val.detach().item(),
                "theta_q": self.osam.theta_q.detach().item(),
                "theta_k": self.osam.theta_k.detach().item(),
                "omega": self.osam.omega.detach().item()
            }
        return logits, tel

    @torch.no_grad()
    def prime_state(self, prompt_ids:torch.Tensor):
        """Prime internal state (K,H,FK) by running through a prompt (no logits returned)."""
        device = prompt_ids.device
        T0 = prompt_ids.numel()
        K, H, FK = self.osam.zero_state(batch_size=1, device=device)
        e_all = self.embed(prompt_ids.view(1,-1))
        for t in range(T0):
            pos_t = torch.tensor([t], device=device)
            _y, K, H, FK, _psi, _v = self.osam.step(e_all[:,t,:], pos_t, K, H, FK, compute=True)
        return K, H, FK

# ------------------------------ Generation utils ------------------------------ #

def encode_str(s:str, stoi:Dict[str,int]) -> torch.Tensor:
    """Encode a Python string into tensor of indices using provided stoi; fallback to space or first token."""
    unk = stoi[' '] if (' ' in stoi) else next(iter(stoi.values()))
    ids = [stoi.get(ch, unk) for ch in s]
    return torch.tensor(ids, dtype=torch.long)

def sample_next_id(
    logits:torch.Tensor,
    temp:float=1.0,
    topk:int=0,
    topp:float=0.0,
    greedy:bool=False
) -> int:
    """Sample next token index from logits with temperature/top-k/top-p/greedy."""
    if greedy:
        return int(torch.argmax(logits, dim=-1).item())
    logits = logits / max(1e-6, temp)
    probs = torch.softmax(logits, dim=-1)
    # top-k
    if topk and topk > 0:
        topk = min(topk, probs.size(-1))
        vals, idx = torch.topk(probs, k=topk, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, idx, 1.0)
        probs = probs * mask
        probs = probs / probs.sum(dim=-1, keepdim=True)
    # top-p
    if topp and topp > 0.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum <= topp
        keep[...,0] = True
        mask = torch.zeros_like(probs)
        mask.scatter_(1, sorted_idx, keep.float())
        probs = probs * mask
        probs = probs / probs.sum(dim=-1, keepdim=True)
    next_id = torch.multinomial(probs, num_samples=1)
    return int(next_id.item())

@torch.no_grad()
def generate_stream(
    model:OSAMLM,
    stoi:Dict[str,int],
    itos:Dict[int,str],
    prompt:str,
    max_new_tokens:int=64,
    temp:float=1.0,
    topk:int=0,
    topp:float=0.0,
    greedy:bool=False
) -> str:
    """Greedy/sampled generation with state priming from a text prompt. Streams to stdout."""
    device = next(model.parameters()).device
    model.eval()
    prompt_ids = encode_str(prompt, stoi).to(device)

    # Prime state with the full prompt
    K, H, FK = model.prime_state(prompt_ids)

    # Start from last token
    cur = prompt_ids[-1].view(1,1)
    out_ids = prompt_ids.tolist()

    t0 = time.time()
    print(f"\n[DEMO] prompt: {repr(prompt)}")
    print("[DEMO] generating:", end=" ", flush=True)

    for k in range(max_new_tokens):
        e = model.embed(cur)  # (1,1,d)
        pos_t = torch.tensor([prompt_ids.numel() + k], device=device)
        y, K, H, FK, _psi, _v = model.osam.step(e[:,0,:], pos_t, K, H, FK, compute=True)
        logits = model.head(y)  # (1,V)
        nid = sample_next_id(logits, temp=temp, topk=topk, topp=topp, greedy=greedy)
        out_ids.append(nid)
        ch = itos.get(int(nid), '?')
        print(ch, end="", flush=True)
        cur = torch.tensor([[nid]], device=device, dtype=torch.long)

    total = time.time() - t0
    n_new = len(out_ids) - prompt_ids.numel()
    tokps = n_new / max(total, 1e-6)
    print(f"\n[DEMO] done: {n_new} tokens in {total:.3f}s  → {tokps:.1f} tok/s")

    return ''.join([itos.get(int(i),'?') for i in out_ids])

# ------------------------------ Training loop ------------------------------ #

@dataclass
class Config:
    steps: int = 600
    batch_size: int = 32
    seq_len: int = 64
    d_model: int = 64
    m: int = 2048
    readout_hidden: int = 128
    lr: float = 3e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_interval: int = 50
    seed: int = 1337
    enable_val: bool = True
    freeze_phase: bool = False
    corrupt: float = 0.0
    dropout: float = 0.0
    label_smoothing: float = 0.0
    dataset: str = "default"  # or "needle"
    kupdate: int = 1
    freq_cache: bool = False
    # External corpus path
    text_path: Optional[str] = None
    # AMP & telemetry
    amp: str = "none"            # {"none","bf16","fp16"}
    telemetry_stride: int = 1
    # cuFFT plan cache
    cufft_max_plans: int = 4
    no_warmup_plans: bool = False
    # Checkpoint & inference
    save_path: Optional[str] = None
    load_path: Optional[str] = None
    no_train: bool = False
    prompt: Optional[str] = None
    max_new_tokens: int = 64
    temp: float = 1.0
    topk: int = 0
    topp: float = 0.0
    greedy: bool = False
    interactive: bool = False

class EMA:
    """Simple exponential moving average for scalar tracking."""
    def __init__(self, alpha:float=0.1):
        self.alpha = alpha
        self.v = None
    def update(self, x:float) -> float:
        if self.v is None:
            self.v = x
        else:
            self.v = self.alpha * x + (1-self.alpha) * self.v
        return self.v

def save_checkpoint(path:str, model:OSAMLM, cfg:Config, stoi:Dict[str,int], itos:Dict[int,str]):
    """Save model state and metadata."""
    ckpt = {
        'state_dict': model.state_dict(),
        'cfg': {
            'vocab_size': model.vocab_size,
            'd_model': model.d_model,
            'm': model.osam.m,
            'readout_hidden': cfg.readout_hidden,
            'enable_val': cfg.enable_val,
            'use_freq_cache': cfg.freq_cache,
        },
        'train_cfg': asdict(cfg),
        'stoi': stoi,
        'itos': itos,
    }
    torch.save(ckpt, path)
    print(f"[CKPT] saved → {path}")

def load_checkpoint(path:str, device:torch.device) -> Tuple[OSAMLM, Dict[str,int], Dict[int,str]]:
    """Load model and tokenizers from a checkpoint."""
    ckpt = torch.load(path, map_location=device)
    stoi = ckpt['stoi']; itos = ckpt['itos']
    c = ckpt['cfg']
    model = OSAMLM(
        vocab_size=c['vocab_size'],
        d_model=c['d_model'],
        m=c['m'],
        readout_hidden=c['readout_hidden'],
        enable_val=c['enable_val'],
        dropout=0.0,
        use_freq_cache=c.get('use_freq_cache', False),
        kupdate=1
    ).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    print(f"[CKPT] loaded: d={c['d_model']} m={c['m']} vocab={c['vocab_size']}")
    return model, stoi, itos

def load_corpus_from_path(path:str) -> Tuple[str, Dict[str,int], Dict[int,str]]:
    """Read external text file (UTF-8) and build (text, stoi, itos)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if "\n" not in text:
        text = text + "\n"
    if " " not in text:
        text = " " + text
    return build_dataset_from_text(text)

@torch.no_grad()
def _warmup_plans(model:OSAMLM, vocab:int, B:int, T:int, device:torch.device):
    """Prime FFT plan cache for two shapes: (B,T) and (1,T)."""
    xB = torch.randint(0, vocab, (B, T), device=device)
    x1 = torch.randint(0, vocab, (1, T), device=device)
    model.eval()
    model(xB, return_telemetry=False)
    model(x1, return_telemetry=False)

def train_and_eval(cfg:Config):
    torch.manual_seed(cfg.seed)
    device = get_device()
    print(f"device: {device}")

    # ---------- Data ----------
    if cfg.no_train and cfg.load_path:
        text = None; stoi = None; itos = None
    else:
        if cfg.text_path:
            text, stoi, itos = load_corpus_from_path(cfg.text_path)
            print(f"[DATA] loaded external corpus: path='{cfg.text_path}', chars={len(text):,}, vocab={len(stoi)}")
        else:
            if cfg.dataset == "needle":
                text, stoi, itos = build_dataset_needle(repeat=8_000, needle_period=240, needle=" target phrase ")
                print(f"[DATA] using built-in 'needle' corpus: chars={len(text):,}, vocab={len(stoi)}")
            else:
                text, stoi, itos = build_dataset_default(repeat=20_000)
                print(f"[DATA] using built-in 'default' corpus: chars={len(text):,}, vocab={len(stoi)}")

    # ---------- Inference-only ----------
    if cfg.no_train and cfg.load_path:
        model, stoi, itos = load_checkpoint(cfg.load_path, device)
        if torch.cuda.is_available():
            torch.backends.cuda.cufft_plan_cache.max_size = max(1, cfg.cufft_max_plans)
        if cfg.interactive:
            print("\n[REPL] type your prompt. Ctrl+C to quit.\n")
            while True:
                try:
                    s = input(">> ")
                except (EOFError, KeyboardInterrupt):
                    print("\n[REPL] bye.")
                    break
                s = (s or "").rstrip("\n")
                generate_stream(
                    model, stoi, itos,
                    s if s else "hello ",
                    max_new_tokens=cfg.max_new_tokens,
                    temp=cfg.temp, topk=cfg.topk, topp=cfg.topp, greedy=cfg.greedy
                )
        else:
            prompt = cfg.prompt or "hello "
            generate_stream(
                model, stoi, itos, prompt,
                max_new_tokens=cfg.max_new_tokens,
                temp=cfg.temp, topk=cfg.topk, topp=cfg.topp, greedy=cfg.greedy
            )
        return

    # ---------- Model ----------
    model = OSAMLM(
        vocab_size=len(stoi),
        d_model=cfg.d_model,
        m=cfg.m,
        readout_hidden=cfg.readout_hidden,
        enable_val=cfg.enable_val,
        dropout=cfg.dropout,
        use_freq_cache=cfg.freq_cache,
        kupdate=cfg.kupdate,
        telemetry_stride=cfg.telemetry_stride
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.3f}M | vocab={len(stoi)} | d={cfg.d_model} | m={cfg.m} | valpath={cfg.enable_val}")

    use_cuda = torch.cuda.is_available()

    # ---------- cuFFT plan cache & warmup ----------
    if use_cuda:
        torch.backends.cuda.cufft_plan_cache.clear()
        torch.backends.cuda.cufft_plan_cache.max_size = max(2, cfg.cufft_max_plans)
        if not cfg.no_warmup_plans:
            _warmup_plans(model, len(stoi), cfg.batch_size, min(4, cfg.seq_len), device)
            torch.backends.cuda.cufft_plan_cache.max_size = max(2, min(cfg.cufft_max_plans, 2))

    # ---------- Optimizer ----------
    special = [model.osam.gate_feat_raw, model.osam.gate_val_raw, model.osam.theta_q_raw, model.osam.theta_k_raw, model.osam.omega_raw]
    base_params = [p for n,p in model.named_parameters() if all(k not in n for k in ["gate_feat_raw","gate_val_raw","theta_q_raw","theta_k_raw","omega_raw"])]
    opt = torch.optim.AdamW(
        [
            {"params": base_params, "lr": cfg.lr},
            {"params": special, "lr": cfg.lr * 10.0}
        ],
        betas=(0.9,0.95), weight_decay=cfg.weight_decay
    )

    # Optional freeze
    if cfg.freeze_phase:
        for n,p in model.named_parameters():
            if any(k in n for k in ["theta_q_raw","theta_k_raw","omega_raw"]):
                p.requires_grad_(False)

    ema_loss = EMA(0.05)
    tokens_per_step = cfg.batch_size * cfg.seq_len

    # ---------- AMP ----------
    amp_enabled = (use_cuda and cfg.amp in {"bf16","fp16"})
    amp_dtype = torch.bfloat16 if cfg.amp == "bf16" else torch.float16
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=(amp_enabled and cfg.amp == "fp16"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and cfg.amp == "fp16"))

    # ---------- Stats ----------
    stats = {
        'tok_sum': 0.0, 'dt_sum': 0.0,
        'phi_sum': 0.0, 's1_sum': 0.0,
        'gate_feat_sum': 0.0, 'gate_val_sum': 0.0,
        'theta_q_sum': 0.0, 'theta_k_sum': 0.0, 'omega_sum': 0.0,
        'loss_last': None, 'ema_last': None, 'ppl_last': None, 'mem_last': None,
        'cnt': 0,
    }

    # ---------- Loader ----------
    loader = CharLoader(
        text, stoi,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        corrupt=cfg.corrupt
    )

    # ---------- Train ----------
    for step in range(1, cfg.steps+1):
        model.train()
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()

        batch = loader.sample(device)

        if amp_enabled:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits, tel = model(batch.x, return_telemetry=True)
                logits_flat = logits.contiguous().view(-1, logits.size(-1))
                targets_flat = batch.y.contiguous().view(-1)
                loss = F.cross_entropy(logits_flat, targets_flat, label_smoothing=cfg.label_smoothing)
        else:
            logits, tel = model(batch.x, return_telemetry=True)
            logits_flat = logits.contiguous().view(-1, logits.size(-1))
            targets_flat = batch.y.contiguous().view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, label_smoothing=cfg.label_smoothing)

        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        gnorm = grad_global_norm(model)
        ema = ema_loss.update(loss.item())
        if use_cuda:
            torch.cuda.synchronize()
        dt = max(time.time() - t0, 1e-6)
        toks_s = tokens_per_step / dt
        ppl = math.exp(max(min(loss.item(), 20.0), 0.0))

        # Stats
        stats['tok_sum'] += tokens_per_step
        stats['dt_sum']  += dt
        stats['cnt']     += 1
        if 'phi_norm' in tel: stats['phi_sum'] += tel['phi_norm']
        if 's1_norm' in tel:  stats['s1_sum']  += tel['s1_norm']
        if 'gate_feat' in tel: stats['gate_feat_sum'] += tel['gate_feat']
        if 'gate_val'  in tel: stats['gate_val_sum']  += tel['gate_val']
        if 'theta_q' in tel: stats['theta_q_sum'] += tel['theta_q']
        if 'theta_k' in tel: stats['theta_k_sum'] += tel['theta_k']
        if 'omega'   in tel: stats['omega_sum']   += tel['omega']
        stats['loss_last'] = loss.item()
        stats['ema_last']  = ema
        stats['ppl_last']  = ppl
        stats['mem_last']  = cuda_mem_telemetry()

        msg = (
            f"step {step:4d} | loss {loss.item():.4f} (ema {ema:.4f}) | ppl≈{ppl:.2f} | "
            f"gnorm {gnorm:.3f} | toks/s {toks_s:.1f} | "
            f"phi_norm {tel.get('phi_norm',0):.2f} | s1_norm {tel.get('s1_norm',0):.2f} | "
            f"gate_feat {tel.get('gate_feat',0):.4f} | gate_val {tel.get('gate_val',0):.4f} | "
            f"theta_q {tel.get('theta_q',0):.4f} | theta_k {tel.get('theta_k',0):.4f} | omega {tel.get('omega',0):.4f} | "
            f"{stats['mem_last']}"
        )
        print(msg)

        # Periodic mini-sample (greedy)
        if step % cfg.log_interval == 0 or step == cfg.steps:
            model.eval()
            if cfg.text_path:
                prompt = cfg.prompt or "hello "
            else:
                prompt = "hello " if cfg.dataset == "default" else " target "
            _ = generate_stream(model, stoi, itos, prompt, max_new_tokens=16, temp=1.0, topk=0, topp=0.0, greedy=True)

    # ---------- Cleanup ----------
    del opt
    if use_cuda:
        torch.cuda.empty_cache()

    # ---------- Final check ----------
    model.eval()
    if cfg.text_path:
        prompt = cfg.prompt or "hello "
    else:
        prompt = "hello " if cfg.dataset == "default" else " target "
    _ = generate_stream(model, stoi, itos, prompt, max_new_tokens=16, temp=1.0, topk=0, topp=0.0, greedy=True)

    # ---------- Summary ----------
    toks_avg = stats['tok_sum'] / max(1e-9, stats['dt_sum'])
    denom = max(1, stats['cnt'])
    print("== SUMMARY ==")
    print(
        f"config: steps={cfg.steps}, batch={cfg.batch_size}, seq={cfg.seq_len}, d={cfg.d_model}, m={cfg.m}, "
        f"valpath={cfg.enable_val}, corrupt={cfg.corrupt}, freeze_phase={cfg.freeze_phase}, "
        f"dropout={cfg.dropout}, ls={cfg.label_smoothing}, dataset={cfg.dataset}, kupdate={cfg.kupdate}, "
        f"freq_cache={cfg.freq_cache}, text_path={'True' if cfg.text_path else 'False'}, "
        f"amp={cfg.amp}, telemetry_stride={cfg.telemetry_stride}, cufft_max_plans={cfg.cufft_max_plans}"
    )
    if stats['loss_last'] is not None:
        print(f"train: final_loss={stats['loss_last']:.4f}, ema={stats['ema_last']:.4f}, ppl≈{stats['ppl_last']:.2f}, avg_toks/s={toks_avg:.1f}")
        print(
            "telemetry(avg): "
            f"gate_feat={stats['gate_feat_sum']/denom:.4f}, "
            f"gate_val={stats['gate_val_sum']/denom:.4f}, "
            f"theta_q={stats['theta_q_sum']/denom:.4f}, "
            f"theta_k={stats['theta_k_sum']/denom:.4f}, "
            f"omega={stats['omega_sum']/denom:.4f}, "
            f"phi_norm={stats['phi_sum']/denom:.2f}, "
            f"s1_norm={stats['s1_sum']/denom:.2f}"
        )
        print(f"resource(last): {stats['mem_last']}")

    # ---------- Save ----------
    if cfg.save_path:
        save_checkpoint(cfg.save_path, model, cfg, stoi, itos)

# ------------------------------ CLI ------------------------------ #

def parse_args() -> 'Config':
    p = argparse.ArgumentParser(description="OSAM cross+value demo: train and/or generate")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--m", type=int, default=2048)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no_valpath", action="store_true")
    p.add_argument("--freeze_phase", action="store_true")
    p.add_argument("--corrupt", type=float, default=0.0)
    # Extras
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--dataset", type=str, default="default", choices=["default","needle"])
    p.add_argument("--kupdate", type=int, default=1)
    p.add_argument("--freq_cache", action="store_true")
    # External corpus
    p.add_argument("--text_path", type=str, default=None, help="Path to external UTF-8 text corpus (.txt). Overrides --dataset if set.")
    # AMP & telemetry
    p.add_argument("--amp", type=str, default="none", choices=["none","bf16","fp16"], help="Mixed precision type (CUDA only).")
    p.add_argument("--telemetry_stride", type=int, default=1, help="Collect telemetry every N computed timesteps.")
    # cuFFT plan cache
    p.add_argument("--cufft_max_plans", type=int, default=4, help="Maximum cuFFT plan cache size.")
    p.add_argument("--no_warmup_plans", action="store_true", help="Skip plan warmup (not recommended).")
    # Checkpoint / inference
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--load", type=str, default=None)
    p.add_argument("--no_train", action="store_true")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=0)
    p.add_argument("--topp", type=float, default=0.0)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--interactive", action="store_true")

    args = p.parse_args([]) if False else p.parse_args()
    return Config(
        steps=args.steps, batch_size=args.batch, seq_len=args.seq,
        d_model=args.d, m=args.m, readout_hidden=args.hidden,
        lr=args.lr, weight_decay=args.wd, grad_clip=args.clip,
        seed=args.seed, enable_val=(not args.no_valpath), freeze_phase=args.freeze_phase, corrupt=args.corrupt,
        dropout=args.dropout, label_smoothing=args.label_smoothing, dataset=args.dataset,
        kupdate=args.kupdate, freq_cache=args.freq_cache,
        text_path=args.text_path,
        amp=args.amp, telemetry_stride=args.telemetry_stride,
        cufft_max_plans=args.cufft_max_plans, no_warmup_plans=args.no_warmup_plans,
        save_path=args.save, load_path=args.load, no_train=args.no_train, prompt=args.prompt,
        max_new_tokens=args.max_new_tokens, temp=args.temp, topk=args.topk, topp=args.topp,
        greedy=args.greedy, interactive=args.interactive
    )

if __name__ == "__main__":
    cfg = parse_args()
    train_and_eval(cfg)
