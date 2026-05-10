"""
train_simple.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
Modified to run on a single T4 GPU.
"""

import os
import sys
from huggingface_hub import hf_hub_download
# Download the GPT-2 tokens of FinewebEDU10B from huggingface. This
# saves about an hour of startup time compared to regenerating them.
def get(fname):
    local_dir = os.path.join(os.getcwd(), 'finewebedu10B')
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id="kjj0/finewebedu10B-gpt2", filename=fname,
                        repo_type="dataset", local_dir=local_dir)

get("finewebedu_val_%06d.bin" % 0)

num_chunks = 1

for i in range(1, num_chunks+1):
    get("finewebedu_train_%06d.bin" % i)

import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# --- Custom FP16 scaled matmul (avoids overflow) ---

@torch.library.custom_op("nanogpt::mm_fp16_scaled", mutates_args=())
def mm_fp16_scaled_op(
    x: Tensor, w: Tensor, x_s: Tensor, w_s: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Forward: y = (x / x_s) @ (w / w_s) * (x_s * w_s)
    Returns: y (FP32), x_scaled (FP16), w_scaled (FP16)
    x_s, w_s: scalar tensors (shape [])
    """
    @torch.compile
    def impl(x, w, x_s, w_s):
        assert x.is_contiguous()   # w can be non-contig (transposed)
        x_scaled = x.half() / x_s.half()
        w_scaled = w.half() / w_s.half()
        y_scaled = torch.matmul(x_scaled, w_scaled)
        y = y_scaled.float() * (x_s.float() * w_s.float())
        return y, x_scaled, w_scaled
    return impl(x, w, x_s, w_s)

@mm_fp16_scaled_op.register_fake
def _(x, w, x_s, w_s):
    return x @ w, x.half(), w.half()

# --- Backward op ---

@torch.library.custom_op("nanogpt::mm_fp16_scaled_backward", mutates_args=())
def mm_fp16_scaled_backward_op(
    grad_out: Tensor, x_scaled: Tensor, w_scaled: Tensor,
    x_s: Tensor, w_s: Tensor
) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad_out, x_scaled, w_scaled, x_s, w_s):
        grad_max = grad_out.abs().max()
        w_scaled_max = w_scaled.abs().max().float()
        K = w_scaled.shape[1] # inner dimension (out_features)

        # Safely bound the maximum value of the intermediate grad_x_scaled
        # so that `grad_scaled @ w_scaled.T` never overflows FP16's max value (~65500)
        factor = torch.clamp(w_scaled_max * K, min=1.0)
        grad_s = torch.clamp((grad_max / 65000.0) * factor, min=1e-6)

        # Divide in FP32 first to prevent FP16 underflow of grad_s, then cast
        grad_scaled = (grad_out / grad_s).half()

        # grad_x = (grad_scaled * grad_s) @ (w_scaled * w_s).T
        grad_x_scaled = torch.matmul(grad_scaled, w_scaled.T)
        grad_x = grad_x_scaled.float() * (grad_s.float() * w_s.float())

        # grad_w is accurately accumulated in FP32 (which is good practice)
        x_s_f = x_s.float()
        x_f = x_scaled.float() * x_s_f
        grad_w = torch.matmul(x_f.T, grad_out)  

        return grad_x.contiguous(), grad_w.contiguous()

    return impl(grad_out, x_scaled, w_scaled, x_s, w_s)
    
@mm_fp16_scaled_backward_op.register_fake
def _(g, x_scaled, w_scaled, *_):
    # grad_x is same shape & stride as x_scaled
    # grad_w has shape (in_features, out_features), must be row-major
    grad_x_fake = x_scaled.float().contiguous()
    grad_w_fake = w_scaled.float().contiguous()
    return grad_x_fake, grad_w_fake

# --- Autograd setup ---

def backward_t(ctx, grad_out, *_):
    x_scaled, w_scaled, x_s, w_s = ctx.saved_tensors
    grad_x, grad_w = torch.ops.nanogpt.mm_fp16_scaled_backward(
        grad_out, x_scaled, w_scaled, x_s, w_s
    )
    return grad_x, grad_w, None, None

def setup_context_t(ctx, inputs, output):
    x, w, x_s, w_s = inputs
    _, x_scaled, w_scaled = output
    ctx.save_for_backward(x_scaled, w_scaled, x_s, w_s)
    ctx.set_materialize_grads(False)

mm_fp16_scaled_op.register_autograd(backward_t, setup_context=setup_context_t)

########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    files = sorted(Path.cwd().glob(filename_pattern))
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos : pos + batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

REG_MODE = 'baseline'
SIGR_ALPHA = 0.0

def sigreg_weak_loss(x, sketch_dim=64):
    N, C = x.size()
    if C > sketch_dim:
        S = torch.randn(sketch_dim, C, device=x.device, dtype=x.dtype) / (C ** 0.5)
        x = x @ S.T
    else:
        sketch_dim = C
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (N - 1 + 1e-6)
    target = torch.eye(sketch_dim, device=x.device)
    loss = torch.norm(cov - target, p='fro')
    return loss

def zipf_orthogonal_est(x, sketch_dim=64, zipf_s=1.0, lam_ang=1.0, lam_mag=1.0, eps=1e-6):
    N, C = x.size()
    if C > sketch_dim:
        S = torch.randn(sketch_dim, C, device=x.device, dtype=x.dtype) / (C ** 0.5)
        x = x @ S.T
        C = sketch_dim
    x = x - x.mean(dim=0, keepdim=True)
    norms = x.norm(dim=1, keepdim=True).clamp_min(eps)
    u = x / norms
    G = (u.T @ u) / (N - 1 + eps)
    ang_loss = torch.norm(G - torch.diag(torch.diag(G)), p='fro')
    sorted_norms, _ = torch.sort(norms.squeeze(-1), descending=True)
    ranks = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
    zipf_target = ranks.pow(-zipf_s)
    zipf_target = zipf_target / (zipf_target.sum() + eps)
    sorted_norms = sorted_norms / (sorted_norms.sum() + eps)
    mag_loss = torch.norm(sorted_norms - zipf_target, p='fro')
    return lam_ang * ang_loss + lam_mag * mag_loss

def sireg_discrete_loss(x, sketch_dim=64):
    N, C = x.size()
    x = F.normalize(x, p=2, dim=-1) * (C ** 0.5)
    if C > sketch_dim:
        S = torch.randn(sketch_dim, C, device=x.device, dtype=x.dtype) / (C ** 0.5)
        x = x @ S.T
    else:
        sketch_dim = C
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (N - 1 + 1e-6)
    target = torch.eye(sketch_dim, device=x.device)
    loss = torch.norm(cov - target, p='fro')
    return loss

def sigreg_strong_loss(x, sketch_dim=64):
    N, C = x.size()
    A = torch.randn(C, sketch_dim, device=x.device, dtype=x.dtype)
    A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)
    t = torch.linspace(-5, 5, 17, device=x.device)
    exp_f = torch.exp(-0.5 * t**2)
    proj = x @ A
    args = proj.unsqueeze(2) * t.view(1, 1, -1)
    cos_mean = torch.cos(args).mean(dim=0)
    sin_mean = torch.sin(args).mean(dim=0)
    diff_sq = (cos_mean - exp_f.unsqueeze(0)).square() + sin_mean.square()
    err = diff_sq * exp_f.unsqueeze(0)
    loss = torch.trapz(err, t, dim=1) * N
    return loss.mean()

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])

        max_x = x_flat.detach().abs().max()
        max_w = self.weight.data.abs().max()

        # raw scales to keep inputs just inside FP16 range
        x_s_raw = torch.clamp(max_x / 65000.0, min=1e-8)
        w_s_raw = torch.clamp(max_w / 65000.0, min=1e-8)

        K = x_flat.shape[-1]
        # worst-case dot product magnitude before scaling up the scales
        worst_dot = (max_x / x_s_raw) * (max_w / w_s_raw) * K
        safety_factor = torch.sqrt(torch.clamp(worst_dot / 65504, min=1.0))
        x_s = x_s_raw * safety_factor
        w_s = w_s_raw * safety_factor

        y_flat, _, _ = torch.ops.nanogpt.mm_fp16_scaled(
            x_flat, self.weight.T, x_s, w_s
        )

        y = y_flat.view(*orig_shape[:-1], -1)
        if self.bias is not None:
            y = y + self.bias.type_as(y)
        return y

class Linear32(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2).half(), k.transpose(1, 2).half(),
                                           v.transpose(1, 2).half(), scale=0.12, is_causal=True).transpose(1, 2).float()
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        # Physics regularization loss
        reg_loss = torch.tensor(0.0, device=x.device)
        if REG_MODE != 'baseline':
            batch_size, seq_len, hidden_dim = x.shape
            flat_rep = x.reshape(-1, hidden_dim)
            if REG_MODE == 'weak':
                reg_loss = sigreg_weak_loss(flat_rep)
            elif REG_MODE == 'strong':
                reg_loss = sigreg_strong_loss(flat_rep)
            elif REG_MODE == 'discrete':
                reg_loss = sireg_discrete_loss(flat_rep)
            elif REG_MODE == 'zipfian':
                reg_loss = zipf_orthogonal_est(flat_rep)

        return x, reg_loss

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).float()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        total_reg_loss = 0.0
        for block in self.blocks:
            x, reg_loss = block(x)
            total_reg_loss += reg_loss
        logits = self.proj(self.norm2(x))
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum"), (total_reg_loss / len(self.blocks))


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.float()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(5):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def normuon_update(grad, momentum, v_buf, mu=0.95, beta2=0.95, eps=1e-10, nesterov=True):
    """NorMuon direction: NS-orthogonalised gradient followed by Adafactor-style variance
    preconditioning along the SHORT axis (per https://arxiv.org/pdf/2510.05491). The
    variance buffer `v_buf` is an EMA of squared post-NS values along the short axis,
    persistent across optimizer steps."""
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    # Variance per row (if rows >= cols) or per column (otherwise) — Adafactor-style.
    if grad.size(-2) >= grad.size(-1):
        v_new = update.square().mean(dim=-1, keepdim=True)
    else:
        v_new = update.square().mean(dim=-2, keepdim=True)
    # `update` is bf16 after NS5; cast `v_new` to `v_buf`'s dtype (float32) so lerp_ matches.
    v_buf.lerp_(v_new.to(v_buf.dtype), 1 - beta2)
    update = update * v_buf.clamp_min(eps).rsqrt()
    return update


@torch.no_grad()
def scale_invariant_update_(param: Tensor, update: Tensor, lr: float, eps: float = 1e-10) -> None:
    """Hyperball-constrained step: take a preconditioned update of size lr * ||param||,
    then renormalise back onto the Frobenius sphere of the parameter's initial radius. Preserves
    ||param|| exactly across training; the invariant lets us drop weight decay on hidden
    matrices entirely (the constraint already prevents norm growth)."""
    p_norm = param.norm()
    u_norm = update.norm()
    new_param = param - lr * update * p_norm / torch.clamp(u_norm, min=eps)
    new_norm = torch.clamp(new_param.norm(), min=eps)
    param.copy_(new_param / new_norm * p_norm)


class NorMuonH(torch.optim.Optimizer):
    """NorMuonH: NS-orthogonalised gradient + Adafactor-style row/column variance
    preconditioning (NorMuon, https://arxiv.org/pdf/2510.05491) + hyperball Frobenius-
    norm-preserving step. Single‑GPU (no distributed) version."""

    def __init__(self, params, lr=0.018, mu=0.95, beta2=0.95, eps=1e-10):
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, mu=mu, beta2=beta2, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                    # variance buffer shape: keepdim along the short axis
                    v_shape = list(p.shape)
                    if p.size(-2) >= p.size(-1):
                        v_shape[-1] = 1
                    else:
                        v_shape[-2] = 1
                    state["v"] = torch.zeros(
                        v_shape, dtype=p.dtype, device=p.device
                    )

                update = normuon_update(
                    p.grad,
                    state["momentum"],
                    state["v"],
                    mu=group["mu"],
                    beta2=group["beta2"],
                    eps=group["eps"],
                )
                scale_invariant_update_(p, update, group["lr"])

########################################
#                Setup                 #
########################################

# Define device safely for Colab
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# logging setup
os.makedirs("logs", exist_ok=True)
logfile = f"logs/{uuid.uuid4()}.txt"
print(f"Logging to: {logfile}")

def print0(s, console=False, log=True):
    if console:
        print(s)
    if log:
        with open(logfile, "a") as f:
            print(s, file=f)

code = "Running in Colab Cell"
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + (f" on {torch.cuda.get_device_name(device)}" if device.type == "cuda" else " on CPU"), console=True)
print0("="*100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 8
val_inputs, val_targets = next(data_generator("finewebedu10B/finewebedu_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).to(device)
model = torch.compile(model, dynamic=False, fullgraph=True)

print0(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", console=True)

# ----------------- MFU Setup Variables -----------------
num_params = sum(p.numel() for p in model.parameters())
seq_len = 1024
# Standard formulation (6 * N + 12 * L * H * Q * T) derived from PaLM Appendix B
flops_per_token = 6 * num_params + 12 * 12 * 768 * seq_len
flops_per_step = flops_per_token * batch_size

peak_flops = 121e12 # default L4 fallback
if device.type == "cuda":
    gpu_name = torch.cuda.get_device_name()
    if "A100" in gpu_name:
        peak_flops = 312e12
    elif "V100" in gpu_name:
        peak_flops = 125e12
    elif "T4" in gpu_name:
        peak_flops = 65e12
    elif "L4" in gpu_name:
        peak_flops = 121e12
    elif "H100" in gpu_name:
        peak_flops = 989e12
    elif "H200" in gpu_name:
        peak_flops = 989e12

num_trials = 1

for _ in range(num_trials):

    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    # calculate total train steps based on available data
    train_files = sorted(Path.cwd().glob("finewebedu10B/finewebedu_train_*.bin"))
    train_steps = 0
    total_train_tokens = 0
    for f in train_files:
        header = torch.from_file(str(f), False, 256, dtype=torch.int32)
        num_tokens = int(header[2])
        total_train_tokens += num_tokens
        # Number of steps per shard mathematically aligns w/ generator (pos + batch_size + 1 >= num_tokens) condition
        train_steps += (num_tokens - 2) // batch_size

    print0(f"Calculated train_steps = {train_steps} from {total_train_tokens} tokens", console=True)

    # initialize model parameters. Per-module multipliers on the default nn.Linear Kaiming-uniform
    # init (std = 1/sqrt(3*fan_in), so ~0.0208 for fan_in=768 and ~0.0104 for fan_in=3072):
    #   - attn.proj.weight (fan_in=768):  default × 1.25 → std ≈ 0.026
    #   - mlp.proj.weight  (fan_in=3072): default × 3.0  → std ≈ 0.031
    #   - mlp.fc.weight    (fan_in=768):  default × 1.5  → std ≈ 0.031
    # qkv weights keep their default init. The vocab head (proj.weight) and all "proj" biases are
    # zeroed so initial logits are 0.
    for name, p in model.named_parameters():
        if name.endswith(".attn.proj.weight"):
            p.data.mul_(1.25)
        elif name.endswith(".mlp.proj.weight"):
            p.data.mul_(3.0)
        elif name.endswith(".mlp.fc.weight"):
            p.data.mul_(1.5)
        elif name == "proj.weight":
            p.data.zero_()
        elif "proj" in name:
            p.data.zero_()

    # split block-level 2D weights by module class. Each shape class gets its own NorMuonH instance
    # (same hyperparameters, but separate optimizers) so each shape gets its own torch.compile
    # cache for the Newton-Schulz path.
    named_block_params = [(n, p) for n, p in model.named_parameters()
                          if "blocks." in n and p.ndim >= 2]
                          
    qkv_params = [p for n, p in named_block_params
                  if n.endswith(".attn.q.weight") or n.endswith(".attn.k.weight") or n.endswith(".attn.v.weight")]

    mlp_fc_params = [p for n, p in named_block_params if n.endswith(".mlp.fc.weight")]
    attn_proj_params = [p for n, p in named_block_params if n.endswith(".attn.proj.weight")]
    mlp_proj_params = [p for n, p in named_block_params if n.endswith(".mlp.proj.weight")]

    # create the optimizer(s)
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                        dict(params=[model.proj.weight], lr=1/320),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                      betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
    optimizer2 = NorMuonH(qkv_params, lr=0.035)
    optimizer3 = NorMuonH(mlp_fc_params, lr=0.035)
    optimizer4 = NorMuonH(attn_proj_params, lr=0.035)
    optimizer5 = NorMuonH(mlp_proj_params, lr=0.035)
    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5]

    for opt in (optimizer2, optimizer3, optimizer4, optimizer5):
        for group in opt.param_groups:
                group["schedule_type"] = "h"

    for group in optimizer1.param_groups:
        group["schedule_type"] = "aux"

    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())

    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # learning rate schedule: stable then decay. The h (NorMuonH) groups use full linear cooldown
    # over the entire run (cooldown_frac=1.0); the aux (AdamW) group uses a shorter cooldown
    # (cooldown_frac=0.4) to keep the embed/head learning longer before tapering.
    def set_hparams(step):
        progress = step / train_steps
        assert 0 <= progress < 1
        for opt in optimizers:
            for group in opt.param_groups:
                cooldown_frac = 1.0 if group["schedule_type"] == "h" else 0.4
                if progress < 1 - cooldown_frac:
                    eta = 1.0
                else:
                    eta = (1 - progress) / cooldown_frac
                group["lr"] = group["initial_lr"] * eta

    ########################################
    #        Training and Validation       #
    ########################################

    train_loader = data_generator("finewebedu10B/finewebedu_train_*.bin", batch_size)

    # start the clock
    training_time = 0
    last_val_step = 0
    t0 = time.perf_counter()
    for step in range(train_steps + 1):

        # --------------- VALIDATION SECTION -----------------
        val_step_freq = 125 if step / train_steps < 0.9 else 25
        if step == train_steps or step % val_step_freq == 0:
            # stop the clock
            time_since_last_val = time.perf_counter() - t0
            step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += time_since_last_val
            model.eval()
            val_loss = 0
            with torch.no_grad():
                assert len(val_inputs) % mbs == 0
                for i in range(len(val_inputs) // mbs):
                    val_loss_step, _ = model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
                    val_loss += val_loss_step.item()
            val_loss /= val_tokens

            mfu_str = ""
            if step > 0 and device.type == "cuda" and step_avg > 0:
                achieved_flops = flops_per_step / step_avg
                mfu = achieved_flops / peak_flops * 100
                mfu_str = f" MFU:{mfu:.1f}%"

            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                   + f" step_avg:{1000*step_avg:.2f}ms{mfu_str}", console=True)
            model.train()
            # start the clock again
            t0 = time.perf_counter()

        if step == train_steps:
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        train_loss = 0

        # Microbatch (mbs) gradient accumulation runs successfully over batches
        assert len(inputs) % mbs == 0
        for i in range(len(inputs) // mbs):
            loss_step, reg_loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
            train_loss += loss_step.item()

            loss = (1 - SIGR_ALPHA) * loss_step + (SIGR_ALPHA * reg_loss)
            loss.backward()

        train_loss /= batch_size

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # set optimization hyperparameters and take a step
        set_hparams(step)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        approx_training_time = training_time + (time.perf_counter() - t0)
        curr_step_avg = approx_training_time / (step + 1)

        mfu_str_train = ""
        if device.type == "cuda" and curr_step_avg > 0:
            achieved_flops = flops_per_step / curr_step_avg
            mfu = achieved_flops / peak_flops * 100
            mfu_str_train = f" MFU:{mfu:.1f}%"

        print0(f"step:{step+1}/{train_steps} train_loss:{train_loss:.3f} train_time:{approx_training_time:.3f}s"
               + f" step_avg:{1000*curr_step_avg:.2f}ms{mfu_str_train}", console=True, log=False)