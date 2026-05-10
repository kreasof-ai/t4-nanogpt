import torch
from torch import Tensor

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