import torch
import torch.nn.functional as F


import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    n: int,
    qlens: int,
    kvlens: int,
    head_dim: int,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,  # 64
    stride_qk: tl.constexpr,  # 1
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,  # 64
    stride_kk: tl.constexpr,  # 1
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_ok: tl.constexpr,
    block_size_c: tl.constexpr,
    block_size_r: tl.constexpr,
    # block_size_k: tl.constexpr, # 和dim相同，暂时假设一次计算完
):
    pid = tl.program_id(0)

    offset_k = tl.arange(0, head_dim)
    q_block_ptr = (
        q_ptr + (pid * stride_qb + tl.arange(0, block_size_r) * stride_qh)[:, None] + offset_k[None, :] * stride_qk
    )
    o_block = tl.zeros((block_size_r, block_size_c), dtype=tl.float32)
    l = tl.zeros((block_size_r), dtype=tl.float32)
    m = tl.ones((block_size_r), dtype=tl.float32) * (-float("inf"))
    m_old = m
    o_block_ptr = o_ptr + (pid * stride_ob + tl.arange(0, block_size_r) * stride_ok)[:, None] + offset_k[None, :] * stride_qk

    k_block_ptr = (
        k_ptr + (pid * stride_kb + tl.arange(0, block_size_r) * stride_kh)[:, None] + offset_k[None, :] * stride_kk
    )
    v_block_ptr = (
        v_ptr + (pid * stride_vb + tl.arange(0, block_size_r) * stride_vh)[:, None] + offset_k[None, :] * stride_vk
    )

    # s = tl.zeros([block_size_r, block_size_c], dtype=tl.float32)

    q_block = tl.load(q_block_ptr + tl.arange(0, block_size_r) * stride_qk)

    for i in range(0, kvlens, block_size_c):
        # k_block_ptr = k_ptr + i * stride_kh + tl.arange(0, block_size_c) * stride_kk
        # v_block_ptr = v_ptr + i * stride_vh + tl.arange(0, block_size_c) * stride_vk
        k_block = tl.load(k_block_ptr + i * stride_kh)

        s = q_block @ k_block.transpose(-1, -2) / (head_dim**0.5)

        row_max = tl.max(s, 1)
        m = tl.maximum(m, row_max)

        p = tl.exp(s - m[:, None])
        l = l * tl.exp(m_old - m) + tl.sum(p, 1)
        v_block = tl.load(v_block_ptr + i * stride_vh)
        o_block = (o_block * tl.exp(m_old - m)[:, None] + p @ v_block) / l[:, None]

    tl.store(o_block_ptr, o_block)

    tl.store(lse_ptr + pid, l)
    


# multi-head attention
def attention(q, k, v, block_size_b, block_size_h, block_size_k, block_size_v, block_size_o):
    bs, h, qlens, d = q.shape
    _, _, kvlens, _ = k.shape

    q = q.view(-1, qlens, d)
    k = k.view(-1, kvlens, d)
    v = v.view(-1, kvlens, d)

    n = bs * h
    grid = (bs * h,)
    o = torch.empty_like(q)

    block_size_c = 128
    block_size_r = 64

    _kernel[grid](
        q,
        k,
        v,
        o,
        n,
        qlens,
        kvlens,
        d,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        block_size_c,
        block_size_r,
    )
    return o.view(bs, h, qlens, d)


bs = 2
qlens = 128
kvlens = 256
h = 12
d = 64


# DEVICE = triton.runtime.driver.active.get_active_torch_device()
DEVICE = "cpu"
dtype = torch.float16
Q = torch.randn(bs, h, qlens, d, device=DEVICE, dtype=dtype)
K = torch.randn(bs, h, kvlens, d, device=DEVICE, dtype=dtype)
V = torch.randn(bs, h, kvlens, d, device=DEVICE, dtype=dtype)


def native(q, k, v):
    S = Q @ K.transpose(-1, -2) / (d**0.5)
    O = S.softmax(dim=-1) @ V
    return O


def spda(q, k, v):
    O = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )
    return O


native_out = native(Q, K, V)
spda_out = spda(Q, K, V)


print(native_out)
print(spda_out)

print(f"max diff: {torch.max(torch.abs(native_out - spda_out))}")
print(torch.allclose(native_out, spda_out, atol=1e-3, rtol=1e-3))
