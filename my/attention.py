import torch
import torch.nn.functional as F


import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _fa2_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    n: int,
    qlens: int,
    kvlens: int,
    head_dim: tl.constexpr,
    softmax_scale,
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
    block_r: tl.constexpr,
    block_c: tl.constexpr,
    # block_size_k: tl.constexpr, # 和dim相同，暂时假设一次计算完
):
    head_idx = tl.program_id(0)
    q_block_idx = tl.program_id(1)

    q_head_ptr = q_ptr + head_idx * stride_qb + q_block_idx * block_r * stride_qh
    o_head_ptr = o_ptr + head_idx * stride_ob + q_block_idx * block_r * stride_oh
    k_head_ptr = k_ptr + head_idx * stride_kb
    v_head_ptr = v_ptr + head_idx * stride_vb

    offset_m = tl.arange(0, block_r)
    offset_k = tl.arange(0, head_dim)
    offset_n = tl.arange(0, block_c)

    q_block_ptr = q_head_ptr + (offset_m * stride_qh)[:, None] + (offset_k * stride_qk)[None, :]

    o_block = tl.zeros([block_r, head_dim], dtype=tl.float32)
    l_i = tl.zeros([block_r], dtype=tl.float32)
    m_i = tl.zeros([block_r], dtype=tl.float32) - float("inf")

    k_block_ptr = k_head_ptr + (offset_n * stride_kh)[None, :] + (offset_k * stride_kk)[:, None]  # 转置加载k
    v_block_ptr = v_head_ptr + (offset_n * stride_vh)[:, None] + offset_k[None, :] * stride_vk

    q_block = tl.load(q_block_ptr)

    for i in range(0, kvlens, block_c):
        k_block = tl.load(k_block_ptr + i * stride_kh)
        v_block = tl.load(v_block_ptr + i * stride_vh)

        s = tl.dot(q_block, k_block) * softmax_scale

        m_ij = tl.max(s, 1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)

        p = tl.exp(s - m_new[:, None])
        p_fp16 = p.to(v_block.dtype)
        l_i = l_i * alpha + tl.sum(p, 1)
        o_block = o_block * alpha[:, None] + tl.dot(p_fp16, v_block)

        m_i = m_new

    o_block = o_block / l_i[:, None]

    o_block_ptr = o_head_ptr + (offset_m * stride_oh)[:, None] + offset_k[None, :] * stride_ok

    tl.store(o_block_ptr, o_block.to(o_ptr.dtype.element_ty))

    lse_idx = head_idx * qlens + q_block_idx * block_r + tl.arange(0, block_r)
    tl.store(lse_ptr + lse_idx, m_i + tl.log(l_i))


# multi-head attention_fa2
def attention_fa2(q, k, v):
    bs, h, qlens, d = q.shape
    _, _, kvlens, _ = k.shape

    q = q.view(bs * h, qlens, d)
    k = k.view(bs * h, kvlens, d)
    v = v.view(bs * h, kvlens, d)

    n = bs * h

    # block_q = 64
    block_r = 64
    block_c = 64
    grid = (bs * h, triton.cdiv(qlens, block_r))

    softmax_scale = 1.0 / (d**0.5)

    o = torch.empty_like(q)
    lse = torch.empty((bs * h, qlens), dtype=torch.float32, device=q.device)

    _fa2_fwd[grid](
        q,
        k,
        v,
        o,
        lse,
        n,
        qlens,
        kvlens,
        d,
        softmax_scale,
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
        block_r,
        block_c,
    )
    return o.view(bs, h, qlens, d)


bs = 2
qlens = 128
kvlens = 256
h = 12
d = 128


DEVICE = triton.runtime.driver.active.get_active_torch_device()
# DEVICE = "cpu"
dtype = torch.float16
Q = torch.randn(bs, h, qlens, d, device=DEVICE, dtype=dtype)
K = torch.randn(bs, h, kvlens, d, device=DEVICE, dtype=dtype)
V = torch.randn(bs, h, kvlens, d, device=DEVICE, dtype=dtype)


def native(q, k, v):
    # 简单的 PyTorch 实现用于验证
    # q: [B, H, L, D]
    scale = d**-0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


def spda(q, k, v):
    O = F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=False,
    )
    return O


triton_out = attention_fa2(Q, K, V)

native_out = native(Q, K, V)
spda_out = spda(Q, K, V)

# for i in range(bs):
#     for j in range(h):
#         for k in range(qlens):
#             if torch.allclose(triton_out[i, j, k, :], native_out[i, j, k, :], atol=4e-2, rtol=4e-2):
#                 continue
#             else:
#                 print(f"-----------------{i} {j} {k}")
#                 print(f"max abs diff {torch.max(torch.abs(triton_out[i, j, k, :] - native_out[i, j, k, :]))}")
#                 print(triton_out[i, j, k, :] - native_out[i, j, k, :])
#                 print("Triton Out Sample:", triton_out[i, j, k, :])
#                 print("Native Out Sample:", native_out[i, j, k, :])
#                 exit()


# print("Triton Out Sample:", triton_out[0, 0, 0, :])
# print("Native Out Sample:", native_out[0, 0, 0, :])

# 误差验证
diff = torch.max(torch.abs(triton_out - native_out))
print(f"Max Diff (Native): {diff}")

diff_spda = torch.max(torch.abs(triton_out - spda_out))
print(f"Max Diff (SPDA): {diff_spda}")

assert torch.allclose(triton_out, native_out, atol=1e-2, rtol=1e-2), "Mismatch with Native implementation"
print("Test Passed!")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["S"],
        x_vals=[128 * i for i in range(2, 20)],
        line_arg="provider",
        line_vals=["naive", "torch", "triton", ],
        line_names=["naive","Torch", "Triton", ],
        styles=[("red", "-"), ("green", "-"), ("blue", "-"), ("yellow", "-")],
        ylabel="TFLOPS/s",
        plot_name="attention_performance",
        args={'BS' : 1, 'H' : 16, 'D' : 128},
    )
)
def benchmark(BS, H, S, D, provider):
    Q = torch.randn(BS, H, S, D, device=DEVICE, dtype=dtype)
    K = torch.randn(BS, H, S, D, device=DEVICE, dtype=dtype)
    V = torch.randn(BS, H, S, D, device=DEVICE, dtype=dtype)
    if provider == "naive":
        ms = triton.testing.do_bench(lambda: native(Q, K, V))
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: spda(Q, K, V))
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: attention_fa2(Q, K, V))

    flops_per_matmul = 2.0 * BS * H * S * S * D
    total_flops = 2 * flops_per_matmul

    tfps = lambda ms: total_flops * 1e-12 / (ms * 1e-3)

    return tfps(ms)

benchmark.run(show_plots=False, print_data=True, save_path='attention_fa2')