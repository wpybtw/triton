import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    a_offset = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    b_offset = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_block_ptr = a_ptr + a_offset[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_block_ptr = b_ptr + offs_k[:, None] * stride_bk + b_offset[None, :] * stride_bn

    offset_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    c_block_ptr = c_ptr + offset_cm[:, None] * stride_cm + offset_cn[None, :] * stride_cn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a_block = tl.load(a_block_ptr, mask=offs_k[None, :] < K - k, other=0.0)
        b_block = tl.load(b_block_ptr, mask=offs_k[:, None] < K - k, other=0.0)
        acc = tl.dot(a_block, b_block, acc)
        a_block_ptr += BLOCK_SIZE_K * stride_ak
        b_block_ptr += BLOCK_SIZE_K * stride_bk
    c_mask = (offset_cm[:, None] < M) & (offset_cn[None, :] < N)
    
    acc_bf16 = acc.to(tl.float16)

    tl.store(c_block_ptr, acc_bf16, mask=c_mask)


def matmul(A, B, block_size_m=128, block_size_n=128, block_size_k=64):
    M, K = A.shape
    Kb, N = B.shape

    assert K == Kb, "Inner dimensions must match"
    assert A.stride(-1) == 1, "A must be contiguous"

    C = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(N, META["BLOCK_SIZE_N"]))
    matmul_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
    )
    return C


m = 384
n = 256
k = 512



A = torch.rand((m, k), device=DEVICE, dtype=torch.float16)
B = torch.rand((k, n), device=DEVICE, dtype=torch.float16)
golden = torch.matmul(A, B)

C = matmul(A, B, block_size_m=128, block_size_n=128, block_size_k=64)

print(f'max diff: {torch.max(torch.abs(C - golden))}')
print(torch.allclose(C, golden))

print(' B transposed ')

A = torch.rand((m, k), device=DEVICE, dtype=torch.float16)
B = torch.rand((n, k), device=DEVICE, dtype=torch.float16)
golden = torch.matmul(A, B.t())

C = matmul(A, B.t(), block_size_m=128, block_size_n=128, block_size_k=64)

print(f'max diff: {torch.max(torch.abs(C - golden))}')
print(torch.allclose(C, golden))