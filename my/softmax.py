import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl
from triton.runtime import driver

import pytest
import logging


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()


DEVICE = triton.runtime.driver.active.get_active_torch_device()

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

logger.info(f"{NUM_SM=}")
logger.info(f"{NUM_REGS=}")
logger.info(f"{SIZE_SMEM=}")
logger.info(f"{WARP_SIZE=}")


@triton.jit
def _softmax(x_ptr, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    # row = tl.program_id(0)
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):  # , num_stages=num_stages
        x_row_ptr = x_ptr + row_idx * n_cols
        output_row_ptr = output_ptr + row_idx * n_cols

        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        input_ptrs = x_row_ptr + col_offsets

        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row)
        row_minus_max_exp = tl.exp(row_minus_max)
        y = row_minus_max_exp / tl.sum(row_minus_max_exp)
        tl.store(output_row_ptr + tl.arange(0, BLOCK_SIZE), y, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    _softmax[grid](x, y, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y


def softmax_search(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    kernel = _softmax.warmup(x, y, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, grid=(1,))

    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    occupancy_reg = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy_reg, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy
    num_programs = min(num_programs, n_rows)

    grid = lambda meta: (num_programs,)
    _softmax[grid](x, y, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y


@triton.jit
def _softmax_stage(x_ptr, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    # row = tl.program_id(0)
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):  #
        x_row_ptr = x_ptr + row_idx * n_cols
        output_row_ptr = output_ptr + row_idx * n_cols

        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        input_ptrs = x_row_ptr + col_offsets

        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row)
        row_minus_max_exp = tl.exp(row_minus_max)
        y = row_minus_max_exp / tl.sum(row_minus_max_exp)
        tl.store(output_row_ptr + tl.arange(0, BLOCK_SIZE), y, mask=mask)


def softmax_stage(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    kernel = _softmax_stage.warmup(
        x, y, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages, num_warps=num_warps, grid=(1,)
    )

    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    occupancy_reg = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy_reg, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy
    num_programs = min(num_programs, n_rows)

    grid = lambda meta: (num_programs,)
    _softmax_stage[grid](x, y, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages)
    return y


import torch.profiler

# 1. Define the handler to save traces
# trace_handler = torch.profiler.tensorboard_trace_handler('./profiler_results')

# # 2. Configure the profiler
# with torch.profiler.profile(
#     activities=[
#         torch.profiler.ProfilerActivity.CPU,
#         torch.profiler.ProfilerActivity.CUDA,  # Tracks GPU kernels
#     ],
#     schedule=torch.profiler.schedule(
#         wait=1,      # Skip first N steps
#         warmup=2,    # Warm up for N steps (jit, etc.)
#         active=1,    # Record for N steps
#         repeat=1     # How many times to repeat this cycle
#     ),
#     on_trace_ready=trace_handler,  # Automatically saves to disk
#     record_shapes=True,            # Optional: saves tensor dimensions
#     with_stack=True,               # Optional: records source code line
#     profile_memory=True            # Optional: tracks memory allocation
# ) as prof:

#     for i in range(5):
#         for (m,n) in [(128, 128), (128, 256), (128, 512), (128, 1024), (128, 2048), (128, 4096)]:
#             x = torch.randn(m, n, device=DEVICE)
#             golden = F.softmax(x, dim=1)
#             res = softmax(x)
#         prof.step()

m = 1024
n = 128
x = torch.randn(m, n, device=DEVICE)
# golden = F.softmax(x, dim=1)
res = softmax(x)

# print(res)
# print(golden)

# logger.info(f"max diff {torch.max(torch.abs(res - golden))}")
# if not torch.allclose(res, golden):
#     logger.error(f"shape: [{m, n}], max diff: {torch.max(torch.abs(res - golden))}")

# torch.testing.assert_close(res, golden, rtol=1e-5, atol=1e-5)

# TEST_SHAPES = [
#     (1, 10),      # 单个向量
#     (16, 128),    # 常规 batch
#     (64, 1000),   # 类似 ImageNet 分类层
#     # (128, 4096)   # 大维度
# ]

# @pytest.mark.parametrize("m, n", TEST_SHAPES)
# def test_softmax_consistency(m, n):
#     torch.manual_seed(0)
#     x = torch.randn(m, n, device=DEVICE)
#     golden = F.softmax(x, dim=1)
#     res = softmax(x)
#     # print(res)
#     # print(golden)
#     if not torch.allclose(res, golden):
#         logger.error(f"shape: [{m,n}], max diff: {torch.max(torch.abs(res - golden))}")

#     torch.testing.assert_close(res, golden, rtol=1e-5, atol=1e-5)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(2, 40)],
        line_arg="provider",
        line_vals=["torch", "triton", "triton_search", "triton_search_stage"],
        line_names=["Torch", "Triton", "triton_search", "triton_search_stage"],
        styles=[("red", "-"), ("green", "-"), ("blue", "-"), ("yellow", "-")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M": 1024},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: F.softmax(x, dim=1))
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == "triton_search":
        ms = triton.testing.do_bench(lambda: softmax_search(x))
    if provider == "triton_search_stage":
        ms = triton.testing.do_bench(lambda: softmax_stage(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


# benchmark.run(show_plots=True, print_data=True, save_path='softmax')
