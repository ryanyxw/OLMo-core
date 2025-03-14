import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

VEC_SIZE = 128


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 1 * VEC_SIZE,
                "BLOCK_SIZE_N": 1 * VEC_SIZE,
                "BLOCK_SIZE_K": 1 * VEC_SIZE,
                "GROUP_SIZE_M": 8,
                "VEC_SIZE": VEC_SIZE,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 1 * VEC_SIZE,
                "BLOCK_SIZE_N": 1 * VEC_SIZE,
                "BLOCK_SIZE_K": 1 * VEC_SIZE,
                "GROUP_SIZE_M": 8,
                "VEC_SIZE": VEC_SIZE,
            },
            num_stages=3,
            num_warps=8,
        ),
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel_fp8_fp8_bf16(
    # Pointers to matrices
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
    stride_bk,
    stride_bn,
    stride_bsk,
    stride_bsn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    - A has shape (M, K) and scale with shape (M, K // 128)
    - B has shape (K, N) and scale with shape (K // 128, N // 128)
    - C has shape (M, N)
    """
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B and the corresponding scales.
    # We'll advance these pointers as we move in the K direction and accumulate.
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers.
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers.
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_sm = (pid_m * (BLOCK_SIZE_M // VEC_SIZE) + tl.arange(0, BLOCK_SIZE_M // VEC_SIZE)) % M
    offs_sn = (pid_n * (BLOCK_SIZE_N // VEC_SIZE) + tl.arange(0, BLOCK_SIZE_N // VEC_SIZE)) % N
    # TODO: this isn't right
    a_scale_ptrs = a_scale_ptr + (offs_sm[:, None] * stride_asm + offs_k[None, :] * stride_ask)
    b_scale_ptrs = b_scale_ptr + (offs_k[:, None] * stride_bsk + offs_sn[None, :] * stride_bsn)

    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block of FP32 values for higher accuracy.
    # `accumulator` will be converted back to BF16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A, B, and their scales, then generate a mask by checking the K dimension.
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a_scale = tl.load(a_scale_ptrs)
        b_scale = tl.load(b_scale_ptrs)

        # We accumulate along the K dimension.
        accumulator = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", accumulator)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        a_scale_ptrs += (BLOCK_SIZE_K // VEC_SIZE) * stride_ask
        b_scale_ptrs += (BLOCK_SIZE_K // VEC_SIZE) * stride_bsk

    c = accumulator.to(tl.bfloat16)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_fp8_fp8_bf16(
    a: torch.Tensor, a_scale: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor
) -> torch.Tensor:
    # Check constraints.
    assert a.dim() == 2, "Expected LHS to be a 2D tensor"
    assert b.dim() == 2, "Expected RHS to be a 2D tensor"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "LHS must be contiguous"
    M, K = a.shape
    K, N = b.shape
    assert (
        M % VEC_SIZE == 0 and K % VEC_SIZE == 0 and N % VEC_SIZE == 0
    ), f"All dimensions must be multiples of {VEC_SIZE}"

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    matmul_kernel_fp8_fp8_bf16[grid](  # type: ignore
        a,
        a_scale,
        b,
        b_scale,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        b.stride(0),
        b.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        c.stride(0),
        c.stride(1),
    )

    return c


if __name__ == "__main__":
    from olmo_core.float8.utils import cast_to_fp8, per_block_cast_to_fp8

    device = torch.device("cuda")
    torch.manual_seed(0)

    a = torch.randn((256, 512), device=device, dtype=torch.bfloat16)
    b = torch.randn((256, 512), device=device, dtype=torch.bfloat16)
    a_fp8, a_scale = cast_to_fp8(a)
    b_fp8, b_scale = per_block_cast_to_fp8(b)

    # Pre-transpose B for efficiency.
    b, b_fp8, b_scale = b.T, b_fp8.T, b_scale.T

    triton_output = matmul_fp8_fp8_bf16(a_fp8, a_scale, b_fp8, b_scale)
    torch_output = torch.matmul(a, b)
    assert torch.allclose(triton_output, torch_output, atol=0.125, rtol=0)
