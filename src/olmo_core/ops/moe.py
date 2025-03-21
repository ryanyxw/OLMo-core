from typing import Any, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from olmo_core.utils import move_to_device

from .utils import cast_to

try:
    from olmo_core.kernels import moe as kernels
except ImportError:
    kernels = None  # type: ignore


def _is_eligible(x):
    return x.is_floating_point() and x.is_cuda and (x.dtype is not torch.float64)


def _cast(x, dtype):
    if isinstance(x, torch.Tensor) and _is_eligible(x):
        return x.to(dtype)
    elif isinstance(x, dict):
        return {_cast(k, dtype): _cast(v, dtype) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(map(lambda y: _cast(y, dtype), x))
    return x


class GatherOp(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
    ):
        assert kernels is not None
        ctx.save_for_backward(indices, bin_ids, bins)
        ctx.top_k = top_k
        return kernels.gather(x, indices, bin_ids, None, bins, top_k)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad: torch.Tensor):
        assert kernels is not None
        grad = grad.contiguous()
        indices, bin_ids, bins = ctx.saved_tensors
        out = kernels.scatter(grad, indices, bin_ids, None, bins, ctx.top_k)
        return out, None, None, None, None, None


def gather(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return GatherOp.apply(x, indices, bin_ids, bins, top_k)  # type: ignore


class ScatterOp(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        weights: Optional[torch.Tensor],
        bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        assert kernels is not None
        maybe_x = [x] if ctx.needs_input_grad[3] else []
        ctx.save_for_backward(indices, bin_ids, weights, bins, *maybe_x)
        ctx.top_k = top_k
        ctx.x_shape = x.shape
        return kernels.scatter(x, indices, bin_ids, weights, bins, top_k)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad: torch.Tensor):
        assert kernels is not None

        grad = grad.contiguous()
        saved_tensors = ctx.saved_tensors

        indices, bin_ids, weights, bins = saved_tensors[:4]
        dgrad = None
        if ctx.needs_input_grad[0]:
            dgrad = kernels.gather(
                grad,
                indices,
                bin_ids,
                weights,
                bins,
                ctx.top_k,
            )

        wgrad = None
        if ctx.needs_input_grad[3]:  # need wgrad
            x = saved_tensors[-1]
            wgrad = kernels.scatter_wgrad(
                x,
                grad,
                indices,
                bin_ids,
                bins,
                ctx.top_k,
            )
        return dgrad, None, None, wgrad, None, None, None


def scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: Optional[torch.Tensor],
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return ScatterOp.apply(x, indices, bin_ids, weights, bins, top_k)  # type: ignore


class BinnedGatherOp(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bins: torch.Tensor,
        bin_size: int,
        top_k: int,
    ):
        assert kernels is not None
        ctx.save_for_backward(indices, bins)
        ctx.top_k = top_k
        return kernels.binned_gather(x, indices, None, bins, bin_size, top_k)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad: torch.Tensor):
        assert kernels is not None
        grad = grad.contiguous()
        indices, bins = ctx.saved_tensors
        out = kernels.binned_scatter(grad, indices, None, bins, ctx.top_k)
        return out, None, None, None, None


def binned_gather(
    x: torch.Tensor, indices: torch.Tensor, bins: torch.Tensor, bin_size: int, top_k: int
) -> torch.Tensor:
    return BinnedGatherOp.apply(x, indices, bins, bin_size, top_k)  # type: ignore


class BinnedScatterOp(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: Optional[torch.Tensor],
        bins: torch.Tensor,
        top_k: int,
    ):
        assert kernels is not None

        assert len(x.size()) == 3
        ctx.bin_size = x.size(1)
        ctx.top_k = top_k

        # TODO: Don't save 'x' for backwards if we don't need to
        # calculate the gradient w.r.t. 'weights'.
        ctx.save_for_backward(x, indices, weights, bins)
        return kernels.binned_scatter(x, indices, weights, bins, top_k)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad: torch.Tensor):
        assert kernels is not None

        grad = grad.contiguous()
        x, indices, weights, bins = ctx.saved_tensors
        out = kernels.binned_gather(
            grad,
            indices,
            weights,
            bins,
            ctx.bin_size,
            ctx.top_k,
        )

        wgrad = None
        if ctx.needs_input_grad[2]:
            wgrad = kernels.binned_scatter_wgrad(
                x,
                grad,
                indices,
                bins,
                ctx.top_k,
            )
        return out, None, wgrad, None, None


def binned_scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: Optional[torch.Tensor],
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return BinnedScatterOp.apply(x, indices, weights, bins, top_k)  # type: ignore


def repeat(x: torch.Tensor, tiling: Union[torch.Size, Tuple[int, ...]]) -> torch.Tensor:
    if all((t == 1 for t in tiling)):
        return x
    return x.repeat(*tiling)


class AllToAllOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        output_split_sizes: Optional[List[int]],
        input_split_sizes: Optional[List[int]],
        group: Optional[dist.ProcessGroup],
        async_op: bool,
        dtype: Optional[torch.dtype],
    ):
        og_dtype = x.dtype
        if dtype is None:
            dtype = og_dtype

        # Maybe cast ``x`` to the target communication dtype.
        x, scale = cast_to(x, dtype)

        # If ``x`` was downcast to, say, FP8, we need to all-to-all the ``scale``.
        scale_handle: Optional[dist.Work] = None
        out_scale: Optional[torch.Tensor] = None
        if scale is not None:
            if output_split_sizes is not None:
                out_scale = torch.empty(
                    (sum(output_split_sizes),) + scale.shape[1:], device=x.device, dtype=scale.dtype
                )
            else:
                out_scale = torch.empty_like(scale)

            scale_handle = dist.all_to_all_single(
                out_scale,
                scale,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True,
            )

        # Now all-to-all ``x``.
        if output_split_sizes is not None:
            out = torch.empty(
                (sum(output_split_sizes),) + x.shape[1:], device=x.device, dtype=dtype
            )
        else:
            out = torch.empty_like(x, dtype=dtype)

        handle = dist.all_to_all_single(
            out,
            x,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
        assert handle is not None

        ctx.input_scale_shape = None if scale is None else scale.shape
        ctx.input_shape = x.shape
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group
        ctx.dtype = dtype

        # Cast ``out`` back to input dtype if needed.
        if out.dtype != og_dtype:
            if scale_handle is not None:
                scale_handle.wait()
            handle.wait()
            out, _ = cast_to(out, og_dtype, out_scale)

        if not async_op:
            handle.wait()
            handle = None

        return out, handle

    @staticmethod
    def backward(ctx, grad, _):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None

        og_dtype = grad.dtype
        dtype = ctx.dtype

        # Maybe cast ``grad`` to the target communication dtype.
        grad, grad_scale = cast_to(grad, dtype)

        # If ``grad`` was downcast to, say, FP8, we need to all-to-all the ``scale``.
        scale_handle: Optional[dist.Work] = None
        out_scale: Optional[torch.Tensor] = None
        if grad_scale is not None:
            assert ctx.input_scale_shape is not None
            out_scale = torch.empty(
                ctx.input_scale_shape, device=grad.device, dtype=grad_scale.dtype
            )
            scale_handle = dist.all_to_all_single(
                out_scale,
                grad_scale,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
                async_op=True,
            )

        # All-to-all the gradient.
        out = torch.empty(
            ctx.input_shape,
            device=grad.device,
            dtype=dtype,
        )
        dist.all_to_all_single(
            out,
            grad,
            output_split_sizes=ctx.input_split_sizes,
            input_split_sizes=ctx.output_split_sizes,
            group=ctx.group,
        )

        # Cast output back to gradient dtype if needed.
        if out.dtype != og_dtype:
            if scale_handle is not None:
                scale_handle.wait()
            out, _ = cast_to(out, og_dtype, out_scale)

        return out, None, None, None, None, None


def all_to_all(
    x: torch.Tensor,
    output_split_sizes: Optional[List[int]] = None,
    input_split_sizes: Optional[List[int]] = None,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, dist.Work]:
    return AllToAllOp.apply(  # type: ignore
        x,
        output_split_sizes,
        input_split_sizes,
        group,
        async_op,
        dtype,
    )


def sum_tensor(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if x.shape[dim] == 1:
        return x.squeeze(dim=dim)
    return x.sum(dim=dim)


def batched_histc(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    A batched version of ``torch.histc``.
    """
    hist = move_to_device(torch.zeros((*x.shape[:-1], num_classes), dtype=x.dtype), x.device)
    ones = move_to_device(torch.tensor(1, dtype=x.dtype), x.device).expand_as(x)
    hist.scatter_add_(-1, ((x * num_classes) // (x.max() + 1)).long(), ones)
    return hist
