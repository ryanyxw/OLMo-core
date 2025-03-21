import torch

from olmo_core.float8.utils import cast_from_fp8, cast_to_fp8, per_block_cast_to_fp8


def test_cast_to_fp8():
    x = torch.randn(2, 3, 3 * 128)
    x_fp8, s = cast_to_fp8(x)
    assert x_fp8.shape == x.shape
    assert s.shape == (2, 3, 3)
    y = cast_from_fp8(x_fp8, s, dtype=x.dtype)
    torch.testing.assert_close(x, y)


def test_per_block_cast_to_fp8():
    x = torch.randn(8, 3 * 128, 2 * 128)
    x_fp8, s = per_block_cast_to_fp8(x)
    assert x_fp8.shape == x.shape
    assert s.shape == (8, 3, 2)
