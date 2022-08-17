import math

import torch
import torch.fft as fft
import torch.nn.functional as F  # noqa


def pad_to(tensor: torch.Tensor, target_length: int, mode: str = "constant", value: float = 0):
    """Pad the given tensor to the given length, with 0s on the right."""
    return F.pad(tensor, (0, target_length - tensor.shape[-1]), mode=mode, value=value)


def unfold(x, kernel_size: int, stride: int):
    """1D only unfolding similar to the one from PyTorch. However, PyTorch unfold is extremely
    slow. Given an input tensor of size `[*, T]` this will return a tensor `[*, F, K]` with `K` the
    kernel size, and `F` the number.

    of frames. The i-th frame is a view onto `i * stride: i * stride + kernel_size`.
    This will automatically pad the input to cover at least once all entries in `input`.
    Args:
        x (Tensor): tensor for which to return the frames.
        kernel_size (int): size of each frame.
        stride (int): stride between each frame.
    Shape:
        - Inputs: `input` is `[*, T]`
        - Output: `[*, F, kernel_size]` with `F = 1 + ceil((T - kernel_size) / stride)`
    ..Warning:: unlike PyTorch unfold, this will pad the input
        so that any position in `input` is covered by at least one frame.
    """
    shape = list(x.shape)
    length = shape.pop(-1)
    n_frames = math.ceil((max(length, kernel_size) - kernel_size) / stride) + 1
    tgt_length = (n_frames - 1) * stride + kernel_size
    padded = F.pad(x, (0, tgt_length - length)).contiguous()
    strides = []
    for dim in range(padded.dim()):
        strides.append(padded.stride(dim))
    assert strides.pop(-1) == 1, "data should be contiguous"
    strides = strides + [stride, 1]
    return padded.as_strided(shape + [n_frames, kernel_size], strides)


def fft_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride: int = 1,
    padding: int = 0,
    block_ratio: float = 5,
):
    """
    Same as `torch.nn.functional.conv1d` but using FFT for the convolution.
    Please check PyTorch documentation for more information.
    Args:
        x (Tensor): input signal of shape `[B, C, T]`.
        weight (Tensor): weight of the convolution `[D, C, K]` with `D` the number
            of output channels.
        bias (Tensor or None): if not None, bias term for the convolution.
        stride (int): stride of convolution.
        padding (int): padding to apply to the input.
        block_ratio (float): can be tuned for speed. The input is split in chunks
            with a size of `int(block_ratio * kernel_size)`.
    Shape:
        - Inputs: `input` is `[B, C, T]`, `weight` is `[D, C, K]` and bias is `[D]`.
        - Output: `(*, T)`
    ..note::
        This function is faster than `torch.nn.functional.conv1d` only in specific cases.
        Typically, the kernel size should be of the order of 256 to see any real gain,
        for a stride of 1.
    ..Warning::
        Dilation and groups are not supported at the moment. This function might use
        more memory than the default Conv1d implementation.
    """
    x = F.pad(x, (padding, padding))
    batch, channels, length = x.shape
    out_channels, _, kernel_size = weight.shape

    if length < kernel_size:
        raise RuntimeError(
            f"Input should be at least as large as the kernel size {kernel_size}, "
            f"but it is only {length} samples long."
        )
    if block_ratio < 1:
        raise RuntimeError("Block ratio must be greater than 1.")

    # We are going to process the input blocks by blocks, as for some reason it is faster
    # and less memory intensive (I think the culprit is `torch.einsum`).
    block_size: int = min(int(kernel_size * block_ratio), length)
    fold_stride = block_size - kernel_size + 1
    weight = pad_to(weight, block_size)
    weight_z = fft.rfft(weight)

    # We pad the input and get the different frames, on which
    frames = unfold(x, block_size, fold_stride)

    frames_z = fft.rfft(frames)
    out_z = frames_z * weight_z.conj()
    out = fft.irfft(out_z, block_size)
    # The last bit is invalid, because FFT will do a circular convolution.
    out = out[..., : -kernel_size + 1]
    out = out.reshape(batch, out_channels, -1)
    out = out[..., ::stride]
    target_length = (length - kernel_size) // stride + 1

    # TODO: this line throws away the tail. Will be necessary for real-time synth.
    out = out[..., :target_length]
    if bias is not None:
        out += bias[:, None]
    return out
