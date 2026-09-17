"""Height queries on an unshifted regular triangle grid (IsaacGym diagonal 00--11)."""
import torch


def sample_triangle_heights(height_samples, xy, horizontal_scale, vertical_scale, border_size):
    grid = (xy + border_size) / horizontal_scale
    x = grid[..., 0].clamp(0, height_samples.shape[0] - 1)
    y = grid[..., 1].clamp(0, height_samples.shape[1] - 1)
    # NaN survives floating-point clamp and converts to INT64_MIN. Keep the
    # lookup safe while preserving NaN in the result for the caller to detect.
    ix = x.floor().long().clamp(0, height_samples.shape[0] - 2)
    iy = y.floor().long().clamp(0, height_samples.shape[1] - 2)
    u, v = x - ix, y - iy
    h00 = height_samples[ix, iy].to(x.dtype)
    h10 = height_samples[ix + 1, iy].to(x.dtype)
    h01 = height_samples[ix, iy + 1].to(x.dtype)
    h11 = height_samples[ix + 1, iy + 1].to(x.dtype)
    return torch.where(u >= v,
                       (1 - u) * h00 + (u - v) * h10 + v * h11,
                       (1 - v) * h00 + (v - u) * h01 + u * h11) * vertical_scale
