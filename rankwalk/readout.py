"""Optional subject-level readouts; node-embedding training is unchanged."""
import torch
import torch.nn.functional as F


def pool_trajectories(embeddings, subject_ids, times=None, *, mode='mean',
                      n_timepoints=4, time_grid=None, raw_trajectories=None,
                      return_subject_ids=False):
    """Pool Torch node embeddings into sorted subject order, retaining device/dtype.

    ``mean`` returns the ordinary unnormalized mean and keeps the embedding width.
    ``ordered`` concatenates L2-normalized blocks: mean embedding, embeddings
    interpolated at an absolute-time grid (feature-major order), and optionally
    flattened raw trajectories (time-major order). With embedding width D, four
    grid points and raw shape (subjects, 16, P), its width is 5*D + 16*P.

    By default ordered pooling requires times normalized to [0, 1]. Supply an
    explicit increasing ``time_grid`` for physical units. Interpolation is linear
    with constant endpoint extrapolation. At duplicate times the last node in
    input order is used. A subject with one node is constant across the grid.

    Raw arrays must already be prepared/scaled without prohibited future data,
    and rows must follow the sorted subject IDs. This function does not fit a
    scaler or construct raw trajectories. Set ``return_subject_ids=True`` to get
    ``(pooled_embeddings, sorted_ids)`` for explicit alignment.
    """
    if mode not in ('mean', 'ordered'):
        raise ValueError("mode must be 'mean' or 'ordered'")
    if embeddings.ndim != 2 or len(embeddings) == 0 or not torch.is_floating_point(embeddings):
        raise ValueError('embeddings must be a non-empty floating-point matrix')
    if not torch.isfinite(embeddings).all():
        raise ValueError('embeddings must be finite')
    ids = torch.as_tensor(subject_ids, device=embeddings.device)
    if ids.ndim != 1 or len(ids) != len(embeddings) or not torch.isfinite(ids).all():
        raise ValueError('subject_ids must contain one finite numeric ID per node')
    subjects, inverse = torch.unique(ids, sorted=True, return_inverse=True)
    sums = embeddings.new_zeros((len(subjects), embeddings.shape[1]))
    sums.index_add_(0, inverse, embeddings)
    mean = sums / torch.bincount(inverse).to(embeddings.dtype)[:, None]
    if mode == 'mean':
        if raw_trajectories is not None:
            raise ValueError("raw_trajectories requires mode='ordered'")
        result = mean
    else:
        if times is None:
            raise ValueError("times are required for mode='ordered'")
        time = torch.as_tensor(times, device=embeddings.device)
        if not torch.is_floating_point(time):
            time = time.to(torch.float64)
        if time.ndim != 1 or len(time) != len(embeddings) or not torch.isfinite(time).all():
            raise ValueError('times must contain one finite value per node')
        if time_grid is None:
            if not isinstance(n_timepoints, int) or n_timepoints < 1:
                raise ValueError('n_timepoints must be a positive integer')
            if (time < 0).any() or (time > 1).any():
                raise ValueError('Normalize times to [0, 1] or supply an explicit time_grid')
            grid = torch.linspace(0, 1, n_timepoints, device=time.device, dtype=time.dtype)
        else:
            grid = torch.as_tensor(time_grid, device=time.device, dtype=time.dtype)
            if grid.ndim != 1 or not len(grid) or not torch.isfinite(grid).all() or not (grid[1:] > grid[:-1]).all():
                raise ValueError('time_grid must be a finite, strictly increasing vector')
        ordered = []
        for i in range(len(subjects)):
            mask = inverse == i
            t, e = time[mask], embeddings[mask]
            order = torch.argsort(t, stable=True)
            t, e = t[order].contiguous(), e[order]
            right = torch.searchsorted(t, grid, right=True)
            lo, hi = (right - 1).clamp(0, len(t) - 1), right.clamp(0, len(t) - 1)
            delta = t[hi] - t[lo]
            safe = torch.where(delta != 0, delta, torch.ones_like(delta))
            weight = ((grid - t[lo]) / safe).clamp(0, 1).to(embeddings.dtype)
            interpolated = e[lo] + weight[:, None] * (e[hi] - e[lo])
            ordered.append(interpolated.T.reshape(-1))
        # The default 1e-12 underflows to zero in float16.
        eps = torch.finfo(embeddings.dtype).eps if embeddings.dtype == torch.float16 else 1e-12
        blocks = [F.normalize(mean, dim=1, eps=eps),
                  F.normalize(torch.stack(ordered), dim=1, eps=eps)]
        if raw_trajectories is not None:
            raw = torch.as_tensor(raw_trajectories, device=embeddings.device, dtype=embeddings.dtype)
            if raw.ndim != 3 or raw.shape[0] != len(subjects) or min(raw.shape[1:]) < 1 or not torch.isfinite(raw).all():
                raise ValueError('raw_trajectories must be finite with shape (subjects, time, features) in sorted subject order')
            blocks.append(F.normalize(raw.reshape(len(subjects), -1), dim=1, eps=eps))
        result = torch.cat(blocks, dim=1)
    return (result, subjects) if return_subject_ids else result
