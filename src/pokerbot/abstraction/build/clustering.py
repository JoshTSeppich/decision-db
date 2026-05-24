"""K-means clustering for postflop bucketing (Spec.html §A).

Two flavors:
    - `kmeans_euclidean`: standard L2 distance — used for OCHS (8-d feature vectors)
    - `kmeans_cdf_emd`: 1-D Earth Mover's Distance on histograms with *ordered*
      bins. EMD(p, q) for ordered bins equals the L1 distance between their
      CDFs, which is fast to compute and cheap to vectorise. Used for
      potential-aware EHS² histograms (50-d feature vectors on flop/turn).

Spec calls this "1D-sort EMD" but the correct 1-D EMD on ordered bins is the
CDF-L1 form — sorting individual bin values would lose the EHS²-ordering that
makes these histograms informative.

Both clusterers use k-means++ init + Lloyd iterations, with deterministic numpy
seeding for reproducible builds.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _kmeanspp_init(
    features: npt.NDArray[np.float32], k: int, rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    """K-means++ initialization. Picks `k` rows of `features` as initial centroids."""
    n = features.shape[0]
    if n < k:
        raise ValueError(f"cannot pick {k} centroids from {n} points")
    centers_idx = [int(rng.integers(n))]
    dists_sq = np.sum((features - features[centers_idx[0]]) ** 2, axis=1)
    for _ in range(k - 1):
        probs = dists_sq / max(dists_sq.sum(), 1e-12)
        nxt = int(rng.choice(n, p=probs))
        centers_idx.append(nxt)
        new_d = np.sum((features - features[nxt]) ** 2, axis=1)
        dists_sq = np.minimum(dists_sq, new_d)
    return features[centers_idx].copy()


def _assign_euclidean(
    features: npt.NDArray[np.float32], centroids: npt.NDArray[np.float32]
) -> npt.NDArray[np.int32]:
    # batch L2 to avoid materializing full distance matrix when features is huge.
    n = features.shape[0]
    k = centroids.shape[0]
    chunk = 4096
    out = np.zeros(n, dtype=np.int32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = features[start:end, None, :] - centroids[None, :, :]
        d = np.einsum("ijk,ijk->ij", diff, diff)
        out[start:end] = d.argmin(axis=1).astype(np.int32)
    _ = k  # silence unused
    return out


def _assign_l1(
    features: npt.NDArray[np.float32],
    centroids: npt.NDArray[np.float32],
) -> npt.NDArray[np.int32]:
    """L1 (Manhattan) distance — for CDF-based EMD assignment."""
    n = features.shape[0]
    out = np.zeros(n, dtype=np.int32)
    chunk = 2048
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = features[start:end, None, :] - centroids[None, :, :]
        d = np.abs(diff).sum(axis=2)
        out[start:end] = d.argmin(axis=1).astype(np.int32)
    return out


def _update_centroids(
    features: npt.NDArray[np.float32],
    assignments: npt.NDArray[np.int32],
    k: int,
) -> npt.NDArray[np.float32]:
    dim = features.shape[1]
    new_centroids = np.zeros((k, dim), dtype=np.float32)
    counts = np.zeros(k, dtype=np.int32)
    np.add.at(new_centroids, assignments, features)
    np.add.at(counts, assignments, 1)
    nonempty = counts > 0
    new_centroids[nonempty] /= counts[nonempty, None]
    # Empty centroids: reseed to a random feature row to avoid collapse.
    if not nonempty.all():
        rng = np.random.default_rng(0)
        for i in np.where(~nonempty)[0]:
            new_centroids[i] = features[int(rng.integers(features.shape[0]))]
    return new_centroids


def kmeans_euclidean(
    features: npt.NDArray[np.float32],
    k: int,
    *,
    iters: int = 20,
    seed: int = 0xC0FFEE,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """K-means with L2 distance. Returns (assignments, centroids)."""
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got {features.shape}")
    rng = np.random.default_rng(seed)
    feats = np.ascontiguousarray(features, dtype=np.float32)
    centroids = _kmeanspp_init(feats, k, rng)
    assignments = _assign_euclidean(feats, centroids)
    for _ in range(iters):
        centroids = _update_centroids(feats, assignments, k)
        new_assign = _assign_euclidean(feats, centroids)
        if np.array_equal(new_assign, assignments):
            break
        assignments = new_assign
    return assignments, centroids


def kmeans_cdf_emd(
    features: npt.NDArray[np.float32],
    k: int,
    *,
    iters: int = 20,
    seed: int = 0xC0FFEE,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """K-means under 1-D EMD on histograms with *ordered* bins.

    EMD between two ordered-bin distributions equals the L1 distance between
    their cumulative distribution functions. We compute the CDF of each input
    once, then run standard L1 k-means in CDF space. Returned centroids are
    also in CDF space (monotone non-decreasing rows).
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got {features.shape}")
    feats = np.ascontiguousarray(features, dtype=np.float32)
    cdfs = np.cumsum(feats, axis=1).astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    cdf_centroids = _kmeanspp_init(cdfs, k, rng)
    assignments = _assign_l1(cdfs, cdf_centroids)
    for _ in range(iters):
        cdf_centroids = _update_centroids(cdfs, assignments, k)
        # CDFs are non-decreasing; centroid average preserves that, but enforce
        # numerically (in case of float drift after updates).
        cdf_centroids = np.maximum.accumulate(cdf_centroids, axis=1)
        new_assign = _assign_l1(cdfs, cdf_centroids)
        if np.array_equal(new_assign, assignments):
            break
        assignments = new_assign
    return assignments, cdf_centroids


__all__ = ["kmeans_cdf_emd", "kmeans_euclidean"]
