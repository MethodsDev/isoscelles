import logging

import igraph as ig
import numba as nb
import numpy as np
import sparse

log = logging.getLogger(__name__)


@nb.njit(parallel=True, fastmath=True)
def kng_to_edgelist(kng: np.ndarray, knd: np.ndarray, min_weight: float = 0.0):
    """
    Convert a knn graph and distances into an array of unique edges with weights.
    Removes self-edges. Note: does *not* convert distances to similarity scores
    """
    n, k = kng.shape
    edges = np.vstack((np.repeat(np.arange(n).astype(np.int32), k), kng.flatten())).T
    weights = np.zeros(n * k, dtype=knd.dtype)

    for i in nb.prange(n):
        for jj, j in enumerate(kng[i, :]):
            if i < j:
                # this edge is fine
                weights[i * k + jj] = knd[i, jj]
            elif i > j:
                for k in kng[j, :]:
                    if i == k:
                        # this is already included on the other end
                        break
                else:
                    weights[i * k + jj] = knd[i, jj]

    ix = weights > min_weight
    return edges[ix, :], weights[ix]


@nb.njit(parallel=True, fastmath=True)
def similarity_to_edgelist(sims: np.ndarray, min_weight: float = 0.0):
    """
    Convert from a cosine similarity matrix to the cosine similarity graph
    """
    n = sims.shape[0]

    nc2 = n * (n - 1) // 2
    edges = np.empty((nc2, 2), dtype=np.int32)
    weights = np.zeros(nc2, dtype=np.float64)

    for i in nb.prange(n - 1):
        nic2 = (n - i) * (n - i - 1) // 2
        for j in range(i + 1, n):
            k = nc2 - nic2 + (j - i - 1)
            edges[k, 0] = i
            edges[k, 1] = j
            weights[k] = sims[i, j]

    ix = weights > min_weight
    return edges[ix, :], weights[ix]


@nb.njit(fastmath=True)
def cosine_similarity(u: np.ndarray, v: np.ndarray):
    """
    Compute the cosine similarity (not distance) of two vectors
    """
    m = u.shape[0]
    udotv = 0.0
    u_norm = 0.0
    v_norm = 0.0
    for i in range(m):
        udotv += u[i] * v[i]
        u_norm += u[i] * u[i]
        v_norm += v[i] * v[i]

    u_norm = np.sqrt(u_norm)
    v_norm = np.sqrt(v_norm)

    if (u_norm == 0) or (v_norm == 0):
        ratio = 1.0
    else:
        ratio = udotv / (u_norm * v_norm)
    return ratio


@nb.njit(parallel=True, fastmath=True)
def full_cosine_similarity(data: np.ndarray):
    """
    Computes all-by-all cosine similarities and returns the dense array
    """
    n = data.shape[0]
    similarity = np.eye(n, dtype=np.float64)

    # computing similarity here so higher -> better
    for i in nb.prange(n - 1):
        for j in range(i + 1, n):
            similarity[i, j] = cosine_similarity(data[i, :], data[j, :])
            similarity[j, i] = similarity[i, j]

    return similarity


@nb.njit(parallel=True, fastmath=True)
def sparse_norm(data: np.ndarray, indptr: np.ndarray):
    """
    Computes the L2 norm for each row of a sparse 2d GCXS (i.e. CSR) array
    """

    n = indptr.shape[0] - 1
    u_norms = np.zeros(n, dtype=np.float64)

    for i in nb.prange(n):
        p0 = indptr[i]
        p1 = indptr[i + 1]

        for j in range(p0, p1):
            u_norms[i] += data[j] * data[j]

        u_norms[i] = np.sqrt(u_norms[i])

    return u_norms


@nb.njit(parallel=True, fastmath=True)
def full_sparse_cosine_similarity(
    data: np.ndarray, indices: np.ndarray, indptr: np.ndarray
):
    """
    Computes all-by-all cosine similarities from a sparse GCXS array, and returns the
    dense array
    """
    u_norms = sparse_norm(data, indptr)

    n = indptr.shape[0] - 1
    similarity = np.eye(n, dtype=np.float64)

    for i in nb.prange(n - 1):
        if u_norms[i] == 0.0:
            continue

        for j in nb.prange(i + 1, n):
            if u_norms[j] == 0.0:
                continue

            p0 = indptr[i]
            p1 = indptr[i + 1]

            q0 = indptr[j]
            q1 = indptr[j + 1]

            udotv = 0.0
            for ii in range(p0, p1):
                for jj in range(q0, q1):
                    if indices[ii] == indices[jj]:
                        udotv += data[ii] * data[jj]

            similarity[i, j] = udotv / (u_norms[i] * u_norms[j])
            similarity[j, i] = similarity[i, j]

    return similarity


@nb.njit(parallel=True, fastmath=True)
def similarity_to_kng(sims: np.ndarray, k: int):
    """
    Given an array of all-by-all similarities, returns the k-nearest-neighbor array and
    the similarities for those edges.

    *Note* that this function works with _similarities_, not _distances_
    """
    n = sims.shape[0]

    kng = np.zeros((n, k), dtype=np.int32)
    kns = np.zeros((n, k), dtype=np.float64)

    for i in nb.prange(n):
        kng[i, :] = np.argsort(sims[i, :])[: -(k + 1) : -1]
        for jj, j in enumerate(kng[i, :]):
            kns[i, jj] = sims[i, j]

    return kng, kns


@nb.njit(parallel=True, fastmath=True)
def compute_mutual_edges(kng: np.ndarray, knd: np.ndarray, min_weight: float = 0.0):
    """
    Takes the knn graph and distances from pynndescent, computes unique mutual edges
    and converts from distance to edge weight (1 - distance). Removes self-edges
    """
    n, k = kng.shape
    edges = np.vstack((np.repeat(np.arange(n).astype(np.int32), k), kng.flatten())).T
    weights = np.zeros(n * k, dtype=knd.dtype)

    for i in nb.prange(n):
        for jj, j in enumerate(kng[i, :]):
            if j <= i:
                # this edge is already included, or a self-edge
                continue
            for k in kng[j, :]:
                if i == k:
                    weights[i * k + jj] = 1 - knd[i, jj]
                    break

    ix = weights > min_weight
    return edges[ix, :], weights[ix]


@nb.njit(parallel=True, fastmath=True)
def kng_to_jaccard(kng: np.ndarray, min_weight: float = 0.0):
    """
    Takes the knn graph and computes jaccard shared-nearest-neighbor edges and weights
    for all neighbors. Removes self-edges.
    """
    n, k = kng.shape
    edges = np.vstack((np.repeat(np.arange(n).astype(np.int32), k), kng.flatten())).T
    weights = np.zeros(n * k, dtype=np.float32)

    for i in nb.prange(n):
        kngs = set(kng[i, :])

        for jj, j in enumerate(kng[i, :]):
            # skip self-edges
            if i == j:
                continue

            overlap = 0
            skip = False
            for k in kng[j, :]:
                if i == k and j < i:
                    # this edge is already included
                    skip = True
                    break
                if k in kngs:
                    overlap += 1

            if not skip:
                d = overlap / (2 * k - overlap)
                weights[i * k + jj] = d

    ix = weights > min_weight
    return edges[ix, :], weights[ix]


@nb.njit(parallel=True, fastmath=True)
def kng_to_full_jaccard(kng: np.ndarray, min_weight: float = 0.0):
    """
    Takes the knn graph and computes jaccard shared-nearest-neighbor edges and weights
    for all neighbors of neighbors, which approaches the complete SNN graph. Removes
    self-edges.
    """
    n, m = kng.shape

    # bitpack edges into one int64 so we can remove duplicates
    kng2 = np.zeros(n * m * m, dtype=np.int64)

    for i in nb.prange(n):
        # because we have a self-edge, this includes the first-order edges
        js = np.unique(kng[kng[i, :], :])
        js_0 = (js[js < i] << 32) | i
        js_1 = (i << 32) | js[js > i]
        js = np.hstack((js_0, js_1))

        kng2[i * m * m : i * m * m + js.shape[0]] = js

    # remove duplicates
    kng2 = np.unique(kng2[kng2 > 0])

    # unpack back into edges
    edges = np.empty((kng2.shape[0], 2), dtype=np.int32)
    edges[:, 0] = kng2 >> 32
    edges[:, 1] = kng2 & 0xFFFFFFFF

    weights = np.empty(edges.shape[0], dtype=np.float32)

    for ii in nb.prange(edges.shape[0]):
        i = edges[ii, 0]
        j = edges[ii, 1]

        overlap = 0
        for v_i in kng[i, :]:
            for v_j in kng[j, :]:
                if v_i == v_j:
                    overlap += 1

        d = overlap / (2 * m - overlap)
        weights[ii] = d

    ix = weights > min_weight
    return edges[ix, :], weights[ix]


def cosine_edgelist(data: np.ndarray | sparse.GCXS, min_weight: float = 0.0):
    """
    Compute the all-by-all cosine similarity graph directly from data.

    This is faster than the approximate method, for smaller arrays
    """
    if isinstance(data, sparse.GCXS):
        sims = full_sparse_cosine_similarity(data)
    else:
        sims = full_cosine_similarity(data)

    return cosine_edgelist(sims, min_weight=min_weight)


def k_cosine_edgelist(data: np.ndarray | sparse.GCXS, k: int, min_weight: float = 0.0):
    """
    Creates a kNN edgelist by calculating all-by-all similarities first.
    For smaller n, this is faster than using the NNDescent algorithm,
    at the expense of temporarily higher memory usage
    """
    if isinstance(data, sparse.GCXS):
        sims = full_sparse_cosine_similarity(data.data, data.indices, data.indptr)
    else:
        sims = full_cosine_similarity(data)

    kng, kns = similarity_to_kng(sims, k)

    return kng_to_edgelist(kng, kns, min_weight)


def k_jaccard_edgelist(data: np.ndarray | sparse.GCXS, k: int, min_weight: float = 0.0):
    """
    Creates a Jaccard edgelist by calculating all-by-all similarities first.
    For smaller n, this is faster than using the NNDescent algorithm,
    at the expense of temporarily higher memory usage
    """
    if isinstance(data, sparse.GCXS):
        sims = full_sparse_cosine_similarity(data.data, data.indices, data.indptr)
    else:
        sims = full_cosine_similarity(data)

    kng, _ = similarity_to_kng(sims, k)

    return kng_to_jaccard(kng, min_weight)


def calc_graph(data: np.ndarray | sparse.GCXS, k: int = 100):
    """
    Compute the shared nearest neighbor graph from data. This computes a kNN graph
    and then the Jaccard similarity of the neighbors for each pair of
    cells.
    """
    edges, weights = k_jaccard_edgelist(data, k)
    return ig.Graph(n=data.shape[0], edges=edges, edge_attrs={"weight": weights})
