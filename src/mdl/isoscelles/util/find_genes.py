from typing import Union

import dask.array as da
import numba as nb
import numpy as np

from .stats import mannwhitneyu

ArrayLike = Union[np.ndarray, da.Array]


@nb.njit
def calc_nz(
    count_array: np.ndarray,
    nz_array: np.ndarray,
    group1: np.ndarray,
    group2: np.ndarray,
):
    n_1 = count_array[group1].sum()
    nz_1 = nz_array[group1, :].sum(axis=0) / n_1

    n_2 = count_array[group2].sum()
    nz_2 = nz_array[group2, :].sum(axis=0) / n_2

    return nz_1, nz_2


@nb.njit
def calc_filter(
    nz_1: np.ndarray, nz_2: np.ndarray, *, delta_nz: float, max_nz_b: float
):
    nz_filter = (np.minimum(nz_1, nz_2) < max_nz_b) & (np.abs(nz_1 - nz_2) > delta_nz)

    return nz_filter


def calc_subsample(n_samples: int, subsample: int):
    if n_samples <= subsample:
        return np.arange(n_samples)
    else:
        return np.sort(np.random.choice(n_samples, size=subsample, replace=False))


def de(
    data: ArrayLike,
    clusters: np.ndarray,
    group1: np.ndarray,
    group2: np.ndarray,
    gene_filter: np.ndarray,
    subsample: int = None,
):
    c_a = np.isin(clusters, group1)
    c_b = np.isin(clusters, group2)

    full_u = np.zeros(data.shape[1])
    full_p = np.zeros(data.shape[1])  # logp, no result = 0

    if np.any(gene_filter):
        ds_a = data[c_a, :][:, gene_filter]
        ds_b = data[c_b, :][:, gene_filter]
        if subsample is not None:
            ds_a = ds_a[calc_subsample(ds_a.shape[0], subsample), :]
            ds_b = ds_b[calc_subsample(ds_b.shape[0], subsample), :]

        if isinstance(data, da.Array):
            ds_a, ds_b = da.compute(ds_a, ds_b)

        u, logp = mannwhitneyu(ds_a, ds_b)

        full_u[gene_filter] = u
        full_p[gene_filter] = logp

    return full_u, full_p
