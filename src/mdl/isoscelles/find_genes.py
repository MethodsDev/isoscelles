import numba as nb
import numpy as np

from .stats import mannwhitneyu


@nb.njit
def calc_nz(
    count_array: np.ndarray[int],
    nz_array: np.ndarray[float],
    group1: np.ndarray[int],
    group2: np.ndarray[int],
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """Aggregates the percent nonzero for two groups of clusters, using pre-calculated
    count and number-nonzero arrays.

    Args:
        count_array: an `n_clusters` array of the number of points in each cluster
        nz_array: an `n_clusters` x `n_features` containing the number of nonzero features
            in each cluster. Note that this should be the counts, not the fraction
        group1: index array into `n_clusters` designating clusters in group 1
        group2: index array into `n_clusters` designating clusters in group 2

    Returns:
        percent nonzero arrays for each of the groups
    """
    n_1 = count_array[group1].sum()
    nz_1 = nz_array[group1, :].sum(axis=0) / n_1

    n_2 = count_array[group2].sum()
    nz_2 = nz_array[group2, :].sum(axis=0) / n_2

    return nz_1, nz_2


@nb.njit
def calc_filter(
    nz_1: np.ndarray[float],
    nz_2: np.ndarray[float],
    *,
    delta_nz: float,
    max_nz_b: float,
) -> np.ndarray[bool]:
    """Calculate the per-feature filter for a comparison: the nonzero percentage must
    exceed `delta_nz` and the percentage for the lower of the groups must be less than
    `max_nz_b`

    Args:
        nz_1: percent nonzero for each feature, for group 1
        nz_2: percent nonzero for each feature, for group 2
        delta_nz: the difference in percent nonzero must exceed this value
        max_nz_b: the lower of the two percents must be below this value

    Returns:
        a boolean array that indicates which features to compare
    """
    nz_filter = (np.minimum(nz_1, nz_2) < max_nz_b) & (np.abs(nz_1 - nz_2) > delta_nz)

    return nz_filter


def calc_subsample(n_samples: int, subsample: int) -> np.ndarray[int]:
    """
    Provides an index for a random subsample if `n_sample` is greater than `subsample`,
    otherwise provides the full index

    Args:
        n_samples: the number of data points to be sampled
        subsample: the size of the desired subsample

    Returns:
        an array of indices into the original sample that is at most `subsample` points
    """
    if n_samples <= subsample:
        return np.arange(n_samples)
    else:
        return np.sort(np.random.choice(n_samples, size=subsample, replace=False))


def de(
    data: np.ndarray,
    clusters: np.ndarray[int],
    group1: np.ndarray[int],
    group2: np.ndarray[int],
    gene_filter: np.ndarray[bool],
    subsample: int = None,
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Compute differential expression of two groups of clusters using Mann-Whitney U-test.
    This function assumes that the data has already been clustered and we are comparing
    two groups of clusters.

    Args:
        data: a barcode-by-feature array of counts
        clusters: an array denoting cluster membership
        group1: index array into `n_clusters` designating clusters in group 1
        group2: index array into `n_clusters` designating clusters in group 2
        gene_filter: boolean array that indicates which features to compare
        subsample: the maximum number of barcodes to compare

    Returns:
        u: an array of U values for the tests. Excluded features are set to 0
        logp: log(p-value) for each feature. Excluded features are set to 0
    """
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

        u, logp = mannwhitneyu(ds_a, ds_b)

        full_u[gene_filter] = u
        full_p[gene_filter] = logp

    return full_u, full_p
