import numba as nb
import numpy as np
import scipy.stats


@nb.njit(parallel=True)
def tiecorrect(rankvals: np.ndarray) -> np.ndarray:
    """
    parallelized version of scipy.stats.tiecorrect

    Args:
        rankvals: p x n array of ranked data (output of rankdata function)

    Returns:
        array of n tie-correction factors
    """
    tc = np.ones(rankvals.shape[1], dtype=np.float64)
    for j in nb.prange(rankvals.shape[1]):
        arr = np.sort(np.ravel(rankvals[:, j]))
        idx = np.nonzero(
            np.concatenate((np.array([True]), arr[1:] != arr[:-1], np.array([True])))
        )[0]
        t_k = np.diff(idx).astype(np.float64)

        size = np.float64(arr.size)
        if size >= 2:
            tc[j] = 1.0 - (t_k**3 - t_k).sum() / (size**3 - size)

    return tc


@nb.njit(parallel=True)
def rankdata(data: np.ndarray) -> np.ndarray:
    """
    parallelized version of scipy.stats.rankdata

    Args:
        data: p x n array of data to rank, column-wise

    Returns:
        ranked: p x n array of rank values for each column of data, adjusted for ties
    """
    ranked = np.empty(data.shape, dtype=np.float64)
    for j in nb.prange(data.shape[1]):
        arr = np.ravel(data[:, j])
        sorter = np.argsort(arr)

        arr = arr[sorter]
        obs = np.concatenate((np.array([True]), arr[1:] != arr[:-1]))

        dense = np.empty(obs.size, dtype=np.int64)
        dense[sorter] = obs.cumsum()

        # cumulative counts of each unique value
        count = np.concatenate((np.nonzero(obs)[0], np.array([len(obs)])))
        ranked[:, j] = 0.5 * (count[dense] + count[dense - 1] + 1)

    return ranked


def spearmanr(x: np.ndarray) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Version of Spearman correlation that runs in parallel on a 2d array. Based
    on scipy.stats.spearmanr, with some simplifications

    This computes all-by-all correlation and returns the arrays of r values
    and log p-values, using a two-sided test.

    Args:
        x: 2-d array array of n_observations x n_variables

    Returns:
        r: array of size n_variables x n_variables containing the r coefficient for
           each pairwise comparison
        logp: array of size n_variables x n_variables containing the log(p-value) for
              each pairwise comparison
    """
    n_obs, _ = x.shape
    a_ranked = rankdata(x)  # this is parallelized

    r = np.corrcoef(a_ranked, rowvar=False)
    dof = n_obs - 2  # degrees of freedom

    # r can have elements equal to 1, so avoid zero division warnings
    with np.errstate(divide="ignore"):
        # clip the small negative values possibly caused by rounding
        # errors before taking the square root
        t = r * np.sqrt((dof / ((r + 1.0) * (1.0 - r))).clip(0))

    logp = np.minimum(scipy.stats.t.logcdf(-np.abs(t), dof) + np.log(2), 0)

    return r, logp


def mannwhitneyu(x: np.ndarray, y: np.ndarray, use_continuity: bool = True):
    """
    Version of Mann-Whitney U-test that runs in parallel on 2d arrays

    This is the two-sided test, asymptotic algo only. Returns log p-values

    Args:
        x, y: 2-d arrays of samples. Arrays must have the same number of columns, which
              are the features compared by the test.
        use_continuity: Whether a continuity correction (1/2) should be applied.

    Returns:
        u: an array of U values for the tests. Excluded features have value 0
        logp: log(p-value) for each feature. Excluded features are set to 0
    """
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape[1] == y.shape[1]

    n1 = x.shape[0]
    n2 = y.shape[0]

    ranked = rankdata(np.concatenate((x, y)))
    rankx = ranked[:n1, :]  # get the x-ranks
    u1 = n1 * n2 + (n1 * (n1 + 1)) / 2.0 - np.sum(rankx, axis=0)  # calc U for x
    u2 = n1 * n2 - u1  # remainder is U for y
    T = tiecorrect(ranked)

    # if *everything* is identical we'll raise an error, not otherwise
    if np.all(T == 0):
        raise ValueError("All numbers are identical in mannwhitneyu")
    sd = np.sqrt(T * n1 * n2 * (n1 + n2 + 1) / 12.0)

    meanrank = n1 * n2 / 2.0 + 0.5 * use_continuity
    bigu = np.maximum(u1, u2)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = (bigu - meanrank) / sd

    logp = np.minimum(scipy.stats.norm.logsf(z) + np.log(2), 0)

    return u2, logp
