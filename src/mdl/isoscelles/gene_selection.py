import logging

import numpy as np
import scipy.stats
import sparse

log = logging.getLogger(__name__)


# blockwise poisson fit of gene counts
def fit_poisson(
    counts: np.ndarray | sparse.GCXS, numis: np.ndarray = None, blocksize: int = 128_000
):
    is_sparse = isinstance(counts, sparse.GCXS)

    def if_sparse(X):
        if is_sparse:
            return X.todense()
        return X

    n_cells = counts.shape[0]

    # pre-compute these values
    log.debug("computing percent nonzero per gene")
    pct = if_sparse(np.sign(counts).sum(0))
    pct = pct / n_cells

    log.debug("computing average expression per gene")
    exp = if_sparse(counts.sum(0, keepdims=True))  # 1 x n_genes
    exp = exp / exp.sum()

    if numis is None:
        numis = if_sparse(counts.sum(1, keepdims=True))

    exp_nz = np.zeros(exp.shape)  # 1 x n_genes
    var_nz = np.zeros(exp.shape)  # 1 x n_genes

    log.debug("computing expected percent nonzero")
    # run in chunks (still large, but seems easier for dask to handle)
    for i in range(0, n_cells, blocksize):
        if i % (blocksize * 10) == 0:
            log.debug(f"{i} ...")

        numis_t = numis[i : i + blocksize, :].T
        prob_zero = np.exp(-exp.T.dot(numis_t))  # n_genes x b

        exp_nz_b = (1 - prob_zero).sum(1)  # n_genes
        var_nz_b = (prob_zero * (1 - prob_zero)).sum(1)

        exp_nz += exp_nz_b
        var_nz += var_nz_b

    exp_nz = exp_nz.squeeze() / n_cells
    std_nz = np.sqrt(var_nz.squeeze()) / n_cells

    log.debug("... done")

    exp_p = np.zeros_like(pct)
    ix = (std_nz != 0).flatten()
    exp_p[ix] = scipy.stats.norm.logcdf(pct[ix], loc=exp_nz[ix], scale=std_nz[ix])

    return exp_nz, pct, exp_p
