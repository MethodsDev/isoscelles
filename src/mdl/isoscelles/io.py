from pathlib import Path

import h5py
import numpy as np
import scipy.io
import sparse


def read_10x_h5(path: str):
    """
    Read a 10x cellranger h5 file and return the data as a sparse GCXS array,
    along with tuples containing the barcodes and genes
    """
    with h5py.File(path, "r") as fh:
        M, N = fh["matrix"]["shape"]
        gene_names = tuple(fh["matrix"]["features"]["name"].asstr())
        gene_ids = tuple(fh["matrix"]["features"]["id"].asstr())

        genes = tuple(zip(gene_names, gene_ids))
        barcodes = tuple(fh["matrix"]["barcodes"].asstr())

        data = np.asarray(fh["matrix"]["data"])
        indices = np.asarray(fh["matrix"]["indices"])
        indptr = np.asarray(fh["matrix"]["indptr"])

    matrix = sparse.GCXS((data, indices, indptr), shape=(N, M), compressed_axes=(0,))

    return matrix, barcodes, genes


def read_mtx(path: str | Path):
    """
    Read an mtx file and return a sparse GCXS array. Transposes the input,
    because files are usually gene x cell and we want cell x gene
    """
    with open(path, "rb") as fh:
        m = scipy.io.mmread(fh).astype(np.int32)

    return sparse.GCXS(m.T, compressed_axes=(0,))
