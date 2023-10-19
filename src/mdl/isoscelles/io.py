import csv
import gzip
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import scipy.io
import sparse


def _optional_gzip(path: str | Path, mode: str = "rt"):
    if Path(path).suffix == ".gz":
        return gzip.open(path, mode)
    else:
        return open(path, mode)


def read_10x_h5(path: str | Path):
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
    with _optional_gzip(path, "rb") as fh:
        m = scipy.io.mmread(fh).astype(np.int32)

    return sparse.GCXS(m.T, compressed_axes=(0,))


def isoquant_matrix(
    isoquant_path: str | Path,
    read_to_barcode_umi: dict[str, tuple[str, str]],
    *,
    valid_assignments=("unique", "unique_minor_difference"),
    barcode_list: Sequence[str] = None,
    feature_list: Sequence[tuple[str, str]] = None,
):
    """
    Takes the output file from IsoQuant, along with a mapping from read name to
    barcode+umi. Returns a sparse array of UMI counts in GCXS format. The size of the
    array depends on the size of the barcode and feature lists, if they are provided

    Args:
        isoquant_path: Path to the isoquant read_assignments.tsv[.gz] file
        read_to_barcode_umi: a mapping from read name to (barcode, UMI). This can be
            created by extracting the relevant part of the reads. Only the reads in this
            mapping (and thus the barcodes) will be included in the output array
        valid_assignments: isoquant assignments that should be counted. See the isoquant
            documentation for more information
        barcode_list: sequence of barcodes, if a specific ordering is desired. If None,
            will sort the barcodes seen
        feature_list: sequence of (isoform_id, gene_id) features, if a specific ordering
            is desired. If None, will sort the features seen

    Returns:
        The sparse count array, along with the barcodes and features in the same order
    """
    valid_assignments = set(valid_assignments)

    rname_to_tx = dict()
    tx_umi_count = defaultdict(lambda: defaultdict(set))

    with _optional_gzip(isoquant_path) as fh:
        for r in csv.DictReader(islice(fh, 2, None), delimiter="\t"):
            if r["#read_id"] in read_to_barcode_umi:
                if r["assignment_type"] in valid_assignments:
                    rname_to_tx[r["#read_id"]] = (r["isoform_id"], r["gene_id"])
                    bc, umi = read_to_barcode_umi[r["#read_id"]]
                    tx_umi_count[bc][(r["isoform_id"], r["gene_id"])].add(umi)

    if barcode_list is None:
        barcode_list = sorted(tx_umi_count)

    barcode_index = {bc: i for i, bc in enumerate(barcode_list)}

    if feature_list is None:
        feature_list = sorted(set(rname_to_tx.values()))

    feature_index = {tx: i for i, tx in enumerate(feature_list)}

    matrix = sparse.COO.from_iter(
        (
            ((barcode_index[bc], feature_index[tx]), len(tx_umi_count[bc][tx]))
            for bc in tx_umi_count
            for tx in tx_umi_count[bc]
        ),
        shape=(len(barcode_list), len(feature_list)),
        fill_value=0,
        dtype=int,
    ).asformat("gcxs", compressed_axes=(0,))

    return matrix, barcode_list, feature_list
