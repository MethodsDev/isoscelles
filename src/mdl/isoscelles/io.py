import csv
import gzip
import logging
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Sequence, TextIO

import h5py
import numpy as np
import scipy.io
import sparse

log = logging.getLogger(__name__)

BarcodeFeatureTuple = tuple[sparse.GCXS, Sequence[str], Sequence[tuple[str, str]]]


def _optional_gzip(path: str | Path, mode: str = "rt") -> TextIO:
    if Path(path).suffix == ".gz":
        return gzip.open(path, mode)
    else:
        return open(path, mode)


def read_10x_h5(path: str | Path) -> BarcodeFeatureTuple:
    """
    Read a 10x cellranger h5 file and return the data as a sparse GCXS array,
    along with tuples containing the barcodes and genes

    Args:
        path: the path to 10x-formatted hdf5 file

    Returns:
        The sparse count array, along with the barcodes and features

    """
    with h5py.File(path, "r") as fh:
        M, N = fh["matrix"]["shape"]
        gene_names = tuple(fh["matrix"]["features"]["name"].asstr())
        gene_ids = tuple(fh["matrix"]["features"]["id"].asstr())

        features = tuple(zip(gene_names, gene_ids))
        barcodes = tuple(fh["matrix"]["barcodes"].asstr())

        data = np.asarray(fh["matrix"]["data"])
        indices = np.asarray(fh["matrix"]["indices"])
        indptr = np.asarray(fh["matrix"]["indptr"])

    matrix = sparse.GCXS((data, indices, indptr), shape=(N, M), compressed_axes=(0,))

    return matrix, barcodes, features


def read_mtx(path: str | Path, transpose: bool = True) -> sparse.GCXS:
    """
    Read an mtx file and return a sparse GCXS array. Transposes the input by default,
    because these files are usually gene x cell and we want cell x gene

    Args:
        path: path to an mtx-formatted count file, optionally gzipped
        transpose: whether to transpose the input matrix

    Returns:
        The sparse count array
    """
    with _optional_gzip(path, "rb") as fh:
        m = scipy.io.mmread(fh).astype(np.int32)

    if transpose:
        return sparse.GCXS(m.T, compressed_axes=(0,))
    else:
        return sparse.GCXS(m, compressed_axes=(0,))


def read_mtx_dir(
    path: str | Path,
    matrix_file: str = "matrix.mtx",
    barcode_file: str = "barcodes.tsv",
    feature_file: str = "genes.tsv",
):
    """
    Reads the contents of an mtx directory and returns as sparse GCXS array, a list of
    barcodes, and a list of features
    """
    path = Path(path)

    with _optional_gzip(path / barcode_file) as fh:
        barcodes = [line.strip() for line in fh]

    with _optional_gzip(path / feature_file) as fh:
        features = [row[0] for row in csv.reader(fh, delimiter="\t")]

    matrix = read_mtx(path / matrix_file)

    return matrix, barcodes, features


def isoquant_matrix(
    isoquant_path: str | Path,
    read_to_barcode_umi: dict[str, tuple[str, str]],
    *,
    valid_assignments=("unique", "unique_minor_difference"),
    barcode_list: Sequence[str] = None,
    feature_list: Sequence[tuple[str, str]] = None,
) -> BarcodeFeatureTuple:
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


def to_anndata(
    matrix: sparse.GCXS,
    barcode_list: Sequence[str],
    feature_list: Sequence[tuple[str, str]],
):
    """
    Create an AnnData object out of a sparse array, barcode list and feature list. This
    function assumes the feature list consists of tuples of (isoform_id, gene_id).
    """
    try:
        import anndata as ad
    except ImportError:
        log.error("Could not import anndata, please make sure it is installed")
        return None

    try:
        import pandas as pd
    except ImportError:
        log.error("Could not import pandas, required to create h5ad")
        return None

    isoform_list, gene_list, *_ = zip(*feature_list)

    return ad.AnnData(
        matrix.to_scipy_sparse(),
        obs=pd.DataFrame(index=barcode_list),
        var=pd.DataFrame(index=isoform_list, data={"gene": gene_list}),
    )


def write_mtx(
    matrix: sparse.GCXS,
    barcode_list: Sequence[str],
    feature_list: Sequence[tuple[str, str]],
    output_path: str | Path,
):
    """
    Write a matrix, barcode list and feature list to disk in the standard mtx format
    """
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir()
    elif not output_path.isdir():
        raise ValueError(f"{output_path} must be a directory")

    matrix = matrix.to_scipy_sparse()
    with gzip.open(output_path / "matrix.mtx.gz", "wb") as out:
        scipy.io.mmwrite(out, matrix)

    with gzip.open(output_path / "barcodes.tsv.gz", "wt") as out:
        print("\n".join(barcode_list), file=out)

    with gzip.open(output_path / "features.tsv.gz", "wt") as out:
        print("\n".join("\t".join(ft) for ft in feature_list), file=out)
