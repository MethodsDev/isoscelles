import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import scipy.io
import sparse

import pandas as pd
import shutil
import pysam
import os
import gzip

import scanpy as sc

def isoquant_matrix(
    isoquant_path: str | Path,
    bam_path: str | Path,
    valid_assignments=("unique", "unique_minor_difference"),
    barcode_list: Sequence[int] = None,
    feature_list: Sequence[tuple[str, str]] = None,
    out_isoform_seurat_path: str = "isoforms_seurat",
    out_h5ad_path: str = "isoforms_h5ad",
    gene_Ensembl_to_name: str = "gene_Ensembl_to_name.tsv"
):
    """
    Takes the output file from IsoQuant, along with a mapping from read name to
    barcode+umi. Returns spase matrix, together with barcode and feature lists. Output also include h5ad file which can be directly used for scanpy downstream analysis. The size of the array depends on the size of the barcode and feature lists, if they are provided.

    Args:
        isoquant_path: Path to the isoquant read_assignments.tsv file
        bam_path: Path to a bam file that contain read name, cell barcodem and UMI barcode,
            after the isoseq refine step
        valid_assignments: isoquant assignments that should be counted. See the isoquant
            documentation for more information. Default/Unique mapping only is recommended
        barcode_list: sequence of barcodes, if a specific ordering is desired. If None,
            will sort the barcodes seen
        feature_list: sequence of (isoform_id, gene_id) features, if a specific ordering
            is desired. If None, will sort the features seen
        out_isoform_seurat_path: Path to a folder containing files that can be used to construct seurat object, if the path alrady exist and non-empaty, it will be replaced
            by new outputs
        out_h5ad: Path to h5ad file
        gene_Ensembl_to_name: a tsv file with first column contaning Ensembl ID and second column containing gene name

    Opututs:
        A isoforms_seurat folder containing: the sparse count matrix, barcodes and features
            -cellranger and spaceranger standard output like format
        A isoforms_h5ad folder containing h5ad file that can be used for scanpy downstream analysis

    """
    valid_assignments = set(valid_assignments)

    samfile = pysam.AlignmentFile(bam_path, "rb", check_sq = False)
    read_to_barcode_umi = {}
    for read in samfile:
        read_to_barcode_umi.update({read.query_name:(read.get_tag("CB"),read.get_tag("XM"))})

    rname_to_tx = dict()
    tx_umi_count = defaultdict(lambda: defaultdict(set))

    with open(isoquant_path) as fh:
        fh.readline()
        fh.readline()
        for r in csv.DictReader(fh, delimiter="\t"):
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
        dtype=int).transpose()

    matrix = pd.DataFrame.sparse.from_spmatrix(matrix)
    matrix.astype(pd.SparseDtype("float"))

    if os.path.exists(out_isoform_seurat_path):
        shutil.rmtree(out_isoform_seurat_path)
    os.mkdir(out_isoform_seurat_path)

    with gzip.open(out_isoform_seurat_path + "/matrix.mtx.gz", 'wb') as f:
        scipy.io.mmwrite(f, scipy.sparse.coo_matrix(matrix))

    pd.DataFrame(barcode_list).to_csv(out_isoform_seurat_path + "/barcodes.tsv.gz", sep="\t",header=False,index=False, compression='gzip')

    Ensembl = list(pd.read_csv(gene_Ensembl_to_name, sep='\t', header = None)[0])
    name = list(pd.read_csv(gene_Ensembl_to_name, sep='\t', header = None)[1])
    feature_Ensembl = list(pd.DataFrame(feature_list)[1])
    Enzembl_to_name = {Ensembl[i]: name[i] for i in range(len(Ensembl))}
    feature_list_pd = pd.DataFrame(feature_list)
    gene_ids = feature_list_pd[1]
    feature_list_pd[1] = feature_list_pd[0] + ":" + [Enzembl_to_name[feature_Ensembl[i]] for i in range(len(feature_Ensembl))]
    feature_list_pd = feature_list_pd[[0,1]]
    feature_list_pd[2] = "Gene Expression"
    feature_list_pd[0] = gene_ids
    feature_list_pd.to_csv(out_isoform_seurat_path + "/features.tsv.gz", sep="\t",header=False, index=False, compression='gzip')

    if os.path.exists(out_h5ad_path):
        shutil.rmtree(out_h5ad_path)
    os.mkdir(out_h5ad_path)

    anndata = sc.read_10x_mtx(out_isoform_seurat_path)
    anndata.write_h5ad(out_h5ad_path + "/isoform_feature_bc_matrix.h5ad")
