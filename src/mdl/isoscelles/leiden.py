import logging
from collections import Counter, defaultdict

import igraph as ig
import leidenalg as la
import numpy as np
from sparse import GCXS

from .gene_selection import fit_poisson
from .neighbors import calc_graph

log = logging.getLogger(__name__)

NodeID = tuple[int, ...]


def leiden_sweep(
    graph: ig.Graph,
    res_list: list[float],
    *,
    cutoff: float = None,
    cached_arrays: dict[float, np.ndarray] = None,
) -> tuple[dict[float, np.ndarray], dict[float, Counter[int, int]]]:
    membership = None
    opt = la.Optimiser()

    membership_arrays = {}
    membership_counts = {}
    if cached_arrays is None:
        cached_arrays = {}
    else:
        log.debug(f"Got {len(cached_arrays)} cached membership arrays")

    for res in res_list:
        log.info(f"Leiden clustering at resolution: {res}")
        if res in cached_arrays:
            membership = cached_arrays[res]
        else:
            # initializing the partition with the previous membership, if available
            partition = la.CPMVertexPartition(
                graph,
                initial_membership=membership,
                weights="weight",
                resolution_parameter=res,
            )
            opt.optimise_partition(partition, n_iterations=-1)
            membership = partition.membership

        membership_arrays[res] = np.array(membership)
        membership_counts[res] = Counter(membership_arrays[res])

        if membership_counts[res][1] > 0:
            c0c1_ratio = membership_counts[res][0] / membership_counts[res][1]
        else:
            continue

        if cutoff is not None and c0c1_ratio < cutoff:
            log.debug(
                f"Reached nontrivial clustering with c0/c1 ratio {c0c1_ratio:.1f},"
                " stopping"
            )
            break
    else:
        if cutoff is not None:
            log.debug(
                f"Finished resolution list without reaching c0/c1 ratio of {cutoff}"
            )

    return membership_arrays, membership_counts


def subcluster(
    data: GCXS | np.ndarray,
    res_list: list[float],
    *,
    jacc_n: int = 80,
    feature_cutoff_pct: float = 0.05,
    feature_cutoff_logp: int | float = -5,
) -> tuple[dict[float, np.ndarray], dict[float, Counter[int, int]]]:
    """
    Cluster the input data over a list of different resolutions. Note that
    this code computes an exact kNN which involves an all-by-all distance
    calculation. For larger data it is better to approximately compute kNN

    Args:
        data: an array (ndarray or sparse compressed rows) of raw count data
        res_list: list of clustering resolutions to perform
        jacc_n: number of neighbors in the kNN and SNN calculations
        feature_cutoff_pct: threshold for delta between expected and observed % nonzero
        feature_cutoff_logp threshold for log p-value on expected % nonzero

    Returns:
        membership_arrays: dictionary of resolution to cluster membership
        membership_counts: dictionary of resolution to cluster sizes
    """
    # select genes for this cell population
    is_sparse = isinstance(data, GCXS)
    exp_nz, pct, exp_p = fit_poisson(data)
    selected_feat = ((exp_nz - pct) > feature_cutoff_pct) & (
        exp_p < feature_cutoff_logp
    )
    exp = np.sqrt(data[:, selected_feat])
    if is_sparse:
        exp = exp.asformat("gcxs", compressed_axes=(0,))

    # compute shared nearest-neighbor graph
    graph = calc_graph(exp, k=jacc_n)
    if len(graph.components()) > 1:
        # SNN graph has multiple distinct components, this is likely
        # too fragmented to meaningfully cluster
        log.debug(f"Found {len(graph.components())} components")
        return {1: np.zeros(data.shape[0], dtype=int)}, {0: Counter({0: data.shape[0]})}

    # perform leiden clustering over a range of resolutions
    return leiden_sweep(graph, res_list)


def recursive_cluster(
    data: GCXS | np.ndarray,
    res_list: list[float],
    *,
    jacc_n: int = 80,
    feature_cutoff_pct: float = 0.05,
    feature_cutoff_logp: int | float = -5,
    cluster_ratio: int | float = 4,
) -> tuple[dict[NodeID, np.ndarray], dict[NodeID, float]]:
    """
    Given a complete dataset, recursively subcluster the cells until no more clusters
    can be found using the given thresholds.

    Args:
        data: a sparse array (compressed rows) of raw count data
        res_list: list of clustering resolutions to perform
        jacc_n: number of neighbors in the kNN and SNN calculations
        feature_cutoff_pct: threshold for delta between expected and observed % nonzero
        feature_cutoff_logp threshold for log p-value on expected % nonzero
        cluster_ratio: cutoff for calling a clustering as nontrivial. We select
                       the lowest resolution such that |c0| < cluster_ratio * |c1|

    Returns:
        clusters: mapping from level to cluster membership at that level (-1 if NA)
        cluster_res: mapping from level to the resolution chosen for that clustering
    """
    next_level = [()]
    clusters = defaultdict(lambda: -1 * np.ones(data.shape[0], dtype=int))
    cluster_res = {}  # record the resolution we used at each level

    # starting from the full data, go down the clustering tree, stopping when
    # there doesn't seem to be multiple clusters anymore
    while len(next_level):
        lvl = next_level.pop()
        if lvl == ():
            ci = np.ones(data.shape[0], dtype=bool)
            rl = res_list
        else:
            ci = clusters[lvl[:-1]] == lvl[-1]
            rl = [r for r in res_list if r >= cluster_res[lvl[:-1]] / 10]

        log.debug(f"Clustering {lvl} with {ci.sum()} cells")

        res_arrays, res_counts = subcluster(
            data[ci, :],
            res_list=rl,
            jacc_n=jacc_n,
            feature_cutoff_pct=feature_cutoff_pct,
            feature_cutoff_logp=feature_cutoff_logp,
        )
        # find the lowest resolution where cluster 0 doesn't dominate.
        # if there isn't one, we're done
        res = min(
            (
                r
                for r, v in res_counts.items()
                if len(v) > 1 and v[0] < cluster_ratio * v[1]
            ),
            default=1,
        )
        if res == 1:
            log.debug(f"Reached leaf clustering at {lvl}")
            continue

        # save the selected resolution and clustering
        cluster_res[lvl] = res
        clusters[lvl][ci] = res_arrays[res]

        # subcluster the results if they have enough cells
        # for the SNN network calculation
        next_level.extend(
            lvl + (i,) for i in res_counts[res] if res_counts[res][i] > jacc_n
        )

    # convert to dict so output can be pickled and to prevent extra keys
    return dict(clusters), cluster_res


def cluster_leaf_nodes(
    clusters: dict[NodeID, np.ndarray], *, n: int = 80
) -> Counter[NodeID]:
    """
    Takes a recursive clustering and applies a size cutoff to each level, pruning the
    outliers. Returns a counter of leaf nodes (terminal clusters) and their cell counts
    """
    subcluster_counts = {k: Counter(clusters[k]) for k in clusters}

    leaf_nodes = Counter(
        {
            node_id: subcluster_counts[k][i]
            for k in subcluster_counts
            for i in subcluster_counts[k]
            if i > -1
            and (node_id := (k + (i,))) not in subcluster_counts
            and subcluster_counts[k][i] > n
        }
    )

    return leaf_nodes


def cluster_labels(
    clusters: dict[NodeID, np.ndarray],
    leaf_nodes: Counter[NodeID] = None,
    *,
    n: int = 80,
) -> list[NodeID]:
    """
    Converts from the dictionary of clustering to a list of cell labels reflecting the
    hierarchical clustering, with internal unclustered cells labeled as best they can be
    """
    n_cells = clusters[()].shape[0]
    max_len = max(map(len, clusters))

    cell_labels = -1 * np.ones((n_cells, max_len + 1), dtype=np.int16)

    for node_id in sorted(clusters, key=lambda k: (len(k), k)):
        if node_id == ():
            ci = slice(None)
        else:
            ci = clusters[node_id[:-1]] == node_id[-1]
        cell_labels[ci, len(node_id)] = clusters[node_id][ci]

    a = cell_labels > -1

    assert np.all(a.all(1) | (a.sum(1) == a.argmin(1)))

    if leaf_nodes is None:
        leaf_nodes = cluster_leaf_nodes(clusters, n)

    cell_labels = [
        k[: i - (k[:i] not in leaf_nodes)]
        for k, i in zip(map(tuple, cell_labels), a.sum(1))
    ]
    return cell_labels
