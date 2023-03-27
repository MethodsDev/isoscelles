import logging
from collections import Counter, defaultdict

import igraph as ig
import leidenalg as la
import numpy as np

from .gene_selection import fit_poission
from .neighbors import calc_graph


log = logging.getLogger(__name__)


def leiden_sweep(
    graph: ig.Graph,
    res_list: list[float],
    cutoff: float = None,
    cached_arrays: dict[float, np.ndarray] = None,
):
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
            log.info(
                f"Reached nontrivial clustering with c0/c1 ratio {c0c1_ratio:.1f},"
                " stopping"
            )
            break
    else:
        if cutoff is not None:
            log.info(
                f"Finished resolution list without reaching c0/c1 ratio of {cutoff}"
            )

    return membership_arrays, membership_counts


def subcluster(m, res_list, *, jacc_n=100, pct_cutoff=0.05, logp_cutoff=-5):
    # select genes for this cell population
    exp_nz, pct, exp_p = fit_poission(m, sparse=True)
    sel_g = ((exp_nz - pct) > pct_cutoff) & (exp_p < logp_cutoff)
    exp = np.sqrt(m[:, sel_g]).todense()
    # compute shared nearest-neighbor graph
    graph = calc_graph(exp, n=jacc_n)
    if len(graph.components()) > 1:
        # SNN graph has multiple distinct components, this is likely
        # too fragmented to meaningfully cluster
        return {1.0: np.ones(m.shape[0], dtype=int)}, {0: m.shape[0]}

    # perform leiden clustering over a range of resolutions
    return leiden_sweep(graph, res_list)


def recursive_cluster(
    m, res_list, *, jacc_n=100, pct_cutoff=0.05, logp_cutoff=-5, cluster_ratio=4
):
    next_level = [()]
    clusters = defaultdict(lambda: -1 * np.ones(m.shape[0], dtype=int))
    cluster_res = {}  # record the resolution we used at each level

    # starting from the full data, go down the clustering tree, stopping when
    # there doesn't seem to be multiple clusters anymore
    while len(next_level):
        lvl = next_level.pop()
        if lvl == ():
            ci = np.ones(m.shape[0], dtype=bool)
            rl = res_list
        else:
            ci = clusters[lvl[:-1]] == lvl[-1]
            rl = [r for r in res_list if r >= cluster_res[lvl[:-1]] / 10]

        print(lvl, ci.sum())

        res_arrays, res_counts = subcluster(
            m[ci, :],
            res_list=rl,
            jacc_n=jacc_n,
            pct_cutoff=pct_cutoff,
            logp_cutoff=logp_cutoff,
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
            print(f"Reached leaf clustering at {lvl}")
            continue

        cluster_res[lvl] = res
        clusters[lvl][ci] = res_arrays[res]
        # subcluster the results if they have enough cells
        # for the SNN network calculation
        next_level.extend(
            lvl + (i,) for i in res_counts[res] if res_counts[res][i] > jacc_n
        )

    return dict(clusters), cluster_res
