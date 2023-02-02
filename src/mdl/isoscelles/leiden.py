import logging
from collections import Counter

import igraph as ig
import leidenalg as la
import numpy as np

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
                f"Reached nontrivial clustering with c0/c1 ratio {c0c1_ratio:.1f}, stopping"
            )
            break
    else:
        if cutoff is not None:
            log.info(
                f"Finished resolution list without reaching c0/c1 ratio of {cutoff}"
            )

    return membership_arrays, membership_counts
