"""End-to-end test mimicking real user exploratory data analysis workflow."""

import collections
import logging
import time
from typing import Any, TypedDict

import starlings as sl

logger = logging.getLogger(__name__)


class EntityDict(TypedDict):
    """Entity cluster from pre-merged sources."""

    id: int
    keys: dict[str, list[int]]


def generate_link_graph(
    n_left: int = 2500,
    n_right: int = 2500,
    n_isolates: int = 0,
    thresholds: dict[float, int] | None = None,
) -> tuple[dict[str, list], list[EntityDict]]:
    """Generate a bipartite graph with exact component counts.

    Uses hierarchical block construction method to ensure exact component counts.
    """
    if thresholds is None:
        thresholds = {}

    edges: dict[str, list[Any]] = {"left": [], "right": [], "prob": []}
    total_nodes = n_left + n_right

    # Isolate nodes are taken from the end, we work with the 'active' ones.
    active_n_left = n_left
    active_n_right = n_right - n_isolates

    if thresholds:
        # Sort thresholds by component count, ASCENDING (coarsest to finest)
        hierarchy = sorted([
            (comp_count, prob) for prob, comp_count in thresholds.items()
        ])

        # 1. PARTITION NODES INTO BLOCKS FOR EACH HIERARCHICAL LEVEL
        blocks_by_level: dict[int, dict[int, list[int]]] = {}
        for n_components, _prob in hierarchy:
            blocks = collections.defaultdict(list)
            # Partition left and right nodes separately to ensure connect-ability
            for i in range(active_n_left):
                blocks[i % n_components].append(i)
            for i in range(active_n_right):
                # Global index for a right node with local index `i` is `n_left + i`
                blocks[i % n_components].append(n_left + i)
            blocks_by_level[n_components] = blocks

        # 2. GENERATE EDGES
        def create_intra_block_edges(node_list: list[int], prob: float):
            if len(node_list) <= 1:
                return
            lefts = sorted([n for n in node_list if n < n_left])
            rights = sorted([n for n in node_list if n >= n_left])
            if not lefts or not rights:
                return

            root_node = lefts[0]
            for r_node in rights:
                edges["left"].append(root_node)
                edges["right"].append(r_node)
                edges["prob"].append(prob)

            first_right_node = rights[0]
            for l_node in lefts[1:]:
                edges["left"].append(l_node)
                edges["right"].append(first_right_node)
                edges["prob"].append(prob)

        def create_inter_block_edge(
            block_a: list[int], block_b: list[int], prob: float
        ):
            l_nodes_a = [n for n in block_a if n < n_left]
            r_nodes_b = [n for n in block_b if n >= n_left]
            if l_nodes_a and r_nodes_b:
                edges["left"].append(l_nodes_a[0])
                edges["right"].append(r_nodes_b[0])
                edges["prob"].append(prob)
                return

            r_nodes_a = [n for n in block_a if n >= n_left]
            l_nodes_b = [n for n in block_b if n < n_left]
            if r_nodes_a and l_nodes_b:
                edges["left"].append(l_nodes_b[0])
                edges["right"].append(r_nodes_a[0])
                edges["prob"].append(prob)

        # Step 2a: Create intra-block edges for the finest partition
        finest_n_components, finest_prob = hierarchy[-1]
        finest_blocks = blocks_by_level[finest_n_components]
        for block_id in range(finest_n_components):
            create_intra_block_edges(finest_blocks[block_id], finest_prob)

        # Step 2b: Create inter-block linking edges for all coarser partitions
        for i in range(len(hierarchy) - 1, 0, -1):
            n_comp_finer, _prob_finer = hierarchy[i]
            n_comp_coarser, prob_coarser = hierarchy[i - 1]
            blocks_finer = blocks_by_level[n_comp_finer]

            for j in range(n_comp_coarser, n_comp_finer):
                target_coarse_block_id = j % n_comp_coarser
                block_curr = blocks_finer[j]
                block_target = blocks_finer[target_coarse_block_id]
                create_inter_block_edge(block_curr, block_target, prob_coarser)

    # 3. FINALISATION: Generate the raw, un-merged entity list
    entity_list: list[EntityDict] = []
    n_left_half, n_right_half = n_left // 2, n_right // 2
    for i in range(total_nodes):
        keys: dict[str, list[int]] = {
            "source_1": [],
            "source_2": [],
            "source_3": [],
            "source_4": [],
        }
        if i < n_left:  # Node is on the left side
            if i < n_left_half:
                keys["source_1"].append(i)
            else:
                keys["source_2"].append(i - n_left_half)
        else:  # Node is on the right side
            r_idx = i - n_left
            if r_idx < n_right_half:
                keys["source_3"].append(r_idx)
            else:
                keys["source_4"].append(r_idx - n_right_half)

        entity_list.append({"id": i, "keys": keys})

    return edges, entity_list


def test_user_eda_workflow():
    """Simple EDA workflow: load 10k edges and explore thresholds."""
    # Generate ~10k edges with realistic structure
    thresholds = {
        0.9: 2_000,
        0.7: 1_000,
        0.5: 500,
    }

    # Time graph generation
    start_time = time.monotonic()
    edges, entities = generate_link_graph(
        n_left=5_500,
        n_right=5_500,
        n_isolates=0,  # Current MVP doesn't support isolates
        thresholds=thresholds,
    )
    graph_time = time.monotonic() - start_time

    # Time collection creation (known bottleneck)
    start_time = time.monotonic()
    collection = sl.Collection.from_edges(
        list(zip(edges["left"], edges["right"], edges["prob"], strict=True))
    )
    collection_time = time.monotonic() - start_time

    # Test at exact thresholds where edges were created
    prev_components: int = 0
    for threshold, components in thresholds.items():
        partition = collection.at(threshold)
        assert partition.num_entities == components >= prev_components

    # Test precision: at threshold 1.0 should have all singletons
    singleton_partition = collection.at(1.0)
    assert singleton_partition.num_entities == len(entities)

    # Quick EDA sweep
    thresholds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    counts = [collection.at(t).num_entities for t in thresholds]

    edge_count = sum(len(edge_list) for edge_list in edges.values()) // 3
    logger.info(
        "EDA workflow: %d edges, %d entities. Graph: %.2fs, Collection: %.2fs. %s",
        edge_count,
        len(entities),
        graph_time,
        collection_time,
        dict(zip(thresholds, counts, strict=False)),
    )
