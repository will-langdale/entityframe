"""End-to-end test mimicking real user exploratory data analysis workflow."""

import logging
import time

import starlings as sl

logger = logging.getLogger(__name__)


def test_user_eda_workflow():
    """Production-scale EDA workflow: Million-record minimum scale for library."""
    # Million-scale is the MINIMUM expected dataset size for production use

    # Time graph generation using Rust implementation
    start_time = time.monotonic()
    edges, total_nodes = sl.generate_production_1m_graph()
    graph_time = time.monotonic() - start_time

    # Time collection creation (optimised Python->Rust boundary)
    start_time = time.monotonic()
    collection = sl.Collection.from_edges(edges)
    collection_time = time.monotonic() - start_time

    # Test hierarchical behavior: lower thresholds should have fewer or equal entities
    # (because more records get merged at lower thresholds)
    test_thresholds = [0.9, 0.7, 0.5]
    prev_entities: float = float("inf")

    for threshold in sorted(test_thresholds, reverse=True):
        partition = collection.at(threshold)
        current_entities = partition.num_entities

        # Lower thresholds should have <= entities than higher thresholds
        assert current_entities <= prev_entities, (
            f"Hierarchy violation: {threshold} has {current_entities} entities, "
            f"previous higher threshold had {prev_entities}"
        )

        # Should have meaningful reduction (not all singletons, not single component)
        assert 1 < current_entities < total_nodes, (
            f"At {threshold}: {current_entities} entities should be "
            f"between 1 and {total_nodes}"
        )

        prev_entities = current_entities

    # Test precision: at threshold 1.0 should have all singletons
    singleton_partition = collection.at(1.0)
    assert singleton_partition.num_entities == total_nodes

    # Quick EDA sweep
    test_thresholds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    counts = [collection.at(t).num_entities for t in test_thresholds]

    logger.info(
        "EDA workflow: %d edges, %d total nodes. Graph: %.2fs, Collection: %.2fs. %s",
        len(edges),
        total_nodes,
        graph_time,
        collection_time,
        dict(zip(test_thresholds, counts, strict=False)),
    )
