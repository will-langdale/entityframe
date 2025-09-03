//! Test utilities for generating realistic entity resolution graph patterns.
//!
//! This module provides graph generation functions that mirror production entity resolution
//! patterns, including hierarchical threshold structures and realistic component distributions.
//! These patterns are used consistently across both Rust benchmarks and Python tests.

use crate::core::{DataContext, Key};
use std::collections::HashMap;

/// Entity cluster configuration for hierarchical graph generation.
#[derive(Debug, Clone)]
pub struct ThresholdConfig {
    /// Similarity threshold (0.0 to 1.0)
    pub threshold: f64,
    /// Target number of entities at this threshold
    pub target_entities: usize,
}

/// Configuration for generating realistic entity resolution graphs.
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Number of left-side records (e.g., customers)
    pub n_left: usize,
    /// Number of right-side records (e.g., transactions)  
    pub n_right: usize,
    /// Number of isolated records (no edges)
    pub n_isolates: usize,
    /// Hierarchical threshold configuration (threshold -> target entities)
    pub thresholds: Vec<ThresholdConfig>,
}

impl GraphConfig {
    /// Create a production-scale configuration for million-record testing.
    pub fn production_1m() -> Self {
        Self {
            n_left: 550_000,
            n_right: 550_000,
            n_isolates: 0,
            thresholds: vec![
                ThresholdConfig {
                    threshold: 0.9,
                    target_entities: 200_000,
                },
                ThresholdConfig {
                    threshold: 0.7,
                    target_entities: 100_000,
                },
                ThresholdConfig {
                    threshold: 0.5,
                    target_entities: 50_000,
                },
            ],
        }
    }

    /// Create a large-scale configuration for 10M+ record testing.
    pub fn production_10m() -> Self {
        Self {
            n_left: 5_500_000,
            n_right: 5_500_000,
            n_isolates: 0,
            thresholds: vec![
                ThresholdConfig {
                    threshold: 0.9,
                    target_entities: 2_000_000,
                },
                ThresholdConfig {
                    threshold: 0.7,
                    target_entities: 1_000_000,
                },
                ThresholdConfig {
                    threshold: 0.5,
                    target_entities: 500_000,
                },
            ],
        }
    }
}

/// Generated graph data with edges and entity records.
pub struct GraphData {
    /// Edge list: (left_id, right_id, similarity_weight)
    pub edges: Vec<(u32, u32, f64)>,
    /// Data context with all records
    pub context: DataContext,
    /// Total number of nodes in the graph
    pub total_nodes: usize,
}

/// Generate a hierarchical bipartite graph with exact component counts.
///
/// This function mirrors the Python `generate_link_graph` function, creating
/// realistic entity resolution patterns with hierarchical threshold structures.
/// Uses the same block-based construction method to ensure exact component counts.
///
/// # Arguments
/// * `config` - Graph generation configuration
///
/// # Returns
/// * `GraphData` - Generated edges, context, and metadata
///
/// # Algorithm
/// 1. Partition nodes into hierarchical blocks for each threshold level
/// 2. Create intra-block edges at the finest threshold level
/// 3. Create inter-block linking edges for coarser levels
/// 4. Generate realistic record keys across multiple source types
pub fn generate_hierarchical_graph(config: GraphConfig) -> GraphData {
    let total_nodes = config.n_left + config.n_right;
    let active_n_left = config.n_left;
    let active_n_right = config.n_right - config.n_isolates;

    let mut edges = Vec::new();
    let context = DataContext::new();

    // Generate all record keys first with mixed types for realism
    let n_left_half = config.n_left / 2;
    let n_right_half = config.n_right / 2;

    for i in 0..total_nodes {
        let key = if i < config.n_left {
            // Left-side records (customers, entities)
            if i < n_left_half {
                Key::String(format!("cust_{}", i))
            } else {
                Key::String(format!("entity_{}", i - n_left_half))
            }
        } else {
            // Right-side records (transactions, addresses)
            let r_idx = i - config.n_left;
            if r_idx < n_right_half {
                Key::U64(1000000 + r_idx as u64)
            } else {
                Key::Bytes(format!("addr_{}", r_idx - n_right_half).into_bytes())
            }
        };

        let source_name = match i % 4 {
            0 => "source_1",
            1 => "source_2",
            2 => "source_3",
            _ => "source_4",
        };

        context.ensure_record(source_name, key);
    }

    if !config.thresholds.is_empty() {
        // Sort thresholds by target entities, ASCENDING (coarsest to finest)
        let mut hierarchy: Vec<_> = config
            .thresholds
            .iter()
            .map(|tc| (tc.target_entities, tc.threshold))
            .collect();
        hierarchy.sort_by_key(|&(entities, _)| entities);

        // 1. PARTITION NODES INTO BLOCKS FOR EACH HIERARCHICAL LEVEL
        let mut blocks_by_level: HashMap<usize, HashMap<usize, Vec<usize>>> = HashMap::new();

        for &(n_components, _threshold) in &hierarchy {
            let mut blocks: HashMap<usize, Vec<usize>> = HashMap::new();

            // Partition left and right nodes separately to ensure connectivity
            for i in 0..active_n_left {
                blocks.entry(i % n_components).or_default().push(i);
            }
            for i in 0..active_n_right {
                // Global index for right node with local index `i` is `n_left + i`
                blocks
                    .entry(i % n_components)
                    .or_default()
                    .push(config.n_left + i);
            }
            blocks_by_level.insert(n_components, blocks);
        }

        // 2. GENERATE EDGES

        // Helper: Create intra-block edges (within same component)
        let create_intra_block_edges =
            |node_list: &[usize], prob: f64, edges: &mut Vec<(u32, u32, f64)>| {
                if node_list.len() <= 1 {
                    return;
                }

                let mut lefts: Vec<_> = node_list
                    .iter()
                    .filter(|&&n| n < config.n_left)
                    .copied()
                    .collect();
                let mut rights: Vec<_> = node_list
                    .iter()
                    .filter(|&&n| n >= config.n_left)
                    .copied()
                    .collect();
                lefts.sort_unstable();
                rights.sort_unstable();

                if lefts.is_empty() || rights.is_empty() {
                    return;
                }

                // Star pattern: connect first left to all rights
                let root_node = lefts[0];
                for &r_node in &rights {
                    edges.push((root_node as u32, r_node as u32, prob));
                }

                // Connect remaining lefts to first right
                let first_right_node = rights[0];
                for &l_node in &lefts[1..] {
                    edges.push((l_node as u32, first_right_node as u32, prob));
                }
            };

        // Helper: Create inter-block edge (between components)
        let create_inter_block_edge =
            |block_a: &[usize], block_b: &[usize], prob: f64, edges: &mut Vec<(u32, u32, f64)>| {
                let l_nodes_a: Vec<_> = block_a
                    .iter()
                    .filter(|&&n| n < config.n_left)
                    .copied()
                    .collect();
                let r_nodes_b: Vec<_> = block_b
                    .iter()
                    .filter(|&&n| n >= config.n_left)
                    .copied()
                    .collect();

                if !l_nodes_a.is_empty() && !r_nodes_b.is_empty() {
                    edges.push((l_nodes_a[0] as u32, r_nodes_b[0] as u32, prob));
                    return;
                }

                let r_nodes_a: Vec<_> = block_a
                    .iter()
                    .filter(|&&n| n >= config.n_left)
                    .copied()
                    .collect();
                let l_nodes_b: Vec<_> = block_b
                    .iter()
                    .filter(|&&n| n < config.n_left)
                    .copied()
                    .collect();

                if !r_nodes_a.is_empty() && !l_nodes_b.is_empty() {
                    edges.push((l_nodes_b[0] as u32, r_nodes_a[0] as u32, prob));
                }
            };

        // Step 2a: Create intra-block edges for the finest partition
        let (finest_n_components, finest_prob) = hierarchy[hierarchy.len() - 1];
        if let Some(finest_blocks) = blocks_by_level.get(&finest_n_components) {
            for block_id in 0..finest_n_components {
                if let Some(block_nodes) = finest_blocks.get(&block_id) {
                    create_intra_block_edges(block_nodes, finest_prob, &mut edges);
                }
            }
        }

        // Step 2b: Create inter-block linking edges for all coarser partitions
        for i in (1..hierarchy.len()).rev() {
            let (n_comp_finer, _) = hierarchy[i];
            let (n_comp_coarser, prob_coarser) = hierarchy[i - 1];

            if let Some(blocks_finer) = blocks_by_level.get(&n_comp_finer) {
                for j in n_comp_coarser..n_comp_finer {
                    let target_coarse_block_id = j % n_comp_coarser;
                    if let (Some(block_curr), Some(block_target)) = (
                        blocks_finer.get(&j),
                        blocks_finer.get(&target_coarse_block_id),
                    ) {
                        create_inter_block_edge(block_curr, block_target, prob_coarser, &mut edges);
                    }
                }
            }
        }
    }

    GraphData {
        edges,
        context,
        total_nodes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_graph_generation() {
        let config = GraphConfig {
            n_left: 100,
            n_right: 100,
            n_isolates: 0,
            thresholds: vec![
                ThresholdConfig {
                    threshold: 0.9,
                    target_entities: 50,
                },
                ThresholdConfig {
                    threshold: 0.7,
                    target_entities: 25,
                },
            ],
        };

        let graph_data = generate_hierarchical_graph(config);

        // Should have created records for all nodes
        assert_eq!(graph_data.context.len(), 200);
        assert_eq!(graph_data.total_nodes, 200);

        // Should have edges at the specified thresholds
        use std::collections::HashSet;
        let thresholds: HashSet<_> = graph_data
            .edges
            .iter()
            .map(|(_, _, w)| (*w * 100.0).round() as i32)
            .collect();

        // Should have exactly 2 distinct thresholds
        assert_eq!(thresholds.len(), 2);
        assert!(thresholds.contains(&90)); // 0.9 * 100
        assert!(thresholds.contains(&70)); // 0.7 * 100
    }

    #[test]
    fn test_production_configs() {
        let config_1m = GraphConfig::production_1m();
        assert_eq!(config_1m.n_left, 550_000);
        assert_eq!(config_1m.n_right, 550_000);
        assert_eq!(config_1m.thresholds.len(), 3);

        let config_10m = GraphConfig::production_10m();
        assert_eq!(config_10m.n_left, 5_500_000);
        assert_eq!(config_10m.n_right, 5_500_000);
        assert_eq!(config_10m.thresholds.len(), 3);
    }
}
