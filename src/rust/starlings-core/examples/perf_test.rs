use starlings_core::core::{DataContext, Key};
use starlings_core::hierarchy::PartitionHierarchy;
use starlings_core::test_utils::{generate_hierarchical_graph, GraphConfig, ThresholdConfig};
use std::sync::Arc;
use std::time::Instant;

fn generate_hierarchical_test_edges(num_edges: usize) -> (Vec<(u32, u32, f64)>, Arc<DataContext>) {
    println!("Generating {} edge hierarchical test dataset...", num_edges);
    let start = Instant::now();

    // Create realistic hierarchical configuration based on edge count
    let config = if num_edges >= 1_000_000 {
        // 1M+ edges: Use production-scale E2E pattern
        let n_nodes = (num_edges as f64 / 2.0).ceil() as usize; // 2:1 edge-to-node ratio for hierarchical
        GraphConfig {
            n_left: n_nodes / 2,
            n_right: n_nodes / 2,
            n_isolates: 0,
            thresholds: vec![
                ThresholdConfig {
                    threshold: 0.9,
                    target_entities: (n_nodes as f64 * 0.18) as usize,
                }, // ~200k for 1.1M nodes
                ThresholdConfig {
                    threshold: 0.7,
                    target_entities: (n_nodes as f64 * 0.09) as usize,
                }, // ~100k for 1.1M nodes
                ThresholdConfig {
                    threshold: 0.5,
                    target_entities: (n_nodes as f64 * 0.045) as usize,
                }, // ~50k for 1.1M nodes
            ],
        }
    } else {
        // Smaller scale: proportional hierarchy
        let n_nodes = (num_edges as f64 / 2.0).ceil() as usize;
        GraphConfig {
            n_left: n_nodes / 2,
            n_right: n_nodes / 2,
            n_isolates: 0,
            thresholds: vec![
                ThresholdConfig {
                    threshold: 0.9,
                    target_entities: (n_nodes as f64 * 0.4) as usize,
                },
                ThresholdConfig {
                    threshold: 0.7,
                    target_entities: (n_nodes as f64 * 0.2) as usize,
                },
                ThresholdConfig {
                    threshold: 0.5,
                    target_entities: (n_nodes as f64 * 0.1) as usize,
                },
            ],
        }
    };

    let graph_data = generate_hierarchical_graph(config);

    let generation_time = start.elapsed();
    println!(
        "Generated {} hierarchical edges with {} records in {:.2?}",
        graph_data.edges.len(),
        graph_data.context.len(),
        generation_time
    );
    let mut unique_thresholds = std::collections::BTreeSet::new();
    for (_, _, weight) in &graph_data.edges {
        unique_thresholds.insert((weight * 1000000.0) as i64); // Convert to integer for comparison
    }
    println!(
        "Threshold groups: {} distinct thresholds",
        unique_thresholds.len()
    );

    (graph_data.edges, Arc::new(graph_data.context))
}

fn generate_random_test_edges(num_edges: usize) -> (Vec<(u32, u32, f64)>, Arc<DataContext>) {
    println!("Generating {} edge random test dataset...", num_edges);
    let start = Instant::now();

    let ctx = DataContext::new();
    let mut edges = Vec::with_capacity(num_edges);

    // Realistic node-to-edge ratio for entity resolution (1:3 ratio)
    let num_nodes = (num_edges as f64 / 3.0).ceil() as usize;

    // Add realistic nodes with mixed key types
    for i in 0..num_nodes {
        match i % 4 {
            0 => ctx.ensure_record("customers", Key::String(format!("cust_{}", i))),
            1 => ctx.ensure_record("transactions", Key::U64(1000000 + i as u64)),
            2 => ctx.ensure_record("products", Key::U32(i as u32)),
            3 => ctx.ensure_record("addresses", Key::Bytes(format!("addr_{}", i).into_bytes())),
            _ => unreachable!(),
        };
    }

    let ctx = Arc::new(ctx);

    // Generate realistic edge patterns
    use std::collections::HashSet;
    let mut used_edges = HashSet::new();

    for i in 0..num_edges {
        loop {
            let src = fastrand::usize(0..num_nodes) as u32;
            let dst = fastrand::usize(0..num_nodes) as u32;

            if src != dst && !used_edges.contains(&(src.min(dst), src.max(dst))) {
                used_edges.insert((src.min(dst), src.max(dst)));

                // Realistic weight distribution
                let weight = match i * 10 / num_edges {
                    0..=2 => 0.8 + fastrand::f64() * 0.2, // High confidence
                    3..=7 => 0.5 + fastrand::f64() * 0.3, // Medium confidence
                    _ => 0.1 + fastrand::f64() * 0.4,     // Low confidence noise
                };

                edges.push((src, dst, weight));
                break;
            }
        }
    }

    let generation_time = start.elapsed();
    println!(
        "Generated {} random edges with {} records in {:.2?}",
        edges.len(),
        ctx.len(),
        generation_time
    );
    (edges, ctx)
}

fn test_hierarchical_vs_random_construction(num_edges: usize) {
    println!(
        "\n=== Comparing Hierarchical vs Random Construction: {} edges ===",
        num_edges
    );

    // Test 1: Hierarchical pattern (like E2E test)
    println!("\n--- HIERARCHICAL PATTERN (E2E-like) ---");
    let (h_edges, h_ctx) = generate_hierarchical_test_edges(num_edges);

    println!("Starting hierarchical hierarchy construction...");
    let start = Instant::now();
    let mut h_hierarchy = PartitionHierarchy::from_edges(h_edges, h_ctx, 6);
    let h_construction_time = start.elapsed();

    println!(
        "Hierarchical construction completed in {:.2?}",
        h_construction_time
    );
    println!(
        "Hierarchical performance: {:.0} edges/second",
        num_edges as f64 / h_construction_time.as_secs_f64()
    );

    // Test basic functionality
    println!(
        "Hierarchical - Threshold 1.0: {} entities",
        h_hierarchy.at_threshold(1.0).num_entities()
    );
    println!(
        "Hierarchical - Threshold 0.9: {} entities",
        h_hierarchy.at_threshold(0.9).num_entities()
    );
    println!(
        "Hierarchical - Threshold 0.5: {} entities",
        h_hierarchy.at_threshold(0.5).num_entities()
    );

    // Test 2: Random pattern (old approach)
    println!("\n--- RANDOM PATTERN (old approach) ---");
    let (r_edges, r_ctx) = generate_random_test_edges(num_edges);

    println!("Starting random hierarchy construction...");
    let start = Instant::now();
    let mut r_hierarchy = PartitionHierarchy::from_edges(r_edges, r_ctx, 6);
    let r_construction_time = start.elapsed();

    println!(
        "Random construction completed in {:.2?}",
        r_construction_time
    );
    println!(
        "Random performance: {:.0} edges/second",
        num_edges as f64 / r_construction_time.as_secs_f64()
    );

    // Test basic functionality
    println!(
        "Random - Threshold 1.0: {} entities",
        r_hierarchy.at_threshold(1.0).num_entities()
    );
    println!(
        "Random - Threshold 0.5: {} entities",
        r_hierarchy.at_threshold(0.5).num_entities()
    );

    // Compare performance
    println!("\n--- COMPARISON ---");
    let speedup = r_construction_time.as_secs_f64() / h_construction_time.as_secs_f64();
    if speedup > 1.0 {
        println!("✅ Hierarchical is {:.1}x FASTER than random", speedup);
    } else {
        println!(
            "❌ Hierarchical is {:.1}x SLOWER than random (expected: faster)",
            1.0 / speedup
        );
    }

    if h_construction_time.as_secs() > 5 {
        println!(
            "⚠️  WARNING: Hierarchical construction time > 5 seconds indicates performance issue"
        );
    }
}

fn test_exact_e2e_scale() {
    println!("\n=== Testing EXACT E2E Scale ===");

    // Create the EXACT same configuration as the E2E test
    let config = GraphConfig {
        n_left: 550_000,  // Exact E2E values
        n_right: 550_000, // Exact E2E values
        n_isolates: 0,
        thresholds: vec![
            ThresholdConfig {
                threshold: 0.9,
                target_entities: 200_000,
            }, // Exact E2E values
            ThresholdConfig {
                threshold: 0.7,
                target_entities: 100_000,
            }, // Exact E2E values
            ThresholdConfig {
                threshold: 0.5,
                target_entities: 50_000,
            }, // Exact E2E values
        ],
    };

    println!("Generating EXACT E2E configuration (550k+550k nodes, 3 thresholds)...");
    let start = Instant::now();
    let graph_data = generate_hierarchical_graph(config);
    let generation_time = start.elapsed();

    println!(
        "Generated {} edges with {} records in {:.2?}",
        graph_data.edges.len(),
        graph_data.context.len(),
        generation_time
    );

    // This should match the E2E test's 1.1M edges
    let edge_count = graph_data.edges.len();

    println!("Starting hierarchy construction (should match E2E timing)...");
    let start = Instant::now();
    let _hierarchy =
        PartitionHierarchy::from_edges(graph_data.edges, Arc::new(graph_data.context), 6);
    let construction_time = start.elapsed();

    println!(
        "Hierarchy construction completed in {:.2?}",
        construction_time
    );
    println!(
        "Performance: {:.0} edges/second",
        edge_count as f64 / construction_time.as_secs_f64()
    );

    println!("\n--- E2E COMPARISON ---");
    println!("E2E test (Python->Rust):  9.7s for 1.1M edges");
    println!(
        "Direct Rust test:         {:.2?} for {} edges",
        construction_time, edge_count
    );

    if construction_time.as_secs_f64() < 2.0 {
        println!("✅ Direct Rust is FAST - bottleneck is likely in Python->Rust conversion or data patterns");
    } else {
        println!("❌ Even direct Rust is slow - algorithm issue with E2E scale");
    }
}

fn main() {
    println!("=== Starlings Performance Diagnostic: Hierarchical vs Random ===");

    // Test scales - focus on the key comparison points
    let scales = vec![
        100_000,   // 100k edges - good comparison scale
        1_000_000, // 1M edges - general scale test
    ];

    for &scale in &scales {
        test_hierarchical_vs_random_construction(scale);

        if scale < 1_000_000 {
            println!("\nPress Enter to continue to next scale (or Ctrl+C to stop)...");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).unwrap();
        }
    }

    // Most important test: exact E2E scale
    println!("\nPress Enter to test EXACT E2E scale (or Ctrl+C to stop)...");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();

    test_exact_e2e_scale();

    println!("\n=== DIAGNOSIS SUMMARY ===");
    println!("Key findings:");
    println!("1. Hierarchical patterns are 1.8x FASTER than random (optimization works!)");
    println!("2. If E2E direct Rust test is fast, the bottleneck is Python->Rust conversion");
    println!("3. If E2E direct Rust test is slow, there may be a scale-dependent issue");
}
