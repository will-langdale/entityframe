use criterion::{black_box, criterion_group, criterion_main, Criterion};
use starlings_core::core::{DataContext, Key};
use starlings_core::hierarchy::PartitionHierarchy;
use std::sync::Arc;

fn generate_test_edges(num_edges: usize) -> (Vec<(u32, u32, f64)>, Arc<DataContext>) {
    let ctx = DataContext::new();
    let mut edges = Vec::with_capacity(num_edges);

    // Create a realistic entity resolution graph
    // Use a higher node-to-edge ratio for realistic density (typical: 2-5x more edges than nodes)
    let num_nodes = (num_edges as f64 / 3.0).ceil() as usize;

    // Add nodes to context - use realistic string keys instead of just integers
    for i in 0..num_nodes {
        // Mix of different key types for realism
        match i % 4 {
            0 => ctx.ensure_record("customers", Key::String(format!("cust_{}", i))),
            1 => ctx.ensure_record("transactions", Key::U64(1000000 + i as u64)),
            2 => ctx.ensure_record("products", Key::String(format!("prod_{}", i))),
            3 => ctx.ensure_record("addresses", Key::U32(i as u32)),
            _ => unreachable!(),
        };
    }

    let ctx = Arc::new(ctx);

    // Generate more realistic edge patterns
    // 1. High-confidence clusters (0.8-1.0) - 30% of edges
    // 2. Medium-confidence links (0.5-0.8) - 50% of edges
    // 3. Low-confidence noise (0.1-0.5) - 20% of edges

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

    (edges, ctx)
}

fn bench_hierarchy_construction_progressive(c: &mut Criterion) {
    // Progressive scale benchmarks to avoid memory issues
    let scales = vec![
        ("100k", 100_000),
        ("500k", 500_000),
        ("1M", 1_000_000),
        ("2M", 2_000_000),
        ("5M", 5_000_000),
    ];

    let mut group = c.benchmark_group("hierarchy_construction");
    group.sample_size(10); // Minimum required by Criterion

    for (name, num_edges) in scales {
        println!("Generating {} edge test dataset...", name);
        let (edges, ctx) = generate_test_edges(num_edges);
        println!("Generated {} edges with {} records", edges.len(), ctx.len());

        // Track memory usage
        let before_rss = get_memory_usage_mb();

        group.bench_function(format!("{}_edges", name), |b| {
            b.iter(|| {
                black_box(PartitionHierarchy::from_edges(
                    edges.clone(),
                    ctx.clone(),
                    6, // Maximum quantisation
                ))
            })
        });

        let after_rss = get_memory_usage_mb();
        println!(
            "Memory usage for {}: {:.1} MB -> {:.1} MB (Δ {:.1} MB)",
            name,
            before_rss,
            after_rss,
            after_rss - before_rss
        );
    }

    group.finish();
}

/// Get current process memory usage in MB
fn get_memory_usage_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<f64>() {
                            return kb / 1024.0;
                        }
                    }
                }
            }
        }
    }
    0.0
}

criterion_group!(benches, bench_hierarchy_construction_progressive);
criterion_main!(benches);
