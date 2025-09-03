use criterion::{black_box, criterion_group, criterion_main, Criterion};
use starlings_core::core::{DataContext, Key};
use starlings_core::hierarchy::PartitionHierarchy;
use std::sync::Arc;

fn generate_test_hierarchy(num_edges: usize) -> PartitionHierarchy {
    println!(
        "Generating {}M edge hierarchy for partition reconstruction testing...",
        num_edges as f64 / 1_000_000.0
    );

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

    // Generate realistic edge weight distribution for entity resolution
    use std::collections::HashSet;
    let mut used_edges = HashSet::new();

    for i in 0..num_edges {
        loop {
            let src = fastrand::usize(0..num_nodes) as u32;
            let dst = fastrand::usize(0..num_nodes) as u32;

            if src != dst && !used_edges.contains(&(src.min(dst), src.max(dst))) {
                used_edges.insert((src.min(dst), src.max(dst)));

                // Realistic weight distribution: high confidence clusters, medium links, noise
                let weight = match i * 10 / num_edges {
                    0..=2 => 0.8 + fastrand::f64() * 0.2, // 30% high confidence
                    3..=7 => 0.5 + fastrand::f64() * 0.3, // 50% medium confidence
                    _ => 0.1 + fastrand::f64() * 0.4,     // 20% noise
                };

                edges.push((src, dst, weight));
                break;
            }
        }
    }

    println!(
        "Built hierarchy with {} edges and {} records",
        edges.len(),
        ctx.len()
    );
    PartitionHierarchy::from_edges(edges, ctx, 6)
}

fn bench_partition_reconstruction_10m(c: &mut Criterion) {
    let mut hierarchy = generate_test_hierarchy(10_000_000);

    let mut group = c.benchmark_group("partition_reconstruction_production");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));

    group.bench_function("10M_edges_threshold_access", |b| {
        b.iter(|| {
            // Test uncached access to different thresholds
            black_box(hierarchy.at_threshold(0.95));
            black_box(hierarchy.at_threshold(0.85));
            black_box(hierarchy.at_threshold(0.75));
        })
    });

    group.finish();
}

criterion_group!(benches, bench_partition_reconstruction_10m);
criterion_main!(benches);
