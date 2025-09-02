use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use starlings_core::core::{DataContext, Key};
use starlings_core::hierarchy::PartitionHierarchy;
use std::sync::Arc;

fn generate_test_hierarchy(num_edges: usize) -> PartitionHierarchy {
    let mut ctx = DataContext::new();
    let mut edges = Vec::with_capacity(num_edges);

    // Create enough nodes for the requested edges
    let num_nodes = ((num_edges as f64).sqrt() * 2.0).ceil() as usize;

    // Add nodes to context
    for i in 0..num_nodes {
        ctx.ensure_record("test", Key::U32(i as u32));
    }

    let ctx = Arc::new(ctx);

    // Generate random edges with varying weights
    use std::collections::HashSet;
    let mut used_edges = HashSet::new();

    for _ in 0..num_edges {
        loop {
            let src = fastrand::usize(0..num_nodes) as u32;
            let dst = fastrand::usize(0..num_nodes) as u32;

            if src != dst && !used_edges.contains(&(src.min(dst), src.max(dst))) {
                used_edges.insert((src.min(dst), src.max(dst)));
                let weight = fastrand::f64();
                edges.push((src, dst, weight));
                break;
            }
        }
    }

    PartitionHierarchy::from_edges(edges, ctx, 6)
}

fn bench_uncached_at_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("uncached_at_threshold");

    for size in [100, 1000, 10000].iter() {
        let mut hierarchy = generate_test_hierarchy(*size);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_edges", size)),
            &size,
            |b, _| {
                b.iter(|| {
                    // Clear cache by accessing many different thresholds
                    for i in 0..12 {
                        let threshold = i as f64 * 0.08;
                        hierarchy.at_threshold(threshold);
                    }
                    // Now access a new threshold (uncached)
                    black_box(hierarchy.at_threshold(0.95));
                })
            },
        );
    }

    group.finish();
}

fn bench_cached_at_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("cached_at_threshold");

    for size in [100, 1000, 10000].iter() {
        let mut hierarchy = generate_test_hierarchy(*size);

        // Pre-warm the cache with threshold 0.5
        hierarchy.at_threshold(0.5);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_edges", size)),
            &size,
            |b, _| {
                b.iter(|| {
                    // Access the cached threshold
                    black_box(hierarchy.at_threshold(0.5));
                })
            },
        );
    }

    group.finish();
}

fn bench_cache_impact(c: &mut Criterion) {
    let mut hierarchy = generate_test_hierarchy(1000);
    let mut group = c.benchmark_group("cache_impact");

    // Benchmark first access (uncached)
    group.bench_function("first_access", |b| {
        let mut counter = 0.0;
        b.iter(|| {
            // Use a different threshold each time to avoid cache
            counter += 0.001;
            let threshold = (counter % 0.9) + 0.05;
            black_box(hierarchy.at_threshold(threshold));
        })
    });

    // Pre-warm cache for a specific threshold
    hierarchy.at_threshold(0.75);

    // Benchmark repeated access (cached)
    group.bench_function("cached_access", |b| {
        b.iter(|| {
            black_box(hierarchy.at_threshold(0.75));
        })
    });

    group.finish();
}

fn bench_multiple_threshold_sweep(c: &mut Criterion) {
    let mut hierarchy = generate_test_hierarchy(5000);

    c.bench_function("threshold_sweep_20_points", |b| {
        b.iter(|| {
            // Sweep through 20 threshold points
            for i in 0..20 {
                let threshold = i as f64 * 0.05;
                black_box(hierarchy.at_threshold(threshold));
            }
        })
    });
}

criterion_group!(
    benches,
    bench_uncached_at_threshold,
    bench_cached_at_threshold,
    bench_cache_impact,
    bench_multiple_threshold_sweep
);
criterion_main!(benches);
