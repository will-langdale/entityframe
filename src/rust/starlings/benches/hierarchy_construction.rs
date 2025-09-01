use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use starlings::core::{DataContext, Key};
use starlings::hierarchy::PartitionHierarchy;
use std::sync::Arc;

fn generate_test_edges(num_edges: usize) -> (Vec<(u32, u32, f64)>, Arc<DataContext>) {
    let mut ctx = DataContext::new();
    let mut edges = Vec::with_capacity(num_edges);

    // Create a connected graph with random edges
    // Ensure we have enough nodes to create the requested number of edges
    let num_nodes = ((num_edges as f64).sqrt() * 2.0).ceil() as usize;

    // Add nodes to context
    for i in 0..num_nodes {
        ctx.ensure_record("test", Key::U32(i as u32));
    }

    let ctx = Arc::new(ctx);

    // Generate random edges with weights between 0.0 and 1.0
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

    (edges, ctx)
}

fn bench_hierarchy_construction_1k(c: &mut Criterion) {
    let (edges, ctx) = generate_test_edges(1000);

    c.bench_with_input(
        BenchmarkId::new("hierarchy_construction", "1k_edges"),
        &(edges, ctx),
        |b, (edges, ctx)| {
            b.iter(|| {
                black_box(PartitionHierarchy::from_edges(
                    edges.clone(),
                    ctx.clone(),
                    6, // Maximum quantisation
                ))
            })
        },
    );
}

fn bench_hierarchy_construction_10k(c: &mut Criterion) {
    let (edges, ctx) = generate_test_edges(10000);

    c.bench_with_input(
        BenchmarkId::new("hierarchy_construction", "10k_edges"),
        &(edges, ctx),
        |b, (edges, ctx)| {
            b.iter(|| {
                black_box(PartitionHierarchy::from_edges(
                    edges.clone(),
                    ctx.clone(),
                    6, // Maximum quantisation
                ))
            })
        },
    );
}

fn bench_hierarchy_construction_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("hierarchy_construction_scale");

    for size in [100, 500, 1000, 2000, 5000].iter() {
        let (edges, ctx) = generate_test_edges(*size);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(edges, ctx),
            |b, (edges, ctx)| {
                b.iter(|| {
                    black_box(PartitionHierarchy::from_edges(
                        edges.clone(),
                        ctx.clone(),
                        6,
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_quantisation_effect(c: &mut Criterion) {
    let (edges, ctx) = generate_test_edges(1000);
    let mut group = c.benchmark_group("quantisation_effect");

    for quantise in [1, 2, 3, 4, 5, 6].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(quantise),
            &(edges.clone(), ctx.clone(), *quantise),
            |b, (edges, ctx, quantise)| {
                b.iter(|| {
                    black_box(PartitionHierarchy::from_edges(
                        edges.clone(),
                        ctx.clone(),
                        *quantise,
                    ))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hierarchy_construction_1k,
    bench_hierarchy_construction_10k,
    bench_hierarchy_construction_scale,
    bench_quantisation_effect
);
criterion_main!(benches);
