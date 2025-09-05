use criterion::{black_box, criterion_group, criterion_main, Criterion};
use starlings_core::core::{DataContext, Key};
use starlings_core::hierarchy::PartitionHierarchy;
use std::sync::Arc;

fn generate_test_edges(num_edges: usize) -> (Vec<(u32, u32, f64)>, Arc<DataContext>) {
    let ctx = DataContext::new();
    let mut edges = Vec::with_capacity(num_edges);
    let num_nodes = (num_edges as f64 / 3.0).ceil() as usize;

    for i in 0..num_nodes {
        match i % 4 {
            0 => ctx.ensure_record("customers", Key::String(format!("cust_{}", i))),
            1 => ctx.ensure_record("transactions", Key::U64(1000000 + i as u64)),
            2 => ctx.ensure_record("products", Key::String(format!("prod_{}", i))),
            3 => ctx.ensure_record("addresses", Key::U32(i as u32)),
            _ => unreachable!(),
        };
    }

    let ctx = Arc::new(ctx);

    use std::collections::HashSet;
    let mut used_edges = HashSet::new();

    for i in 0..num_edges {
        loop {
            let src = fastrand::usize(0..num_nodes) as u32;
            let dst = fastrand::usize(0..num_nodes) as u32;

            if src != dst && !used_edges.contains(&(src.min(dst), src.max(dst))) {
                used_edges.insert((src.min(dst), src.max(dst)));

                let weight = match i * 10 / num_edges {
                    0..=2 => 0.8 + fastrand::f64() * 0.2,
                    3..=7 => 0.5 + fastrand::f64() * 0.3,
                    _ => 0.1 + fastrand::f64() * 0.4,
                };

                edges.push((src, dst, weight));
                break;
            }
        }
    }

    (edges, ctx)
}

fn bench_hierarchy_construction_progressive(c: &mut Criterion) {
    let scales = vec![("100k", 100_000), ("500k", 500_000), ("1M", 1_000_000)];

    let mut group = c.benchmark_group("hierarchy_construction");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(10));

    for (name, num_edges) in scales {
        let (edges, ctx) = generate_test_edges(num_edges);

        group.bench_function(format!("{}_edges", name), |b| {
            b.iter(|| {
                black_box(PartitionHierarchy::from_edges(
                    edges.clone(),
                    ctx.clone(),
                    6,
                ))
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_hierarchy_construction_progressive);
criterion_main!(benches);
