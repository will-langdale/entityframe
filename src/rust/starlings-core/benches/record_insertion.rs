use criterion::{black_box, criterion_group, criterion_main, Criterion};
use starlings_core::core::{DataContext, Key};

fn benchmark_10k_unique_insertions(c: &mut Criterion) {
    c.bench_function("10k unique record insertions", |b| {
        b.iter(|| {
            let mut ctx = DataContext::new();
            for i in 0..10_000 {
                ctx.ensure_record("benchmark_source", Key::U64(black_box(i)));
            }
            black_box(ctx.len())
        });
    });
}

fn benchmark_deduplication_overhead(c: &mut Criterion) {
    c.bench_function("10k duplicate record attempts", |b| {
        b.iter(|| {
            let mut ctx = DataContext::new();
            for _ in 0..10_000 {
                ctx.ensure_record(
                    "benchmark_source",
                    Key::String(black_box("same_key".to_string())),
                );
            }
            black_box(ctx.len())
        });
    });
}

fn benchmark_mixed_key_types(c: &mut Criterion) {
    c.bench_function("10k mixed key type insertions", |b| {
        b.iter(|| {
            let mut ctx = DataContext::new();
            for i in 0..10_000 {
                let key = match i % 4 {
                    0 => Key::U32(black_box(i as u32)),
                    1 => Key::U64(black_box(i as u64)),
                    2 => Key::String(black_box(format!("key_{}", i))),
                    _ => Key::Bytes(black_box(vec![i as u8; 4])),
                };
                ctx.ensure_record("benchmark_source", key);
            }
            black_box(ctx.len())
        });
    });
}

fn benchmark_multiple_sources(c: &mut Criterion) {
    c.bench_function("10k records across 10 sources", |b| {
        b.iter(|| {
            let mut ctx = DataContext::new();
            for i in 0..10_000 {
                let source = format!("source_{}", i % 10);
                ctx.ensure_record(&source, Key::U64(black_box(i)));
            }
            black_box(ctx.len())
        });
    });
}

criterion_group!(
    benches,
    benchmark_10k_unique_insertions,
    benchmark_deduplication_overhead,
    benchmark_mixed_key_types,
    benchmark_multiple_sources
);
criterion_main!(benches);
