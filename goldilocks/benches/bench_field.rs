use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use p3_field_testing::bench_func::{benchmark_iter_sum, benchmark_inv, benchmark_add_latency, benchmark_add_throughput, benchmark_sub_latency, benchmark_sub_throughput};
use p3_field::{AbstractField};
use p3_goldilocks::Goldilocks;

type F = Goldilocks;

fn bench_field(c: &mut Criterion) {
    let name = "Goldilocks";
    benchmark_inv::<F>(c, name);
    benchmark_iter_sum::<F, 1000, 4>(c, name);
    benchmark_iter_sum::<F, 1000, 8>(c, name);
    benchmark_iter_sum::<F, 1000, 12>(c, name);

    benchmark_add_latency::<F, 10000>(c, name);
    benchmark_add_throughput::<F, 1000>(c, name);
    benchmark_sub_latency::<F, 10000>(c, name);
    benchmark_sub_throughput::<F, 10000>(c, name);

    c.bench_function("7th_root", |b| {
        b.iter_batched(
            rand::random::<F>,
            |x| x.exp_u64(10540996611094048183),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(goldilocks_arithmetic, bench_field);
criterion_main!(goldilocks_arithmetic);
