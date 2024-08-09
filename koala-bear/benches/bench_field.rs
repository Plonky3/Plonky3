use std::any::type_name;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use p3_field::{AbstractField, Field};
use p3_field_testing::bench_func::{
    benchmark_add_latency, benchmark_add_throughput, benchmark_inv, benchmark_iter_sum,
    benchmark_mul_latency, benchmark_mul_throughput, benchmark_sub_latency,
    benchmark_sub_throughput,
};
use p3_koala_bear::KoalaBear;

type F = KoalaBear;

fn bench_field(c: &mut Criterion) {
    let name = "KoalaBear";
    const REPS: usize = 1000;
    benchmark_inv::<F>(c, name);
    benchmark_iter_sum::<F, 4, REPS>(c, name);
    benchmark_iter_sum::<F, 8, REPS>(c, name);
    benchmark_iter_sum::<F, 12, REPS>(c, name);

    // Note that each round of throughput has 10 operations
    // So we should have 10 * more repetitions for latency tests.
    const L_REPS: usize = 10 * REPS;
    benchmark_add_latency::<F, L_REPS>(c, name);
    benchmark_add_throughput::<F, REPS>(c, name);
    benchmark_sub_latency::<F, L_REPS>(c, name);
    benchmark_sub_throughput::<F, REPS>(c, name);

    c.bench_function("3rd_root", |b| {
        b.iter_batched(
            rand::random::<F>,
            |x| x.exp_u64(1420470955),
            BatchSize::SmallInput,
        )
    });
}

fn bench_packedfield(c: &mut Criterion) {
    let name = type_name::<<F as Field>::Packing>().to_string();
    // Note that each round of throughput has 10 operations
    // So we should have 10 * more repetitions for latency tests.
    const REPS: usize = 100;
    const L_REPS: usize = 10 * REPS;

    benchmark_add_latency::<<F as Field>::Packing, L_REPS>(c, &name);
    benchmark_add_throughput::<<F as Field>::Packing, REPS>(c, &name);
    benchmark_sub_latency::<<F as Field>::Packing, L_REPS>(c, &name);
    benchmark_sub_throughput::<<F as Field>::Packing, REPS>(c, &name);
    benchmark_mul_latency::<<F as Field>::Packing, L_REPS>(c, &name);
    benchmark_mul_throughput::<<F as Field>::Packing, REPS>(c, &name);
}

criterion_group!(koala_bear_arithmetic, bench_field, bench_packedfield);
criterion_main!(koala_bear_arithmetic);
