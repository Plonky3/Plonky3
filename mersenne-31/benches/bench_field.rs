use core::any::type_name;
use core::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
use p3_field_testing::bench_func::{
    benchmark_add_latency, benchmark_add_throughput, benchmark_chunked_linear_combination,
    benchmark_dot_array, benchmark_inv, benchmark_iter_sum, benchmark_sqrt, benchmark_sub_latency,
    benchmark_sub_throughput, benchmark_sum_array,
};
use p3_mersenne_31::Mersenne31;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = Mersenne31;

fn bench_field(c: &mut Criterion) {
    let name = "Mersenne31";
    const REPS: usize = 500;
    benchmark_inv::<F>(c, name);
    benchmark_sqrt::<F>(c, name);
    benchmark_iter_sum::<F, 4, REPS>(c, name);
    benchmark_sum_array::<F, 4, REPS>(c, name);
    benchmark_iter_sum::<F, 6, REPS>(c, name);
    benchmark_sum_array::<F, 6, REPS>(c, name);

    // Dot product benchmarks
    benchmark_dot_array::<F, 2>(c, name);
    benchmark_dot_array::<F, 3>(c, name);
    benchmark_dot_array::<F, 4>(c, name);
    benchmark_dot_array::<F, 5>(c, name);
    benchmark_dot_array::<F, 6>(c, name);
    benchmark_dot_array::<F, 7>(c, name);
    benchmark_dot_array::<F, 8>(c, name);
    benchmark_dot_array::<F, 16>(c, name);
    benchmark_dot_array::<F, 64>(c, name);

    // Note that each round of throughput has 10 operations
    // So we should have 10 * more repetitions for latency tests.
    const L_REPS: usize = 10 * REPS;
    benchmark_add_latency::<F, L_REPS>(c, name);
    benchmark_add_throughput::<F, REPS>(c, name);
    benchmark_sub_latency::<F, L_REPS>(c, name);
    benchmark_sub_throughput::<F, REPS>(c, name);

    let mut rng = SmallRng::seed_from_u64(1);
    c.bench_function("5th_root", |b| {
        b.iter_batched(
            || rng.random::<F>(),
            |x| x.exp_u64(1717986917),
            BatchSize::SmallInput,
        );
    });
}

fn bench_packedfield(c: &mut Criterion) {
    let scalar_name = type_name::<F>().to_string();
    benchmark_chunked_linear_combination::<F, F, 100>(c, &scalar_name);

    type PF = <F as Field>::Packing;
    let packed_name = type_name::<PF>().to_string();
    benchmark_chunked_linear_combination::<F, PF, 100>(c, &packed_name);
}

fn bench_packed_two_power_scaling(c: &mut Criterion) {
    type PF = <F as Field>::Packing;

    let mut rng = SmallRng::seed_from_u64(2);
    let mut mul_values = (0..1024)
        .map(|_| PF::from_fn(|_| rng.random()))
        .collect::<Vec<_>>();
    c.bench_function("Mersenne31Packing/mul_2exp_u64/1024", |b| {
        b.iter(|| {
            let exp = black_box(17);
            for value in &mut mul_values {
                *value = value.mul_2exp_u64(exp);
            }
            black_box(&mul_values);
        });
    });

    let mut div_values = (0..1024)
        .map(|_| PF::from_fn(|_| rng.random()))
        .collect::<Vec<_>>();
    c.bench_function("Mersenne31Packing/div_2exp_u64/1024", |b| {
        b.iter(|| {
            let exp = black_box(17);
            for value in &mut div_values {
                *value = value.div_2exp_u64(exp);
            }
            black_box(&div_values);
        });
    });
}

criterion_group!(
    mersenne31_arithmetics,
    bench_field,
    bench_packedfield,
    bench_packed_two_power_scaling
);
criterion_main!(mersenne31_arithmetics);
