use core::any::type_name;
use core::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use p3_field::integers::QuotientMap;
use p3_field::{Algebra, Field, PackedValue, PrimeCharacteristicRing};
use p3_field_testing::bench_func::{
    benchmark_add_latency, benchmark_add_throughput, benchmark_chunked_linear_combination,
    benchmark_div_2exp, benchmark_double_latency, benchmark_double_throughput, benchmark_halve,
    benchmark_inv, benchmark_iter_sum, benchmark_mul_2exp, benchmark_neg_latency,
    benchmark_neg_throughput, benchmark_sqrt, benchmark_square, benchmark_sub_latency,
    benchmark_sub_throughput,
};
use p3_field_testing::{
    benchmark_dot_array, benchmark_mixed_dot_array, benchmark_mul_latency,
    benchmark_mul_throughput, benchmark_sum_array,
};
use p3_goldilocks::Goldilocks;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = Goldilocks;

fn benchmark_sqrt_varied<const N: usize>(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0x5A7_C0DE + N as u64);
    let residues: [F; N] = core::array::from_fn(|_| rng.random::<F>().square());
    let raw_inputs: [F; N] = core::array::from_fn(|_| F::new(rng.random()));

    c.bench_function(&format!("Goldilocks sqrt varied residues/{N}"), |b| {
        let mut index = 0;
        b.iter(|| {
            let input = black_box(residues[index]);
            index = (index + 1) % N;
            black_box(input.try_sqrt())
        });
    });

    c.bench_function(&format!("Goldilocks sqrt varied raw/{N}"), |b| {
        let mut index = 0;
        b.iter(|| {
            let input = black_box(raw_inputs[index]);
            index = (index + 1) % N;
            black_box(input.try_sqrt())
        });
    });
}

fn bench_field(c: &mut Criterion) {
    let name = "Goldilocks";
    const REPS: usize = 200;
    benchmark_mul_latency::<F, 100>(c, name);
    benchmark_mul_throughput::<F, 25>(c, name);
    benchmark_square::<F>(c, name);
    benchmark_inv::<F>(c, name);
    benchmark_sqrt::<F>(c, name);
    benchmark_sqrt_varied::<256>(c);
    benchmark_sqrt_varied::<1024>(c);
    benchmark_iter_sum::<F, 4, REPS>(c, name);

    benchmark_sum_array::<F, 4, REPS>(c, name);
    benchmark_sum_array::<F, 5, REPS>(c, name);
    benchmark_sum_array::<F, 6, REPS>(c, name);
    benchmark_sum_array::<F, 7, REPS>(c, name);

    benchmark_dot_array::<F, 1>(c, name);
    benchmark_dot_array::<F, 2>(c, name);
    benchmark_dot_array::<F, 3>(c, name);
    benchmark_dot_array::<F, 4>(c, name);
    benchmark_dot_array::<F, 5>(c, name);
    benchmark_dot_array::<F, 6>(c, name);

    // Note that each round of throughput has 10 operations
    // So we should have 10 * more repetitions for latency tests.
    const L_REPS: usize = 10 * REPS;
    benchmark_add_latency::<F, L_REPS>(c, name);
    benchmark_add_throughput::<F, REPS>(c, name);
    benchmark_sub_latency::<F, L_REPS>(c, name);
    benchmark_sub_throughput::<F, REPS>(c, name);

    benchmark_halve::<F, REPS>(c, name);

    benchmark_mul_2exp::<F, REPS>(c, name, 1);
    benchmark_mul_2exp::<F, REPS>(c, name, 10);
    benchmark_mul_2exp::<F, REPS>(c, name, 32);
    benchmark_mul_2exp::<F, REPS>(c, name, 63);

    benchmark_div_2exp::<F, REPS>(c, name, 1);
    benchmark_div_2exp::<F, REPS>(c, name, 3);
    benchmark_div_2exp::<F, REPS>(c, name, 5);
    benchmark_div_2exp::<F, REPS>(c, name, 10);
    benchmark_div_2exp::<F, REPS>(c, name, 32);

    benchmark_neg_latency::<F, L_REPS>(c, name);
    benchmark_neg_throughput::<F, REPS>(c, name);
    benchmark_double_latency::<F, L_REPS>(c, name);
    benchmark_double_throughput::<F, REPS>(c, name);

    benchmark_chunked_linear_combination::<F, F, 100>(c, name);

    let mut rng = SmallRng::seed_from_u64(1);
    c.bench_function("7th_root", |b| {
        b.iter_batched(
            || rng.random::<F>(),
            |x| x.exp_u64(10540996611094048183),
            BatchSize::SmallInput,
        );
    });
}

fn bench_large_integer_conversion(c: &mut Criterion) {
    const INPUT_COUNT: usize = 1024;

    let mut rng = SmallRng::seed_from_u64(0x128_C0DE);
    let unsigned_inputs: [u128; INPUT_COUNT] = core::array::from_fn(|_| rng.random());
    let signed_inputs: [i128; INPUT_COUNT] = core::array::from_fn(|_| rng.random());

    c.bench_function("from_u128", |b| {
        let mut index = 0;
        b.iter(|| {
            let input = black_box(unsigned_inputs[index]);
            index = (index + 1) % INPUT_COUNT;
            black_box(F::from_int(input))
        });
    });

    c.bench_function("from_i128", |b| {
        let mut index = 0;
        b.iter(|| {
            let input = black_box(signed_inputs[index]);
            index = (index + 1) % INPUT_COUNT;
            black_box(F::from_int(input))
        });
    });
}

fn benchmark_packed_fixed_power<R, const REPS: usize, const EXP: u64>(c: &mut Criterion, name: &str)
where
    R: PrimeCharacteristicRing + Copy,
    StandardUniform: Distribution<R>,
{
    let mut rng = SmallRng::seed_from_u64(0x2E80_2E80 + EXP);
    let input = (0..REPS).map(|_| rng.random::<R>()).collect::<Vec<_>>();
    let mut mul_input = input.clone();
    let mut div_input = input;

    c.bench_function(&format!("{name} mul_2exp_u64 fixed {EXP}"), |b| {
        b.iter(|| {
            mul_input
                .iter_mut()
                .for_each(|value| *value = value.mul_2exp_u64(EXP));
            black_box(mul_input.as_slice());
        });
    });
    c.bench_function(&format!("{name} div_2exp_u64 fixed {EXP}"), |b| {
        b.iter(|| {
            div_input
                .iter_mut()
                .for_each(|value| *value = value.div_2exp_u64(EXP));
            black_box(div_input.as_slice());
        });
    });
}

fn benchmark_packed_runtime_powers<R, const REPS: usize>(c: &mut Criterion, name: &str)
where
    R: PrimeCharacteristicRing + Copy,
    StandardUniform: Distribution<R>,
{
    let mut rng = SmallRng::seed_from_u64(0x2E80_2E80_5EED);
    let input = (0..REPS).map(|_| rng.random::<R>()).collect::<Vec<_>>();
    let exponents = (0..REPS).map(|_| rng.random::<u64>()).collect::<Vec<_>>();
    let mut mul_input = input.clone();
    let mut div_input = input;

    c.bench_function(&format!("{name} mul_2exp_u64 runtime"), |b| {
        b.iter(|| {
            mul_input
                .iter_mut()
                .zip(&exponents)
                .for_each(|(value, &exp)| *value = value.mul_2exp_u64(black_box(exp)));
            black_box(mul_input.as_slice());
        });
    });
    c.bench_function(&format!("{name} div_2exp_u64 runtime"), |b| {
        b.iter(|| {
            div_input
                .iter_mut()
                .zip(&exponents)
                .for_each(|(value, &exp)| *value = value.div_2exp_u64(black_box(exp)));
            black_box(div_input.as_slice());
        });
    });
}

fn benchmark_packed_batched_linear_combination(c: &mut Criterion, name: &str) {
    type PF = <F as Field>::Packing;
    let mut rng = SmallRng::seed_from_u64(0x5E2_51CE);
    for len in [0, 1, 2, 3, 4, 5, 6, 16, 32, 63, 64, 65, 100, 129, 1024] {
        let values = (0..len)
            .map(|_| PF::from_fn(|_| F::new(rng.random())))
            .collect::<Vec<_>>();
        let coeffs = (0..len).map(|_| F::new(rng.random())).collect::<Vec<_>>();
        c.bench_function(&format!("{name} batched_linear_combination/{len}"), |b| {
            b.iter(|| {
                black_box(PF::batched_linear_combination(
                    black_box(values.as_slice()),
                    black_box(coeffs.as_slice()),
                ))
            });
        });
    }
}

fn bench_packedfield(c: &mut Criterion) {
    let name = type_name::<<F as Field>::Packing>().to_string();
    type PF = <F as Field>::Packing;
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

    benchmark_packed_fixed_power::<PF, REPS, 1>(c, &name);
    benchmark_packed_fixed_power::<PF, REPS, 5>(c, &name);
    benchmark_packed_fixed_power::<PF, REPS, 32>(c, &name);
    benchmark_packed_fixed_power::<PF, REPS, 63>(c, &name);
    benchmark_packed_runtime_powers::<PF, REPS>(c, &name);

    benchmark_dot_array::<<F as Field>::Packing, 1>(c, &name);
    benchmark_dot_array::<<F as Field>::Packing, 2>(c, &name);
    benchmark_dot_array::<<F as Field>::Packing, 3>(c, &name);
    benchmark_dot_array::<<F as Field>::Packing, 4>(c, &name);
    benchmark_dot_array::<<F as Field>::Packing, 5>(c, &name);
    benchmark_dot_array::<<F as Field>::Packing, 6>(c, &name);

    benchmark_sum_array::<PF, 3, 100>(c, &name);
    benchmark_sum_array::<PF, 4, 100>(c, &name);
    benchmark_sum_array::<PF, 5, 100>(c, &name);
    benchmark_sum_array::<PF, 6, 100>(c, &name);
    benchmark_sum_array::<PF, 7, 100>(c, &name);
    benchmark_sum_array::<PF, 8, 100>(c, &name);
    benchmark_sum_array::<PF, 12, 100>(c, &name);
    benchmark_sum_array::<PF, 16, 100>(c, &name);
    benchmark_sum_array::<PF, 32, 100>(c, &name);
    benchmark_sum_array::<PF, 64, 100>(c, &name);
    benchmark_sum_array::<PF, 129, 100>(c, &name);

    benchmark_chunked_linear_combination::<F, PF, 100>(c, &name);
    benchmark_packed_batched_linear_combination(c, &name);

    benchmark_mixed_dot_array::<PF, F, 1>(c, &name);
    benchmark_mixed_dot_array::<PF, F, 2>(c, &name);
    benchmark_mixed_dot_array::<PF, F, 3>(c, &name);
    benchmark_mixed_dot_array::<PF, F, 4>(c, &name);
    benchmark_mixed_dot_array::<PF, F, 5>(c, &name);
    benchmark_mixed_dot_array::<PF, F, 6>(c, &name);
}

criterion_group!(
    goldilocks_arithmetic,
    bench_field,
    bench_large_integer_conversion,
    bench_packedfield
);
criterion_main!(goldilocks_arithmetic);
