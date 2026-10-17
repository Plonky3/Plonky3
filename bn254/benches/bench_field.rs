use core::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use p3_bn254::{
    BN254_POSEIDON2_HALF_FULL_ROUNDS, BN254_POSEIDON2_PARTIAL_ROUNDS_3, Bn254, Poseidon2Bn254,
};
use p3_field_testing::bench_func::{
    benchmark_add_latency, benchmark_add_throughput, benchmark_inv, benchmark_sqrt,
    benchmark_sub_latency, benchmark_sub_throughput,
};
use p3_field_testing::{benchmark_halve, benchmark_mul_latency, benchmark_mul_throughput};
use p3_symmetric::Permutation;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = Bn254;

fn bench_field(c: &mut Criterion) {
    let name = "BN254Fr";
    const REPS: usize = 100;
    benchmark_halve::<F, REPS>(c, name);
    benchmark_inv::<F>(c, name);
    benchmark_sqrt::<F>(c, name);

    // Note that each round of throughput has 10 operations
    // So we should have 10 * more repetitions for latency tests.
    const L_REPS: usize = 10 * REPS;
    benchmark_add_latency::<F, L_REPS>(c, name);
    benchmark_add_throughput::<F, REPS>(c, name);
    benchmark_sub_latency::<F, L_REPS>(c, name);
    benchmark_sub_throughput::<F, REPS>(c, name);
    benchmark_mul_latency::<F, L_REPS>(c, name);
    benchmark_mul_throughput::<F, REPS>(c, name);
}

fn bench_poseidon2(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(1);
    let permutation = Poseidon2Bn254::<3>::new_from_rng(
        2 * BN254_POSEIDON2_HALF_FULL_ROUNDS,
        BN254_POSEIDON2_PARTIAL_ROUNDS_3,
        &mut rng,
    );
    let inputs: [[F; 3]; 64] = core::array::from_fn(|_| core::array::from_fn(|_| rng.random()));
    c.bench_function("BN254Fr Poseidon2 varied inputs/3", |b| {
        let mut index = 0;
        b.iter(|| {
            let input = black_box(inputs[index]);
            index = (index + 1) % inputs.len();
            black_box(permutation.permute(input))
        });
    });
}

criterion_group!(bn254fr_arithmetic, bench_field, bench_poseidon2);
criterion_main!(bn254fr_arithmetic);
