use core::time::Duration;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{
    BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS, BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16,
    BABYBEAR_S_BOX_DEGREE, BabyBear, GenericPoseidon2LinearLayersBabyBear, Poseidon2BabyBear,
};
use p3_challenger::DuplexChallenger;
use p3_field::extension::BinomialExtensionField;
use p3_multi_stark::zerocheck::AirZerocheck;
use p3_poseidon2_air::{Poseidon2Air, RoundConstants};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type Ch = DuplexChallenger<F, Perm, 16, 8>;

const POSEIDON2_WIDTH: usize = 16;
const POSEIDON2_SBOX_DEGREE: u64 = BABYBEAR_S_BOX_DEGREE;
const POSEIDON2_SBOX_REGISTERS: usize = 1;
const POSEIDON2_HALF_FULL_ROUNDS: usize = BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS;
const POSEIDON2_PARTIAL_ROUNDS: usize = BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16;

type BabyBearPoseidon2Air = Poseidon2Air<
    F,
    GenericPoseidon2LinearLayersBabyBear,
    POSEIDON2_WIDTH,
    POSEIDON2_SBOX_DEGREE,
    POSEIDON2_SBOX_REGISTERS,
    POSEIDON2_HALF_FULL_ROUNDS,
    POSEIDON2_PARTIAL_ROUNDS,
>;

fn fresh_challenger() -> Ch {
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    let perm = Perm::new_from_rng_128(&mut rng);
    Ch::new(perm)
}

fn poseidon2_air() -> BabyBearPoseidon2Air {
    let mut rng = SmallRng::seed_from_u64(1);
    Poseidon2Air::new(RoundConstants::from_rng(&mut rng))
}

fn bench_num_vars() -> Vec<usize> {
    let value = std::env::var("ZEROCHECK_POSEIDON2_NUM_VARS")
        .ok()
        .unwrap_or_else(|| "15".to_string());

    value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.parse().expect("invalid num_vars"))
        .collect()
}

fn bench_poseidon2_zerocheck_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon2_zerocheck_prove");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(10));

    for num_vars in bench_num_vars() {
        let num_hashes = 1 << num_vars;
        let air = poseidon2_air();
        let trace = air.generate_trace_rows(num_hashes, 0);
        let zerocheck = AirZerocheck::new(&air, 0);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_vars),
            &num_vars,
            |b, &num_vars| {
                b.iter(|| {
                    let mut challenger = fresh_challenger();
                    let (proof, point) = zerocheck.prove::<F, EF, _>(&trace, &[], &mut challenger);
                    black_box((proof, point, num_vars));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_poseidon2_zerocheck_prove
);
criterion_main!(benches);
