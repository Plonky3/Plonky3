use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{
    BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS, BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_16,
    BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_24, BabyBear, MdsMatrixBabyBear, Poseidon1BabyBear,
};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks;
use p3_goldilocks::poseidon1::{default_goldilocks_poseidon1_8, default_goldilocks_poseidon1_12};
use p3_koala_bear::{
    KOALABEAR_POSEIDON_HALF_FULL_ROUNDS, KOALABEAR_POSEIDON_PARTIAL_ROUNDS_16,
    KOALABEAR_POSEIDON_PARTIAL_ROUNDS_24, KoalaBear, MdsMatrixKoalaBear, Poseidon1KoalaBear,
};
use p3_mersenne_31::{
    Mersenne31, default_mersenne31_poseidon1_16, default_mersenne31_poseidon1_32,
};
use p3_symmetric::Permutation;
use p3_util::pretty_name;
use rand::SeedableRng;
use rand::rngs::SmallRng;

fn bench_poseidon1(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(1);

    let mds_bb: MdsMatrixBabyBear = Default::default();

    // BabyBear width 16.
    let poseidon_bb_16 = Poseidon1BabyBear::<16>::new_from_rng(
        BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS,
        BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_16,
        &mds_bb,
        &mut rng,
    );
    poseidon1_scalar::<BabyBear, _, 16>(c, &poseidon_bb_16);
    poseidon1_packed::<BabyBear, _, 16>(c, &poseidon_bb_16);

    // BabyBear width 24.
    let poseidon_bb_24 = Poseidon1BabyBear::<24>::new_from_rng(
        BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS,
        BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_24,
        &mds_bb,
        &mut rng,
    );
    poseidon1_scalar::<BabyBear, _, 24>(c, &poseidon_bb_24);
    poseidon1_packed::<BabyBear, _, 24>(c, &poseidon_bb_24);

    let mds_kb: MdsMatrixKoalaBear = Default::default();

    // KoalaBear width 16.
    let poseidon_kb_16 = Poseidon1KoalaBear::<16>::new_from_rng(
        KOALABEAR_POSEIDON_HALF_FULL_ROUNDS,
        KOALABEAR_POSEIDON_PARTIAL_ROUNDS_16,
        &mds_kb,
        &mut rng,
    );
    poseidon1_scalar::<KoalaBear, _, 16>(c, &poseidon_kb_16);
    poseidon1_packed::<KoalaBear, _, 16>(c, &poseidon_kb_16);

    // KoalaBear width 24.
    let poseidon_kb_24 = Poseidon1KoalaBear::<24>::new_from_rng(
        KOALABEAR_POSEIDON_HALF_FULL_ROUNDS,
        KOALABEAR_POSEIDON_PARTIAL_ROUNDS_24,
        &mds_kb,
        &mut rng,
    );
    poseidon1_scalar::<KoalaBear, _, 24>(c, &poseidon_kb_24);
    poseidon1_packed::<KoalaBear, _, 24>(c, &poseidon_kb_24);

    // Goldilocks width 8.
    let gl_8 = default_goldilocks_poseidon1_8();
    poseidon1_scalar::<Goldilocks, _, 8>(c, &gl_8);
    poseidon1_packed::<Goldilocks, _, 8>(c, &gl_8);

    // Goldilocks width 12.
    let gl_12 = default_goldilocks_poseidon1_12();
    poseidon1_scalar::<Goldilocks, _, 12>(c, &gl_12);
    poseidon1_packed::<Goldilocks, _, 12>(c, &gl_12);

    // Mersenne31 width 16.
    let m31_16 = default_mersenne31_poseidon1_16();
    poseidon1_scalar::<Mersenne31, _, 16>(c, &m31_16);
    poseidon1_packed::<Mersenne31, _, 16>(c, &m31_16);

    // Mersenne31 width 32.
    let m31_32 = default_mersenne31_poseidon1_32();
    poseidon1_scalar::<Mersenne31, _, 32>(c, &m31_32);
    poseidon1_packed::<Mersenne31, _, 32>(c, &m31_32);
}

/// Benchmark using scalar field elements (no SIMD).
fn poseidon1_scalar<F, Perm, const WIDTH: usize>(c: &mut Criterion, poseidon: &Perm)
where
    F: Field,
    Perm: Permutation<[F; WIDTH]>,
{
    let input = [F::ZERO; WIDTH];
    let name = format!("poseidon-scalar::<{}, {}>", pretty_name::<F>(), WIDTH);
    let id = BenchmarkId::new(name, WIDTH);
    c.bench_with_input(id, &input, |b, &input| b.iter(|| poseidon.permute(input)));
}

/// Benchmark using packed (SIMD) field representation.
fn poseidon1_packed<F, Perm, const WIDTH: usize>(c: &mut Criterion, poseidon: &Perm)
where
    F: Field,
    Perm: Permutation<[F::Packing; WIDTH]>,
{
    let input = [F::Packing::ZERO; WIDTH];
    let name = format!(
        "poseidon-packed::<{}, {}>",
        pretty_name::<F::Packing>(),
        WIDTH
    );
    let id = BenchmarkId::new(name, WIDTH);
    c.bench_with_input(id, &input, |b, &input| b.iter(|| poseidon.permute(input)));
}

criterion_group!(benches, bench_poseidon1);
criterion_main!(benches);
