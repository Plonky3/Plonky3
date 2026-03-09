use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{
    BABYBEAR_POSEIDON_HALF_FULL_ROUNDS, BABYBEAR_POSEIDON_PARTIAL_ROUNDS_16,
    BABYBEAR_POSEIDON_PARTIAL_ROUNDS_24, BabyBear, MdsMatrixBabyBear, PoseidonBabyBear,
};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks;
use p3_goldilocks::poseidon::{default_goldilocks_poseidon_8, default_goldilocks_poseidon_12};
use p3_koala_bear::{
    KOALABEAR_POSEIDON_HALF_FULL_ROUNDS, KOALABEAR_POSEIDON_PARTIAL_ROUNDS_16,
    KOALABEAR_POSEIDON_PARTIAL_ROUNDS_24, KoalaBear, MdsMatrixKoalaBear, PoseidonKoalaBear,
};
use p3_mersenne_31::{
    MERSENNE31_POSEIDON2_HALF_FULL_ROUNDS, MERSENNE31_POSEIDON2_PARTIAL_ROUNDS_16,
    MERSENNE31_POSEIDON2_PARTIAL_ROUNDS_24, MERSENNE31_S_BOX_DEGREE, MdsMatrixMersenne31,
    Mersenne31,
};
use p3_poseidon::{Poseidon, PoseidonExternalLayerGeneric, PoseidonInternalLayerGeneric};
use p3_symmetric::Permutation;
use p3_util::pretty_name;
use rand::SeedableRng;
use rand::rngs::SmallRng;

type PoseidonGeneric<F, Mds, const WIDTH: usize, const ALPHA: u64> = Poseidon<
    F,
    PoseidonExternalLayerGeneric<F, Mds, WIDTH>,
    PoseidonInternalLayerGeneric<F, WIDTH>,
    WIDTH,
    ALPHA,
>;

fn bench_poseidon(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(1);

    let mds_bb: MdsMatrixBabyBear = Default::default();

    // BabyBear width 16.
    let poseidon_bb_16 = PoseidonBabyBear::<16>::new_from_rng(
        BABYBEAR_POSEIDON_HALF_FULL_ROUNDS,
        BABYBEAR_POSEIDON_PARTIAL_ROUNDS_16,
        &mds_bb,
        &mut rng,
    );
    poseidon_scalar::<BabyBear, _, 16>(c, &poseidon_bb_16);
    poseidon_packed::<BabyBear, _, 16>(c, &poseidon_bb_16);

    // BabyBear width 24.
    let poseidon_bb_24 = PoseidonBabyBear::<24>::new_from_rng(
        BABYBEAR_POSEIDON_HALF_FULL_ROUNDS,
        BABYBEAR_POSEIDON_PARTIAL_ROUNDS_24,
        &mds_bb,
        &mut rng,
    );
    poseidon_scalar::<BabyBear, _, 24>(c, &poseidon_bb_24);
    poseidon_packed::<BabyBear, _, 24>(c, &poseidon_bb_24);

    let mds_kb: MdsMatrixKoalaBear = Default::default();

    // KoalaBear width 16.
    let poseidon_kb_16 = PoseidonKoalaBear::<16>::new_from_rng(
        KOALABEAR_POSEIDON_HALF_FULL_ROUNDS,
        KOALABEAR_POSEIDON_PARTIAL_ROUNDS_16,
        &mds_kb,
        &mut rng,
    );
    poseidon_scalar::<KoalaBear, _, 16>(c, &poseidon_kb_16);
    poseidon_packed::<KoalaBear, _, 16>(c, &poseidon_kb_16);

    // KoalaBear width 24.
    let poseidon_kb_24 = PoseidonKoalaBear::<24>::new_from_rng(
        KOALABEAR_POSEIDON_HALF_FULL_ROUNDS,
        KOALABEAR_POSEIDON_PARTIAL_ROUNDS_24,
        &mds_kb,
        &mut rng,
    );
    poseidon_scalar::<KoalaBear, _, 24>(c, &poseidon_kb_24);
    poseidon_packed::<KoalaBear, _, 24>(c, &poseidon_kb_24);

    // Goldilocks width 8.
    let gl_8 = default_goldilocks_poseidon_8();
    poseidon_scalar::<Goldilocks, _, 8>(c, &gl_8);
    poseidon_packed::<Goldilocks, _, 8>(c, &gl_8);

    // Goldilocks width 12.
    let gl_12 = default_goldilocks_poseidon_12();
    poseidon_scalar::<Goldilocks, _, 12>(c, &gl_12);
    poseidon_packed::<Goldilocks, _, 12>(c, &gl_12);

    // Mersenne31: generic implementation with random constants.
    // Uses Poseidon2 round number constants (Mersenne31 has no dedicated Poseidon1 module).
    let m31_16: PoseidonGeneric<Mersenne31, MdsMatrixMersenne31, 16, { MERSENNE31_S_BOX_DEGREE }> =
        Poseidon::new_from_rng(
            MERSENNE31_POSEIDON2_HALF_FULL_ROUNDS,
            MERSENNE31_POSEIDON2_PARTIAL_ROUNDS_16,
            &MdsMatrixMersenne31,
            &mut rng,
        );
    poseidon_scalar::<Mersenne31, _, 16>(c, &m31_16);

    let m31_32: PoseidonGeneric<Mersenne31, MdsMatrixMersenne31, 32, { MERSENNE31_S_BOX_DEGREE }> =
        Poseidon::new_from_rng(
            MERSENNE31_POSEIDON2_HALF_FULL_ROUNDS,
            MERSENNE31_POSEIDON2_PARTIAL_ROUNDS_24,
            &MdsMatrixMersenne31,
            &mut rng,
        );
    poseidon_scalar::<Mersenne31, _, 32>(c, &m31_32);
}

/// Benchmark using scalar field elements (no SIMD).
fn poseidon_scalar<F, Perm, const WIDTH: usize>(c: &mut Criterion, poseidon: &Perm)
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
fn poseidon_packed<F, Perm, const WIDTH: usize>(c: &mut Criterion, poseidon: &Perm)
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

criterion_group!(benches, bench_poseidon);
criterion_main!(benches);
