use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, MdsMatrixBabyBear, PoseidonBabyBear};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_goldilocks::{Goldilocks, MdsMatrixGoldilocks};
use p3_koala_bear::{KoalaBear, MdsMatrixKoalaBear, PoseidonKoalaBear};
use p3_mersenne_31::{MdsMatrixMersenne31, Mersenne31};
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
    let poseidon_bb_16 = PoseidonBabyBear::<16>::new_from_rng(4, 13, &mds_bb, &mut rng);
    poseidon_scalar::<BabyBear, _, 16>(c, &poseidon_bb_16);
    poseidon_packed::<BabyBear, _, 16>(c, &poseidon_bb_16);

    // BabyBear width 24.
    let poseidon_bb_24 = PoseidonBabyBear::<24>::new_from_rng(4, 21, &mds_bb, &mut rng);
    poseidon_scalar::<BabyBear, _, 24>(c, &poseidon_bb_24);
    poseidon_packed::<BabyBear, _, 24>(c, &poseidon_bb_24);

    let mds_kb: MdsMatrixKoalaBear = Default::default();

    // KoalaBear width 16.
    let poseidon_kb_16 = PoseidonKoalaBear::<16>::new_from_rng(4, 20, &mds_kb, &mut rng);
    poseidon_scalar::<KoalaBear, _, 16>(c, &poseidon_kb_16);
    poseidon_packed::<KoalaBear, _, 16>(c, &poseidon_kb_16);

    // KoalaBear width 24.
    let poseidon_kb_24 = PoseidonKoalaBear::<24>::new_from_rng(4, 23, &mds_kb, &mut rng);
    poseidon_scalar::<KoalaBear, _, 24>(c, &poseidon_kb_24);
    poseidon_packed::<KoalaBear, _, 24>(c, &poseidon_kb_24);

    // Goldilocks: generic implementation with random constants.
    let gl_8: PoseidonGeneric<Goldilocks, MdsMatrixGoldilocks, 8, 7> =
        Poseidon::new_from_rng(4, 22, &MdsMatrixGoldilocks, &mut rng);
    poseidon_scalar::<Goldilocks, _, 8>(c, &gl_8);

    let gl_12: PoseidonGeneric<Goldilocks, MdsMatrixGoldilocks, 12, 7> =
        Poseidon::new_from_rng(4, 22, &MdsMatrixGoldilocks, &mut rng);
    poseidon_scalar::<Goldilocks, _, 12>(c, &gl_12);

    let gl_16: PoseidonGeneric<Goldilocks, MdsMatrixGoldilocks, 16, 7> =
        Poseidon::new_from_rng(4, 22, &MdsMatrixGoldilocks, &mut rng);
    poseidon_scalar::<Goldilocks, _, 16>(c, &gl_16);

    // Mersenne31: generic implementation with random constants.
    let m31_16: PoseidonGeneric<Mersenne31, MdsMatrixMersenne31, 16, 5> =
        Poseidon::new_from_rng(4, 22, &MdsMatrixMersenne31, &mut rng);
    poseidon_scalar::<Mersenne31, _, 16>(c, &m31_16);

    let m31_32: PoseidonGeneric<Mersenne31, MdsMatrixMersenne31, 32, 5> =
        Poseidon::new_from_rng(4, 22, &MdsMatrixMersenne31, &mut rng);
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
