//! A/B isolation bench: the **generic** Poseidon2 permutation run over the packed Goldilocks
//! backend. The only variable between a NEON build and an SVE build is the packing type
//! (`PackedGoldilocksNeon` vs `PackedGoldilocksSve`) — both run the identical generic algorithm.
//! This separates the per-operation memory round-trip of the array-backed SVE packing from the
//! fused-vs-generic algorithm difference that dominates the standard `poseidon2` bench (where the
//! NEON build uses the bespoke fused kernel and the SVE build uses the generic one).
//!
//! Run the same bench in each build and compare the per-hash throughput (`Melem/s`):
//! ```text
//! # NEON (register-native uint64x2_t packing)
//! cargo bench -p p3-goldilocks --bench poseidon2_packing_ab
//! # SVE on Graviton3 (array-backed packing, one asm block per op)
//! RUSTFLAGS="-C target-cpu=neoverse-v1" cargo bench -p p3-goldilocks --bench poseidon2_packing_ab
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
use p3_goldilocks::{
    Goldilocks, Poseidon2ExternalLayerGoldilocks, Poseidon2InternalLayerGoldilocks,
};
use p3_poseidon2::Poseidon2;
use p3_symmetric::Permutation;
use p3_util::pretty_name;
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// The generic (portable) Poseidon2 for Goldilocks, named explicitly so it is the *same* type in
/// both the NEON and SVE builds. The `Poseidon2Goldilocks` alias would otherwise resolve to the
/// fused NEON kernel under NEON, which is what we are deliberately factoring out here. `D = 7` is
/// the Goldilocks S-box degree.
type GenericPoseidon2Goldilocks<const WIDTH: usize> = Poseidon2<
    Goldilocks,
    Poseidon2ExternalLayerGoldilocks<WIDTH>,
    Poseidon2InternalLayerGoldilocks,
    WIDTH,
    7,
>;

fn run<const WIDTH: usize, Perm>(c: &mut Criterion, perm: &Perm)
where
    Perm: Permutation<[<Goldilocks as Field>::Packing; WIDTH]>,
{
    type Packing = <Goldilocks as Field>::Packing;

    let input = [Packing::ZERO; WIDTH];

    // One `permute` processes `Packing::WIDTH` independent hashes; report per-hash throughput so the
    // NEON (2 lanes) and SVE (4 lanes) builds are directly comparable.
    let name = format!(
        "poseidon2_generic::<{}, {}>",
        pretty_name::<Packing>(),
        WIDTH
    );
    let mut g = c.benchmark_group("poseidon2_packing_ab");
    g.throughput(Throughput::Elements(Packing::WIDTH as u64));
    g.bench_with_input(BenchmarkId::new(name, WIDTH), &input, |b, &input| {
        b.iter(|| perm.permute(input))
    });
    g.finish();
}

fn bench(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(1);
    run::<8, _>(
        c,
        &GenericPoseidon2Goldilocks::<8>::new_from_rng_128(&mut rng),
    );
    run::<12, _>(
        c,
        &GenericPoseidon2Goldilocks::<12>::new_from_rng_128(&mut rng),
    );
    run::<16, _>(
        c,
        &GenericPoseidon2Goldilocks::<16>::new_from_rng_128(&mut rng),
    );
}

criterion_group!(benches, bench);
criterion_main!(benches);
