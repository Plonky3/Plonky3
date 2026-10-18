//! Focused no-PoW checks and benchmarks for ZK-WHIR mask optimizations.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use p3_challenger::DuplexChallenger;
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption};
use p3_whir::pcs::zk::{HidingWhirPcs, ZkParameters, ZkWhirConfig};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::{SmallRng, StdRng};
use rand::{RngExt, SeedableRng};

type F = KoalaBear;
type Quartic = BinomialExtensionField<F, 4>;
type Octic = BinomialExtensionField<F, 8>;
type Poseidon16 = Poseidon2KoalaBear<16>;
type Poseidon24 = Poseidon2KoalaBear<24>;
type Hash = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
type Compress = TruncatedPermutation<Poseidon16, 2, 8, 16>;
type Challenger = DuplexChallenger<F, Poseidon16, 16, 8>;
type PackedF = <F as Field>::Packing;
type Mmcs = MerkleTreeMmcs<PackedF, PackedF, Hash, Compress, 2, 8>;
type Dft = Radix2DFTSmallBatch<F>;

fn challenger() -> Challenger {
    Challenger::new(Poseidon16::new_from_rng_128(&mut SmallRng::seed_from_u64(
        2,
    )))
}

fn lifecycle<EF>(c: &mut Criterion, name: &str, n: usize, security_level: usize)
where
    EF: ExtensionField<F> + TwoAdicField,
    StandardUniform: Distribution<EF>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let mmcs = Mmcs::new(
        Hash::new(Poseidon24::new_from_rng_128(&mut rng)),
        Compress::new(Poseidon16::new_from_rng_128(&mut rng)),
        0,
    );
    let config = ZkWhirConfig::<EF, F, Challenger>::new(
        n,
        ProtocolParameters {
            security_level,
            pow_bits: 0,
            // The old octic rate 4 fixture rejects at n=18: randomness exceeds the next oracle's slack.
            //
            // Rate 8 fits the derived budget.
            round_log_inv_rates: vec![8],
            folding_factor: FoldingFactor::ConstantFromSecondRound(8, 6),
            soundness_type: SecurityAssumption::JohnsonBound,
            starting_log_inv_rate: 1,
        },
        ZkParameters {
            ell_zk: 3,
            mask_log_inv_rate: 3,
        },
    )
    .unwrap();
    let pcs = HidingWhirPcs::new(config, Dft::default(), mmcs, StdRng::seed_from_u64(4));
    let mut rng = SmallRng::seed_from_u64(3);
    let witness = Poly::<F>::rand(&mut rng, n);
    let points = vec![Point::<EF>::rand(&mut rng, n)];
    let mut transcript = challenger();
    let (commitment, data) = pcs.commit(witness.clone(), &mut transcript).unwrap();
    let proof = pcs.open(data, points.clone(), &mut transcript).unwrap();
    pcs.verify(&commitment, &proof, &mut challenger(), points.clone())
        .unwrap();

    let mut group = c.benchmark_group(format!(
        "zk_masks/{name}_n{n}_security{security_level}_no_pow"
    ));
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    group.bench_function("commit", |b| {
        b.iter_batched(
            || (witness.clone(), challenger()),
            |(witness, mut ch)| pcs.commit(witness, &mut ch).unwrap(),
            BatchSize::PerIteration,
        );
    });
    group.bench_function("open", |b| {
        b.iter_batched(
            || {
                let mut ch = challenger();
                let (_, data) = pcs.commit(witness.clone(), &mut ch).unwrap();
                (ch, data)
            },
            |(mut ch, data)| pcs.open(data, points.clone(), &mut ch).unwrap(),
            BatchSize::PerIteration,
        );
    });
    group.bench_function("verify", |b| {
        b.iter_batched(
            challenger,
            |mut ch| {
                pcs.verify(&commitment, &proof, &mut ch, points.clone())
                    .unwrap();
            },
            BatchSize::PerIteration,
        );
    });
    group.bench_function("round_trip", |b| {
        b.iter_batched(
            || (witness.clone(), challenger(), challenger()),
            |(witness, mut prover, mut verifier)| {
                let (commitment, data) = pcs.commit(witness, &mut prover).unwrap();
                let proof = pcs.open(data, points.clone(), &mut prover).unwrap();
                pcs.verify(&commitment, &proof, &mut verifier, points.clone())
                    .unwrap();
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

fn bench_lifecycle(c: &mut Criterion) {
    lifecycle::<Quartic>(c, "quartic", 18, 80);
    lifecycle::<Octic>(c, "octic", 18, 128);
}

fn horner<F, EF>(point: F, coefficients: &[EF]) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    coefficients
        .iter()
        .rev()
        .fold(EF::ZERO, |acc, &coefficient| acc * point + coefficient)
}

fn bench_mask_verify(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(19);
    let coefficients: Vec<Octic> = (0..95).map(|_| rng.random()).collect();
    let generator = Octic::two_adic_generator(10);
    let base_generator = <Octic as ExtensionField<F>>::as_base(&generator).unwrap();
    let positions = [1, 21, 97, 299, 567, 999];
    for &position in &positions {
        assert_eq!(
            horner(generator.exp_u64(position), &coefficients),
            horner(base_generator.exp_u64(position), &coefficients)
        );
    }
    let mut group = c.benchmark_group("zk_masks/mask_verify_octic_95x8x6");
    group.bench_function("extension_points", |b| {
        b.iter(|| {
            for &position in black_box(&positions) {
                let point = generator.exp_u64(position);
                for _ in 0..8 {
                    let _ = black_box(horner(point, black_box(&coefficients)));
                }
            }
        });
    });
    group.bench_function("base_points", |b| {
        b.iter(|| {
            for &position in black_box(&positions) {
                let point = base_generator.exp_u64(position);
                for _ in 0..8 {
                    let _ = black_box(horner(point, black_box(&coefficients)));
                }
            }
        });
    });
    group.finish();
}

criterion_group!(benches, bench_mask_verify, bench_lifecycle);
criterion_main!(benches);
