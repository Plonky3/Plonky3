//! Criterion benches for the ring-switching reduction.
//!
//! The pair is the production one, `GF(2^8)` into `GF(2^128)`, so `κ = 4` and a claim in `ℓ`
//! variables reduces through an `ℓ − 4`-round sumcheck. `ℓ` is the trace height the reduction
//! would be applied at.
//!
//! On x86-64 these numbers are meaningless without `-C target-feature=+pclmulqdq`: without it
//! `GF(2^128)` multiplication falls back to recursive tower multiplication. AArch64 has the
//! instruction in its baseline.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_binary_field::{
    BinaryChallenger, BinaryField8, BinaryField128, BitCoordinates, Ghash128, Poly64, Poly192,
    TowerLevel,
};
use p3_challenger::HashChallenger;
use p3_field::{BasedVectorSpace, ExtensionField, Field, PackedValue, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;
use p3_sumcheck::ring_switch::bits::BitRingSwitch;
use p3_sumcheck::ring_switch::{pack, prove_ring_switch, verify_ring_switch};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = BinaryField8;
type EF = BinaryField128;
type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

/// Variable counts benched. `ℓ' = ℓ − 4` sumcheck rounds each.
const SIZES: [usize; 3] = [20, 22, 24];

const fn challenger() -> Challenger {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// The base polynomial, its packing, the evaluation point and the true claim at that point.
fn fixture(ell: usize) -> (Poly<F>, Poly<EF>, Point<EF>, EF) {
    let mut rng = SmallRng::seed_from_u64(1);
    let t = Poly::<F>::rand(&mut rng, ell);
    let packed = pack::<F, EF>(t.clone());
    let r = Point::<EF>::rand(&mut rng, ell);
    let s = t.eval_base(&r);
    (t, packed, r, s)
}

fn bench_pack(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_switch/pack");
    for ell in SIZES {
        let t = Poly::<F>::rand(&mut SmallRng::seed_from_u64(1), ell);
        group.throughput(Throughput::Elements(1u64 << ell));
        group.bench_with_input(BenchmarkId::from_parameter(ell), &ell, |b, _| {
            b.iter(|| black_box(pack::<F, EF>(black_box(t.clone()))));
        });
    }
    group.finish();
}

fn bench_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_switch/prove");
    for ell in SIZES {
        let (_, packed, r, _) = fixture(ell);
        group.throughput(Throughput::Elements(1u64 << ell));
        group.bench_with_input(BenchmarkId::from_parameter(ell), &ell, |b, _| {
            b.iter(|| {
                black_box(prove_ring_switch::<F, EF, _>(
                    black_box(&packed),
                    black_box(&r),
                    &mut challenger(),
                ))
            });
        });
    }
    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_switch/verify");
    for ell in SIZES {
        let (_, packed, r, s) = fixture(ell);
        let (proof, _, _) = prove_ring_switch::<F, EF, _>(&packed, &r, &mut challenger());
        group.bench_with_input(BenchmarkId::from_parameter(ell), &ell, |b, _| {
            b.iter(|| {
                black_box(
                    verify_ring_switch::<F, EF, _>(
                        black_box(&proof),
                        black_box(&r),
                        black_box(s),
                        &mut challenger(),
                    )
                    .unwrap(),
                )
            });
        });
    }
    group.finish();
}

/// Packed prefix compression reaches the four-term mixed-dot kernel at widths 4/8.
/// Width 2 uses the existing eager fallback; both paths fold the same prefix variables.
fn bench_poly_split_eq(c: &mut Criterion) {
    let mut group = c.benchmark_group("poly64/split_eq_compress_prefix");
    let width = <Poly64 as Field>::Packing::WIDTH;
    assert_eq!(<Poly192 as BasedVectorSpace<Poly64>>::DIMENSION, 3);
    for (variables, eq_variables) in [(12usize, 6usize), (16, 8), (20, 10)] {
        // The packed suffix of the factored eq table holds 8, 16, or 32 scalars.
        // At widths 2/4/8 it is packed and has <= 128 entries; the output is packed too.
        let eq1_evals = 1usize << eq_variables.div_ceil(2);
        assert!(eq1_evals >= width && eq1_evals / width <= 128);
        assert!((1usize << (variables - eq_variables)) >= width);
        let mut rng = SmallRng::seed_from_u64(0x64E9);
        let poly = Poly::<Poly64>::rand(&mut rng, variables);
        let point = Point::<Poly192>::rand(&mut rng, eq_variables);
        let split = SplitEq::<Poly64, Poly192>::new_packed(&point, Poly192::ONE);
        let expected = SplitEq::<Poly64, Poly192>::new_unpacked(&point, Poly192::ONE)
            .compress_prefix(poly.as_view());
        assert_eq!(
            split
                .compress_prefix_to_packed(poly.as_view())
                .unpack::<Poly64, Poly192>(),
            expected,
        );
        group.bench_function(format!("{variables}/eq{eq_variables}"), |b| {
            b.iter(|| split.compress_prefix_to_packed(black_box(poly.as_view())));
        });
    }
    group.finish();
}

fn bench_bit_closing_for<F: TowerLevel, E: BitCoordinates + ExtensionField<F>>(
    c: &mut Criterion,
    label: &str,
) where
    rand::distr::StandardUniform: rand::distr::Distribution<E>,
{
    let mut group = c.benchmark_group(format!("bit_ring_switch/closing_weight/{label}"));
    for free in [12, 24] {
        for prefix in [0, 4] {
            let mut rng = SmallRng::seed_from_u64(0xE964);
            let absorbed = BitRingSwitch::<F, E>::ABSORBED;
            let mut coordinates: Vec<E> = (0..prefix)
                .map(|i| if i % 2 == 0 { E::ZERO } else { E::ONE })
                .collect();
            coordinates.extend((0..free + absorbed).map(|_| rng.random::<E>()));
            let point = Point::new(coordinates);
            let other = Point::<E>::rand(&mut rng, free);
            let batch_point = Point::<E>::rand(&mut rng, BitRingSwitch::<F, E>::BATCHED);
            for successor in [false, true] {
                let reduction = if successor {
                    BitRingSwitch::<F, E>::with_successor(&point, free + absorbed)
                } else {
                    BitRingSwitch::<F, E>::new(&point)
                }
                .unwrap();
                let batch = if reduction.sends_successor_tensors() {
                    reduction.batch_with_successor(&batch_point, rng.random())
                } else {
                    reduction.batch(&batch_point)
                }
                .unwrap();
                group.bench_function(format!("{free}/prefix{prefix}/successor{successor}"), |b| {
                    b.iter(|| batch.closing_weight(black_box(&other)).unwrap());
                });
            }
        }
    }
    group.finish();
}

fn bench_bit_closing(c: &mut Criterion) {
    bench_bit_closing_for::<BinaryField128, BinaryField128>(c, "tower128");
    bench_bit_closing_for::<Ghash128, Ghash128>(c, "ghash128");
    bench_bit_closing_for::<Poly64, Poly192>(c, "poly192");
}

fn bench_equality(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_switch/equality");
    for variables in [12, 24] {
        let mut rng = SmallRng::seed_from_u64(0xE965);
        let high = Point::<EF>::rand(&mut rng, variables);
        let other = Point::<EF>::rand(&mut rng, variables);
        group.bench_function(BenchmarkId::from_parameter(variables), |b| {
            b.iter(|| {
                p3_sumcheck::ring_switch::equality::equality_element::<F, EF>(
                    black_box(&high),
                    black_box(&other),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_pack,
    bench_prove,
    bench_verify,
    bench_poly_split_eq,
    bench_bit_closing,
    bench_equality
);
criterion_main!(benches);
