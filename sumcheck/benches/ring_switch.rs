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
use p3_binary_field::{BinaryChallenger, BinaryField8, BinaryField128};
use p3_challenger::HashChallenger;
use p3_keccak::Keccak256Hash;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::ring_switch::{pack, prove_ring_switch, verify_ring_switch};
use rand::SeedableRng;
use rand::rngs::SmallRng;

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

criterion_group!(benches, bench_pack, bench_prove, bench_verify);
criterion_main!(benches);
