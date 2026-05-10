//! Benchmarks for `Pcs::commit_quotient` on `TwoAdicFriPcs`.
//!
//! Mirrors the workload in `uni-stark`'s prover at the quotient-commit step:
//! a single `RowMajorMatrix<Val>` of width `Challenge::DIMENSION`, height
//! `N · D`, where `N` is the (extended) trace size and `D` is the number of
//! quotient chunks.
//!
//! ```bash
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-fri --bench commit_quotient
//! ```
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::{ExtensionMmcs, Pcs, PolynomialSpace};
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, TwoAdicField};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_util::pretty_name;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

// Common bench parameters (log_n, log_d, log_blowup).
//
// log_n: log2 of trace height N.
// log_d: log2 of constraint degree D = number of quotient chunks.
// log_blowup: log2 of FRI blowup B (the LDE expansion ratio for the chunks).
//
// We focus on the realistic `D < B` regime (constraint degree strictly less
// than FRI blowup). `D == B` is a degenerate case where each chunk's LDE has
// blowup ratio 1; `D > B` is forbidden by the chunk decomposition.
const PARAM_GRID: [(usize, usize, usize); 4] = [
    (16, 1, 2), // N=2^16, D=2, B=4
    (16, 1, 3), // N=2^16, D=2, B=8
    (18, 1, 2), // N=2^18, D=2, B=4
    (18, 1, 3), // N=2^18, D=2, B=8
];

// -------- BabyBear (quartic extension) --------

type BbVal = BabyBear;
type BbChallenge = BinomialExtensionField<BbVal, 4>;
type BbPerm = Poseidon2BabyBear<16>;
type BbHash = PaddingFreeSponge<BbPerm, 16, 8, 8>;
type BbCompress = TruncatedPermutation<BbPerm, 2, 8, 16>;
type BbValMmcs =
    MerkleTreeMmcs<<BbVal as Field>::Packing, <BbVal as Field>::Packing, BbHash, BbCompress, 2, 8>;
type BbChallengeMmcs = ExtensionMmcs<BbVal, BbChallenge, BbValMmcs>;
type BbDft = Radix2DitParallel<BbVal>;
type BbPcs = TwoAdicFriPcs<BbVal, BbDft, BbValMmcs, BbChallengeMmcs>;
type BbChallenger = DuplexChallenger<BbVal, BbPerm, 16, 8>;

fn bb_pcs(log_blowup: usize) -> BbPcs {
    let mut rng = SmallRng::seed_from_u64(0xB1B);
    let perm = BbPerm::new_from_rng_128(&mut rng);
    let hash = BbHash::new(perm.clone());
    let compress = BbCompress::new(perm);
    let val_mmcs = BbValMmcs::new(hash, compress, 0);
    let challenge_mmcs = BbChallengeMmcs::new(val_mmcs.clone());
    let mut params = FriParameters::new_testing(challenge_mmcs, 0);
    params.log_blowup = log_blowup;
    BbPcs::new(BbDft::default(), val_mmcs, params)
}

// -------- Goldilocks (quadratic extension) --------

type GlVal = Goldilocks;
type GlChallenge = BinomialExtensionField<GlVal, 2>;
type GlPerm = Poseidon2Goldilocks<8>;
type GlHash = PaddingFreeSponge<GlPerm, 8, 4, 4>;
type GlCompress = TruncatedPermutation<GlPerm, 2, 4, 8>;
type GlValMmcs =
    MerkleTreeMmcs<<GlVal as Field>::Packing, <GlVal as Field>::Packing, GlHash, GlCompress, 2, 4>;
type GlChallengeMmcs = ExtensionMmcs<GlVal, GlChallenge, GlValMmcs>;
type GlDft = Radix2DitParallel<GlVal>;
type GlPcs = TwoAdicFriPcs<GlVal, GlDft, GlValMmcs, GlChallengeMmcs>;
type GlChallenger = DuplexChallenger<GlVal, GlPerm, 8, 4>;

fn gl_pcs(log_blowup: usize) -> GlPcs {
    let mut rng = SmallRng::seed_from_u64(0xC0DE);
    let perm = GlPerm::new_from_rng_128(&mut rng);
    let hash = GlHash::new(perm.clone());
    let compress = GlCompress::new(perm);
    let val_mmcs = GlValMmcs::new(hash, compress, 0);
    let challenge_mmcs = GlChallengeMmcs::new(val_mmcs.clone());
    let mut params = FriParameters::new_testing(challenge_mmcs, 0);
    params.log_blowup = log_blowup;
    GlPcs::new(GlDft::default(), val_mmcs, params)
}

// -------- Generic bench driver --------

fn bench_commit_quotient_for<Val, Challenge, MyPcs, MyChallenger>(
    c: &mut Criterion,
    pcs_factory: impl Fn(usize) -> MyPcs,
    field_label: &str,
) where
    Val: Field + TwoAdicField,
    Challenge: p3_field::ExtensionField<Val>,
    MyPcs: Pcs<Challenge, MyChallenger>,
    StandardUniform: Distribution<Val>,
    <MyPcs as Pcs<Challenge, MyChallenger>>::Domain: PolynomialSpace<Val = Val>,
{
    let mut group = c.benchmark_group(format!("commit_quotient/{field_label}"));
    group.sample_size(10);

    for (log_n, log_d, log_blowup) in PARAM_GRID {
        let n = 1usize << log_n;
        let d = 1usize << log_d;
        let n_d = n * d;
        let dim = <Challenge as p3_field::BasedVectorSpace<Val>>::DIMENSION;

        let pcs = pcs_factory(log_blowup);
        let quotient_domain = pcs.natural_domain_for_degree(n_d);
        // The actual call site uses GENERATOR-shifted quotient domain; mirror that
        // so the bench reflects the real workload.
        let quotient_domain = quotient_domain.create_disjoint_domain(n_d);

        let mut rng = SmallRng::seed_from_u64(((log_n as u64) << 8) ^ (log_d as u64));
        let values: Vec<Val> = (0..n_d * dim).map(|_| rng.random()).collect();
        let q_flat = RowMajorMatrix::new(values, dim);

        let label = format!("logN={log_n}/D={d}/B={}", 1 << log_blowup);
        group.bench_function(BenchmarkId::from_parameter(label), |bench| {
            bench.iter_batched(
                || q_flat.clone(),
                |m| {
                    let (commitment, data) = pcs.commit_quotient(quotient_domain, m, d);
                    black_box((commitment, data))
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

fn bench_commit_quotient(c: &mut Criterion) {
    bench_commit_quotient_for::<BbVal, BbChallenge, BbPcs, BbChallenger>(
        c,
        bb_pcs,
        pretty_name::<BbVal>().as_str(),
    );
    bench_commit_quotient_for::<GlVal, GlChallenge, GlPcs, GlChallenger>(
        c,
        gl_pcs,
        pretty_name::<GlVal>().as_str(),
    );
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(15))
        .warm_up_time(std::time::Duration::from_secs(2));
    targets = bench_commit_quotient
}
criterion_main!(benches);
