use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, Pcs};
use p3_dft::Radix2DitParallel;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Dft = Radix2DitParallel<Val>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

/// Width of the matrix opened at two points, like a trace opened at `zeta` and `zeta * g`.
const TRACE_WIDTH: usize = 64;
/// Width of each matrix opened at one point, like a flattened quotient chunk.
const QUOTIENT_WIDTH: usize = 4;
/// Number of quotient chunks.
const NUM_QUOTIENT_CHUNKS: usize = 2;

/// Opening a trace-shaped batch: one wide matrix at two points and a few narrow ones at one.
fn bench_open(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(0);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    // Grinding is disabled so the timing reflects the prover's arithmetic only.
    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        max_log_arity: 3,
        num_queries: 100,
        batch_proof_of_work_bits: 0,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
        mmcs: challenge_mmcs,
    };
    let pcs = MyPcs::new(Dft::default(), val_mmcs, fri_params);

    let mut group = c.benchmark_group("pcs_open::<BabyBear, quartic>");
    group.sample_size(10);

    for log_degree in [14, 16, 18, 20] {
        let degree = 1 << log_degree;
        let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, degree);

        let trace = RowMajorMatrix::<Val>::rand(&mut rng, degree, TRACE_WIDTH);
        let (_, trace_data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain, trace)]).unwrap();

        let quotients = (0..NUM_QUOTIENT_CHUNKS)
            .map(|_| {
                (
                    domain,
                    RowMajorMatrix::<Val>::rand(&mut rng, degree, QUOTIENT_WIDTH),
                )
            })
            .collect::<Vec<_>>();
        let (_, quotient_data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, quotients).unwrap();

        let mut challenger = Challenger::new(perm.clone());
        let zeta: Challenge = challenger.sample_algebra_element();
        let zeta_next: Challenge = challenger.sample_algebra_element();

        group.bench_function(BenchmarkId::from_parameter(log_degree), |b| {
            b.iter_batched(
                || challenger.clone(),
                |mut challenger| {
                    let requests = vec![
                        (&trace_data, vec![vec![zeta, zeta_next]]).into(),
                        (&quotient_data, vec![vec![zeta]; NUM_QUOTIENT_CHUNKS]).into(),
                    ];
                    black_box(
                        <MyPcs as Pcs<Challenge, Challenger>>::open(
                            &pcs,
                            requests,
                            &mut challenger,
                        )
                        .unwrap(),
                    )
                },
                BatchSize::LargeInput,
            );
        });
    }
}

criterion_group!(benches, bench_open);
criterion_main!(benches);
