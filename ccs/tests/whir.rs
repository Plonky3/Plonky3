//! End-to-end CCS proofs over a WHIR-committed witness.

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_ccs::{
    BlockDiagonal, Ccs, CommittedProof, CommittedVerificationError, Constraint, Operator,
    PublicRegion, RowSelector, SparseMatrix, Term, TraceColumn, VerificationError, prove_committed,
    verify_committed,
};
use p3_challenger::{CanSample, DuplexChallenger};
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sumcheck::layout::{Layout, PrefixProver, Table};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig, WhirProver};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Base field.
type F = BabyBear;
/// Challenge field.
type EF = BinomialExtensionField<F, 4>;
/// Sponge permutation.
type Perm = Poseidon2BabyBear<16>;
/// Fiat-Shamir sponge.
type Challenger = DuplexChallenger<F, Perm, 16, 8>;
/// Merkle commitment over packed base elements.
type Mmcs = MerkleTreeMmcs<
    <F as Field>::Packing,
    <F as Field>::Packing,
    PaddingFreeSponge<Perm, 16, 8, 8>,
    TruncatedPermutation<Perm, 2, 8, 16>,
    2,
    8,
>;
/// Witness layout.
type L = PrefixProver<F, EF>;
/// WHIR over that layout.
type Pcs = WhirProver<EF, F, Radix2DFTSmallBatch<F>, Mmcs, Challenger, L>;

/// Rejection of a full proof.
type Rejection = CommittedVerificationError<p3_whir::VerifierError>;

/// WHIR folds this many variables per round.
const FOLDING: usize = 4;

fn perm() -> Perm {
    Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1))
}

fn challenger() -> Challenger {
    Challenger::new(perm())
}

fn pcs(num_variables: usize) -> Pcs {
    let mmcs = Mmcs::new(
        PaddingFreeSponge::new(perm()),
        TruncatedPermutation::new(perm()),
        0,
    );
    let config = WhirConfig::new(
        num_variables,
        ProtocolParameters {
            security_level: 100,
            pow_bits: 0,
            round_log_inv_rates: vec![],
            folding_factor: FoldingFactor::Constant(FOLDING),
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        },
    )
    .unwrap();
    Pcs::new(config, Radix2DFTSmallBatch::default(), mmcs)
}

/// Fibonacci on two columns over 2^log_height rows, and its trace.
fn fibonacci(log_height: usize) -> (Ccs<F>, Vec<F>) {
    let cell = |column, offset| Operator::from(TraceColumn::new(log_height, 1, column, offset));
    let ccs = Ccs::new(
        vec![cell(0, 0), cell(1, 0), cell(0, 1), cell(1, 1)],
        vec![
            Constraint::new(vec![Term::new(F::ONE, [2]), Term::new(F::NEG_ONE, [1])]),
            Constraint::new(vec![
                Term::new(F::ONE, [3]),
                Term::new(F::NEG_ONE, [0]),
                Term::new(F::NEG_ONE, [1]),
            ]),
        ]
        .into_iter()
        .map(|k| k.on(RowSelector::Transition))
        .collect(),
    )
    .unwrap();
    let height = 1 << log_height;
    let (mut a, mut b) = (vec![F::ONE], vec![F::ONE]);
    for row in 1..height {
        a.push(b[row - 1]);
        b.push(a[row - 1] + b[row - 1]);
    }
    a.extend(b);
    (ccs, a)
}

/// The first four cells of column a, (1, 1, 2, 3), as public cells.
fn fibonacci_start() -> Vec<PublicRegion<F>> {
    vec![PublicRegion::new(
        0,
        0,
        [1, 1, 2, 3].map(F::from_u32).to_vec(),
    )]
}

/// Commit the witness and prove; returns the commitment, the proof and the prover's final sponge.
fn prove(
    ccs: &Ccs<F>,
    z: &[F],
    public: &[PublicRegion<F>],
) -> (
    <Pcs as MultilinearPcs<EF, Challenger>>::Commitment,
    CommittedProof<F, EF, <Pcs as MultilinearPcs<EF, Challenger>>::Proof>,
    Challenger,
) {
    let pcs = pcs(ccs.log_cols());
    let witness = L::new_witness(
        vec![Table::new(RowMajorMatrix::new(z.to_vec(), z.len()))],
        FOLDING,
    );
    let mut ch = challenger();
    let (commitment, data) = pcs.commit(witness, &mut ch).unwrap();
    let proof = prove_committed::<F, EF, _, _, _>(ccs, &pcs, z, public, data, 0, &mut ch).unwrap();
    (commitment, proof, ch)
}

fn verify(
    ccs: &Ccs<F>,
    commitment: &<Pcs as MultilinearPcs<EF, Challenger>>::Commitment,
    public: &[PublicRegion<F>],
    proof: &CommittedProof<F, EF, <Pcs as MultilinearPcs<EF, Challenger>>::Proof>,
) -> Result<Challenger, Rejection> {
    let mut ch = challenger();
    let pcs = pcs(ccs.log_cols());
    verify_committed(ccs, &pcs, commitment, public, proof, 0, &mut ch)?;
    Ok(ch)
}

/// R1CS of x * y = 12 and x + y = 7 over (one, x, y, 0, ...), with 2^8 cells.
fn product_and_sum() -> Ccs<F> {
    let a = SparseMatrix::new(1, 8, vec![(0, 1, F::ONE), (1, 1, F::ONE), (1, 2, F::ONE)]);
    let b = SparseMatrix::new(1, 8, vec![(0, 2, F::ONE), (1, 0, F::ONE)]);
    let c = SparseMatrix::new(1, 8, vec![(0, 0, F::from_u32(12)), (1, 0, F::from_u32(7))]);
    Ccs::new(
        vec![a.into(), b.into(), c.into()],
        vec![Constraint::new(vec![
            Term::new(F::ONE, [0, 1]),
            Term::new(F::NEG_ONE, [2]),
        ])],
    )
    .unwrap()
}

/// 64 copies of that R1CS, each over (one, x, y, 0), and its strided constant cells.
fn copied_product_and_sum() -> (Ccs<F>, PublicRegion<F>) {
    let block = |entries| BlockDiagonal::new(6, SparseMatrix::new(1, 2, entries)).into();
    let a = block(vec![(0, 1, F::ONE), (1, 1, F::ONE), (1, 2, F::ONE)]);
    let b = block(vec![(0, 2, F::ONE), (1, 0, F::ONE)]);
    let c = block(vec![(0, 0, F::from_u32(12)), (1, 0, F::from_u32(7))]);
    let ccs = Ccs::new(
        vec![a, b, c],
        vec![Constraint::new(vec![
            Term::new(F::ONE, [0, 1]),
            Term::new(F::NEG_ONE, [2]),
        ])],
    )
    .unwrap();
    (ccs, PublicRegion::new(0, 2, vec![F::ONE; 64]))
}

/// The witness (one, x, y) padded with zeros.
fn r1cs_witness(cells: [u32; 3]) -> Vec<F> {
    let mut z = vec![F::ZERO; 256];
    for (cell, value) in z.iter_mut().zip(cells) {
        *cell = F::from_u32(value);
    }
    z
}

#[test]
fn a_committed_fibonacci_trace_proves_and_verifies() {
    // Fixture state: 2^8 rows, witness of 2^9 entries committed by WHIR.
    let (ccs, z) = fibonacci(8);
    let public = fibonacci_start();
    let (commitment, proof, mut prover_ch) = prove(&ccs, &z, &public);
    let mut verifier_ch =
        verify(&ccs, &commitment, &public, &proof).expect("an honest proof verifies");

    // Both sides end on the same sponge state.
    assert_eq!(
        CanSample::<F>::sample(&mut prover_ch),
        CanSample::<F>::sample(&mut verifier_ch)
    );
}

#[test]
fn a_wrong_committed_trace_is_rejected() {
    // Mutation: one cell of column b, so two transitions break.
    let (ccs, mut z) = fibonacci(8);
    z[256 + 40] += F::ONE;
    let (commitment, proof, _) = prove(&ccs, &z, &fibonacci_start());
    assert!(matches!(
        verify(&ccs, &commitment, &fibonacci_start(), &proof),
        Err(Rejection::Reduction(VerificationError::ZerocheckMismatch))
    ));
}

#[test]
fn a_witness_value_that_does_not_open_is_rejected() {
    // Mutation: the reduction's closing value moves, so it no longer matches the commitment.
    //
    // The reduction rejects it first, since the value also closes the linear sumcheck.
    let (ccs, z) = fibonacci(8);
    let (commitment, mut proof, _) = prove(&ccs, &z, &[]);
    proof.reduction.witness_eval += EF::ONE;
    assert!(matches!(
        verify(&ccs, &commitment, &[], &proof),
        Err(Rejection::Reduction(VerificationError::LinearMismatch))
    ));
}

#[test]
fn a_proof_against_another_commitment_is_rejected() {
    // Mutation: verify against the commitment of a different, also valid, trace.
    //
    // Fixture state: the second trace starts from (2, 3) instead of (1, 1).
    let (ccs, z) = fibonacci(8);
    let (_, proof, _) = prove(&ccs, &z, &[]);
    let mut other = vec![F::TWO];
    let mut b = vec![F::from_u32(3)];
    for row in 1..256 {
        other.push(b[row - 1]);
        b.push(other[row - 1] + b[row - 1]);
    }
    other.extend(b);
    let (other_commitment, _, _) = prove(&ccs, &other, &[]);
    assert!(verify(&ccs, &other_commitment, &[], &proof).is_err());
}

#[test]
fn a_tampered_opened_value_is_rejected() {
    // Mutation: the value inside the opening proof moves off the committed one.
    let (ccs, z) = fibonacci(8);
    let public = fibonacci_start();
    let (commitment, honest, _) = prove(&ccs, &z, &public);

    // Mutation: the reduction's value, then the public region's value.
    for batch in 0..2 {
        let mut proof = honest.clone();
        let evals = &proof.opening.evals[batch];
        let mut current = evals.current().to_vec();
        current[0] += EF::ONE;
        proof.opening.evals[batch] = p3_sumcheck::OpeningBatch::new(current, evals.next().to_vec());
        assert!(
            verify(&ccs, &commitment, &public, &proof).is_err(),
            "batch {batch}"
        );
    }
}

#[test]
fn an_r1cs_with_a_public_constant_proves_and_verifies() {
    // Fixture state: (one, x, y) = (1, 3, 4), the constant pinned as a one-cell public region.
    let ccs = product_and_sum();
    let z = r1cs_witness([1, 3, 4]);
    let one = [PublicRegion::new(0, 0, vec![F::ONE])];
    assert_eq!(ccs.check(&z), Ok(()));
    let (commitment, proof, _) = prove(&ccs, &z, &one);
    verify(&ccs, &commitment, &one, &proof).expect("an honest proof verifies");
}

#[test]
fn a_zero_constant_no_longer_satisfies_the_r1cs() {
    // Mutation: (one, x, y) = (0, 0, 9) satisfies every homogeneous row, though 0 * 9 != 12.
    //
    // The public constant catches it: the committed cell is 0, not 1.
    let ccs = product_and_sum();
    let bogus = r1cs_witness([0, 0, 9]);
    let one = [PublicRegion::new(0, 0, vec![F::ONE])];
    assert_eq!(ccs.check(&bogus), Ok(()));
    let (commitment, proof, _) = prove(&ccs, &bogus, &one);
    assert!(matches!(
        verify(&ccs, &commitment, &one, &proof),
        Err(Rejection::PublicMismatch { region: 0 })
    ));
}

#[test]
fn a_zero_constant_in_one_copy_is_caught_by_the_strided_constant() {
    // Fixture state: 64 copies of (1, 3, 4, 0), every constant pinned by one strided region.
    let (ccs, ones) = copied_product_and_sum();
    let honest: Vec<F> = (0..64)
        .flat_map(|_| [1, 3, 4, 0])
        .map(F::from_u32)
        .collect();
    assert!(ones.matches(&honest));
    let (commitment, proof, _) = prove(&ccs, &honest, core::slice::from_ref(&ones));
    verify(&ccs, &commitment, core::slice::from_ref(&ones), &proof)
        .expect("an honest proof verifies");

    // Mutation: copy 37 becomes (0, 0, 9, 0), which every homogeneous row accepts.
    let mut bogus = honest;
    bogus[4 * 37..4 * 38].copy_from_slice(&[0, 0, 9, 0].map(F::from_u32));
    assert_eq!(ccs.check(&bogus), Ok(()));
    assert!(!ones.matches(&bogus));
    let (commitment, proof, _) = prove(&ccs, &bogus, core::slice::from_ref(&ones));
    assert!(matches!(
        verify(&ccs, &commitment, core::slice::from_ref(&ones), &proof),
        Err(Rejection::PublicMismatch { region: 0 })
    ));
}

#[test]
fn a_proof_does_not_transfer_to_other_public_values() {
    // Mutation: the verifier reads (1, 1, 2, 4) where the prover bound (1, 1, 2, 3).
    let (ccs, z) = fibonacci(8);
    let (commitment, proof, _) = prove(&ccs, &z, &fibonacci_start());
    let other = [PublicRegion::new(
        0,
        0,
        [1, 1, 2, 4].map(F::from_u32).to_vec(),
    )];
    assert!(verify(&ccs, &commitment, &other, &proof).is_err());
}

#[test]
fn a_public_region_outside_the_witness_is_rejected() {
    // Mutation: eight cells from 512 start past the 2^9 cells.
    let (ccs, z) = fibonacci(8);
    let (commitment, proof, _) = prove(&ccs, &z, &[]);
    let outside = [PublicRegion::new(512, 0, vec![F::ONE; 8])];
    assert!(matches!(
        verify(&ccs, &commitment, &outside, &proof),
        Err(Rejection::PublicOutOfRange {
            region: 0,
            log_cols: 9
        })
    ));
}
