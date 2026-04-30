//! Adapter implementing the multilinear PCS trait for the WHIR protocol.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, MultilinearOpenedValues, MultilinearPcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use super::committer::reader::CommitmentReader;
use super::committer::writer::CommitmentWriter;
use super::proof::WhirProof;
use super::prover::WhirProver;
use super::verifier::WhirVerifier;
use super::verifier::errors::VerifierError;
use crate::constraints::statement::EqStatement;
use crate::constraints::statement::initial::InitialStatement;
use crate::fiat_shamir::domain_separator::DomainSeparator;
use crate::parameters::{ProtocolParameters, SumcheckStrategy, WhirConfig};

/// WHIR-based multilinear polynomial commitment scheme.
///
/// Wraps the full WHIR IOP of proximity (Construction 5.1 in the paper)
/// behind a generic PCS trait.
///
/// The DFT backend and Fiat-Shamir domain separator are managed internally.
///
/// The const generic `DIGEST_ELEMS` must match the Merkle tree digest width
/// used by the underlying commitment scheme.
#[derive(Debug)]
pub struct WhirPcs<EF, F, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Full protocol configuration derived from the parameters.
    config: WhirConfig<EF, F, MT, Challenger>,
    /// Raw parameters kept around to allocate proof structures.
    protocol_params: ProtocolParameters<MT>,
    /// DFT backend for Reed-Solomon encoding (hidden from the trait surface).
    dft: Dft,
    /// Sumcheck proving strategy: classic constraint batching or split-value optimization.
    sumcheck_strategy: SumcheckStrategy,
}

/// Prover-side data produced by commit, consumed by open.
pub struct WhirProverData<F, EF, MT, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Merkle tree produced during commitment; used to open query positions.
    merkle_data: MT::ProverData<DenseMatrix<F>>,
    /// Statement carrying the polynomial and all equality constraints
    /// (both user-supplied evaluation claims and OOD challenge points).
    statement: InitialStatement<F, EF>,
    /// Proof structure with the initial commitment and OOD answers filled in.
    /// The proving phase fills the remaining round data.
    proof: WhirProof<F, EF, MT>,
    /// Evaluation values computed during commit, indexed per polynomial.
    opened_values: MultilinearOpenedValues<EF>,
}

impl<EF, F, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Create a new WHIR PCS for multilinear polynomials in `num_variables` variables.
    ///
    /// # Arguments
    ///
    /// - `num_variables`: dimension m (the polynomial has 2^m evaluations).
    /// - `protocol_params`: security level, folding factor, rate, Merkle tree, etc.
    /// - `dft`: the DFT backend used for Reed-Solomon encoding.
    /// - `sumcheck_strategy`: classic or split-value optimization.
    pub fn new(
        num_variables: usize,
        protocol_params: ProtocolParameters<MT>,
        dft: Dft,
        sumcheck_strategy: SumcheckStrategy,
    ) -> Self {
        // Derive the full round-by-round configuration from the raw parameters.
        let config = WhirConfig::new(num_variables, protocol_params.clone());
        Self {
            config,
            protocol_params,
            dft,
            sumcheck_strategy,
        }
    }

    /// Build the Fiat-Shamir domain separator for this protocol instance.
    ///
    /// The domain separator encodes all public protocol parameters into
    /// the transcript so the verifier's challenges are bound to this
    /// specific configuration (see Construction 5.1, step 1).
    fn build_domain_separator(&self) -> DomainSeparator<EF, F>
    where
        EF: TwoAdicField,
    {
        // Start with an empty pattern.
        let mut ds = DomainSeparator::new(vec![]);
        // Encode the public parameters (num_variables, security, rate, etc.).
        ds.commit_statement::<MT, Challenger, DIGEST_ELEMS>(&self.config);
        // Encode the full proof structure (round counts, query counts, etc.).
        ds.add_whir_proof::<MT, Challenger, DIGEST_ELEMS>(&self.config);
        ds
    }
}

impl<EF, F, MT, Challenger, Dft, const DIGEST_ELEMS: usize> MultilinearPcs<EF, Challenger>
    for WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<MT::Commitment>
        + CanSampleUniformBits<F>
        + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    type Val = F;
    type Commitment = MT::Commitment;
    type ProverData = WhirProverData<F, EF, MT, DIGEST_ELEMS>;
    type Proof = WhirProof<F, EF, MT>;
    type Error = VerifierError;

    fn num_vars(&self) -> usize {
        self.config.num_variables
    }

    fn commit(
        &self,
        evaluations: RowMajorMatrix<Self::Val>,
        opening_points: &[Vec<Point<EF>>],
        challenger: &mut Challenger,
    ) -> (Self::Commitment, Self::ProverData) {
        // Validate dimensions: single polynomial with 2^m evaluations.
        assert_eq!(
            evaluations.width(),
            1,
            "WHIR currently supports committing a single polynomial"
        );
        assert_eq!(
            evaluations.height(),
            1 << self.config.num_variables,
            "evaluation vector length must be 2^num_variables"
        );
        assert_eq!(
            opening_points.len(),
            1,
            "WHIR currently supports opening a single polynomial"
        );

        // Wrap the raw evaluation vector as a multilinear polynomial f: {0,1}^m -> F.
        let poly = Poly::new(evaluations.values);

        // Build the initial statement and register evaluation claims.
        // Each claim f(z_i) = v_i becomes an equality constraint with weight
        // polynomial w(Z, X) = Z * eq(z_i, X), so that:
        //   sum_{b in {0,1}^m} w(f(b), b) = f(z_i)
        // This is the mechanism described in Section 1.1 of the WHIR paper.
        let mut statement = self.config.initial_statement(poly, self.sumcheck_strategy);
        let mut values = Vec::with_capacity(opening_points[0].len());
        for point in &opening_points[0] {
            // Evaluate the polynomial at this point and record the constraint.
            let eval = statement.evaluate(point);
            values.push(eval);
        }

        // Absorb the protocol configuration into the Fiat-Shamir transcript.
        let ds = self.build_domain_separator();
        ds.observe_domain_separator(challenger);

        // Allocate the proof structure with pre-sized vectors for each round.
        let mut proof =
            WhirProof::from_protocol_parameters(&self.protocol_params, self.config.num_variables);

        // Run the commitment phase:
        //   1. Transpose and zero-pad the evaluation table.
        //   2. Apply DFT to produce the Reed-Solomon codeword.
        //   3. Build a Merkle tree over the codeword rows.
        //   4. Sample OOD challenge points from the transcript (Section 2.1.3, step 3).
        //   5. Evaluate the polynomial at those OOD points and observe the answers.
        let committer = CommitmentWriter::new(&self.config);
        let merkle_data = committer
            .commit(&self.dft, &mut proof, challenger, &mut statement)
            .expect("commitment phase failed");

        // The Merkle root is now stored in the proof; extract it as the public commitment.
        let commitment = proof
            .initial_commitment
            .clone()
            .expect("commitment should be set after commit phase");

        // Bundle everything the prover needs for the opening phase.
        let prover_data = WhirProverData {
            merkle_data,
            statement,
            proof,
            opened_values: vec![values],
        };

        (commitment, prover_data)
    }

    fn open(
        &self,
        mut prover_data: Self::ProverData,
        challenger: &mut Challenger,
    ) -> (MultilinearOpenedValues<EF>, Self::Proof) {
        // Execute the multi-round WHIR proving protocol (Construction 5.1):
        //   For each round i = 0..M-1:
        //     1. Run k_i sumcheck rounds to reduce the constraint claim.
        //     2. Fold the polynomial: f_{i+1}(X) = f_i(alpha, X).
        //     3. Commit the folded codeword via a Merkle tree.
        //     4. Sample OOD points and verify consistency.
        //     5. Perform proof-of-work grinding to bind the transcript.
        //     6. Generate STIR query positions and open Merkle paths.
        //   Final round: send the polynomial coefficients in the clear.
        let prover = WhirProver(&self.config);
        prover
            .prove(
                &self.dft,
                &mut prover_data.proof,
                challenger,
                &prover_data.statement,
                prover_data.merkle_data,
            )
            .expect("proving phase failed");

        (prover_data.opened_values, prover_data.proof)
    }

    fn verify(
        &self,
        _commitment: &Self::Commitment,
        opening_claims: &[Vec<(Point<EF>, EF)>],
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        assert_eq!(
            opening_claims.len(),
            1,
            "WHIR currently supports verifying a single polynomial"
        );

        // Re-derive the same domain separator that the prover used, so
        // the verifier's transcript state is identical.
        let ds: DomainSeparator<EF, F> = self.build_domain_separator();
        ds.observe_domain_separator(challenger);

        // Parse the Merkle root and OOD answers from the proof, replaying
        // the same transcript interactions the prover performed during commit.
        let commitment_reader = CommitmentReader::new(&self.config);
        let parsed_commitment =
            commitment_reader.parse_commitment::<F, DIGEST_ELEMS>(proof, challenger);

        // Reconstruct the equality statement from the opening claims.
        // Each claim (z_i, v_i) becomes a constraint:
        //   sum_{b in {0,1}^m} f(b) * eq(z_i, b) = v_i
        let mut eq_statement = EqStatement::initialize(self.config.num_variables);
        for (point, value) in &opening_claims[0] {
            eq_statement.add_evaluated_constraint(point.clone(), *value);
        }

        // Run the full verification:
        //   1. Combine equality constraints with OOD constraints.
        //   2. Verify each sumcheck round.
        //   3. Check STIR query openings against Merkle proofs.
        //   4. Verify the final polynomial evaluation.
        let verifier = WhirVerifier::new(&self.config);
        verifier.verify(proof, challenger, &parsed_commitment, eq_statement)?;

        Ok(())
    }
}
