//! Adapter implementing the multilinear PCS trait for the WHIR protocol.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs, MultilinearOpenedValues, MultilinearPcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::committer::reader::{CommitmentReader, ParsedCommitment};
use super::committer::writer::CommitmentWriter;
use super::proof::WhirProof;
use super::prover::{WhirBatchedInitialMerkleProverData, WhirProver};
use super::verifier::errors::VerifierError;
use super::verifier::{WhirBatchedInitialVerifierOracle, WhirVerifier};
use crate::constraints::statement::initial::InitialStatement;
use crate::constraints::statement::{
    EqStatement, LinearSigmaOpeningClaim, LinearSigmaReductionError, LinearSigmaReductionProof,
    LinearSigmaStatement,
};
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

/// Prover-side data for a WHIR commitment whose opening points are not known
/// when the commitment is created.
///
/// This keeps the scalar WHIR commitment phase separate from the later
/// constrained-RS proving phase. Accumulation protocols use this shape when
/// Fiat-Shamir challenges determine openings after an oracle has already been
/// bound.
#[derive(Clone)]
pub struct WhirDeferredProverData<F, EF, MT, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Merkle tree produced during commitment; reused by the opening proof.
    merkle_data: MT::ProverData<DenseMatrix<F>>,
    /// Statement carrying the committed polynomial and OOD constraints.
    statement: InitialStatement<F, EF>,
    /// Proof structure with the initial commitment and OOD answers filled in.
    proof: WhirProof<F, EF, MT>,
}

/// Prover-side data for a deferred WHIR commitment to an extension-field
/// initial oracle.
pub struct WhirExtensionDeferredProverData<F, EF, MT, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Merkle tree produced during commitment via `ExtensionMmcs`.
    merkle_data: <ExtensionMmcs<F, EF, MT> as Mmcs<EF>>::ProverData<DenseMatrix<EF>>,
    /// Extension-field polynomial committed in the initial WHIR round.
    poly: Poly<EF>,
    /// OOD constraints sampled and bound during commitment.
    ood_statement: EqStatement<EF>,
    /// Proof structure with the initial commitment and OOD answers filled in.
    proof: WhirProof<F, EF, MT>,
}

/// Prover-side data for one shared base-field root that commits several
/// initial polynomials as an MMCS matrix batch.
pub struct WhirSharedBaseDeferredProverData<F, EF, MT, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Merkle tree produced by committing all initial matrices under one root.
    merkle_data: Arc<MT::ProverData<DenseMatrix<F>>>,
    /// Polynomials committed under the shared root, in matrix-batch order.
    polys: Vec<Poly<F>>,
    /// Shared commitment root.
    commitment: MT::Commitment,
    _phantom: core::marker::PhantomData<EF>,
}

impl<F, EF, MT, const DIGEST_ELEMS: usize> Clone
    for WhirSharedBaseDeferredProverData<F, EF, MT, DIGEST_ELEMS>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    fn clone(&self) -> Self {
        Self {
            merkle_data: self.merkle_data.clone(),
            polys: self.polys.clone(),
            commitment: self.commitment.clone(),
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<F, EF, MT, const DIGEST_ELEMS: usize> Clone
    for WhirExtensionDeferredProverData<F, EF, MT, DIGEST_ELEMS>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
    <ExtensionMmcs<F, EF, MT> as Mmcs<EF>>::ProverData<DenseMatrix<EF>>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            merkle_data: self.merkle_data.clone(),
            poly: self.poly.clone(),
            ood_statement: self.ood_statement.clone(),
            proof: self.proof.clone(),
        }
    }
}

/// WHIR proof that a committed oracle satisfies a linear Sigma statement.
///
/// The reduction first turns the linear claim into one residual opening
/// `f(r) = v`; the ordinary WHIR PCS proof then binds that residual opening to
/// the same Merkle commitment.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize + Send + Sync + Clone, EF: Serialize, MT::Commitment: Serialize, MT::Proof: Serialize",
    deserialize = "F: Deserialize<'de> + Send + Sync + Clone, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
pub struct WhirLinearSigmaProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Linear Sigma sumcheck reducing the public statement to one residual
    /// opening.
    pub reduction: LinearSigmaReductionProof<F, EF>,
    /// WHIR PCS proof for the residual opening claim.
    pub opening: WhirProof<F, EF, MT>,
}

/// Errors from the WHIR-backed linear Sigma proof.
#[derive(Error, Debug)]
pub enum WhirLinearSigmaError {
    /// The linear Sigma reduction failed.
    #[error(transparent)]
    Reduction(#[from] LinearSigmaReductionError),

    /// The residual WHIR opening failed.
    #[error(transparent)]
    Pcs(#[from] VerifierError),
}

/// Deferred prover data for one oracle participating in a batched WHIR
/// initial proof.
pub enum WhirBatchedDeferredProverData<F, EF, MT, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    Base(WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>),
    Extension(WhirExtensionDeferredProverData<F, EF, MT, DIGEST_ELEMS>),
}

/// One initial oracle, or one shared-root group of initial oracles, together
/// with the coefficient used in a batched WHIR virtual polynomial.
pub enum WhirBatchedDeferredProverOracle<F, EF, MT, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    Base {
        coeff: EF,
        data: WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>,
    },
    Extension {
        coeff: EF,
        data: WhirExtensionDeferredProverData<F, EF, MT, DIGEST_ELEMS>,
    },
    SharedBase {
        coeffs: Vec<EF>,
        data: Arc<WhirSharedBaseDeferredProverData<F, EF, MT, DIGEST_ELEMS>>,
    },
}

/// Verifier metadata for one oracle participating in a batched WHIR initial
/// proof.
#[derive(Clone, Debug)]
pub enum WhirBatchedDeferredVerifierOracle<EF, Comm> {
    Base { coeff: EF, commitment: Comm },
    Extension { coeff: EF, commitment: Comm },
    SharedBase { coeffs: Vec<EF>, commitment: Comm },
}

const VIRTUAL_POLY_PAR_THRESHOLD: usize = 1 << 15;

#[inline]
fn add_scaled_base_poly<F, EF>(virtual_values: &mut [EF], coeff: EF, values: &[F])
where
    F: Field,
    EF: ExtensionField<F>,
{
    if coeff.is_zero() {
        return;
    }

    if virtual_values.len() >= VIRTUAL_POLY_PAR_THRESHOLD {
        virtual_values
            .par_iter_mut()
            .zip(values.par_iter())
            .for_each(|(slot, &value)| {
                *slot += coeff * EF::from(value);
            });
    } else {
        for (slot, &value) in virtual_values.iter_mut().zip(values.iter()) {
            *slot += coeff * EF::from(value);
        }
    }
}

#[inline]
fn add_scaled_extension_poly<F, EF>(virtual_values: &mut [EF], coeff: EF, values: &[EF])
where
    F: Field,
    EF: ExtensionField<F>,
{
    if coeff.is_zero() {
        return;
    }

    if virtual_values.len() >= VIRTUAL_POLY_PAR_THRESHOLD {
        virtual_values
            .par_iter_mut()
            .zip(values.par_iter())
            .for_each(|(slot, &value)| {
                *slot += coeff * value;
            });
    } else {
        for (slot, &value) in virtual_values.iter_mut().zip(values.iter()) {
            *slot += coeff * value;
        }
    }
}

#[inline]
fn add_scaled_shared_base_polys<F, EF>(virtual_values: &mut [EF], coeffs: &[EF], polys: &[Poly<F>])
where
    F: Field,
    EF: ExtensionField<F>,
{
    let active = polys
        .iter()
        .zip(coeffs.iter().copied())
        .filter(|(_, coeff)| !coeff.is_zero())
        .collect::<Vec<_>>();
    if active.is_empty() {
        return;
    }

    if virtual_values.len() >= VIRTUAL_POLY_PAR_THRESHOLD && active.len() > 1 {
        virtual_values
            .par_iter_mut()
            .enumerate()
            .for_each(|(row, slot)| {
                let mut value = EF::ZERO;
                for (poly, coeff) in active.iter().copied() {
                    value += coeff * EF::from(poly.as_slice()[row]);
                }
                *slot += value;
            });
    } else {
        for (poly, coeff) in active {
            for (slot, &value) in virtual_values.iter_mut().zip(poly.as_slice().iter()) {
                *slot += coeff * EF::from(value);
            }
        }
    }
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

    /// Commit to a polynomial before its user opening points are known.
    ///
    /// This runs the same initial WHIR commitment phase as
    /// [`MultilinearPcs::commit`], including DFT encoding, Merkle commitment,
    /// and OOD sampling. User equality constraints are added later by
    /// [`open_deferred`](Self::open_deferred).
    pub fn commit_deferred(
        &self,
        evaluations: RowMajorMatrix<F>,
        challenger: &mut Challenger,
    ) -> (
        MT::Commitment,
        WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>,
    )
    where
        MT::Commitment: Clone,
        Challenger: CanObserve<MT::Commitment> + Clone,
        Dft: TwoAdicSubgroupDft<F>,
    {
        assert_eq!(
            evaluations.width(),
            1,
            "WHIR currently supports committing a single polynomial",
        );
        assert_eq!(
            evaluations.height(),
            1 << self.config.num_variables,
            "evaluation vector length must be 2^num_variables",
        );

        let poly = Poly::new(evaluations.values);
        let mut statement = self.config.initial_statement(poly, self.sumcheck_strategy);

        let ds = self.build_domain_separator();
        ds.observe_domain_separator(challenger);

        let mut proof =
            WhirProof::from_protocol_parameters(&self.protocol_params, self.config.num_variables);

        let committer = CommitmentWriter::new(&self.config);
        let merkle_data = committer
            .commit(&self.dft, &mut proof, challenger, &mut statement)
            .expect("commitment phase failed");

        let commitment = proof
            .initial_commitment
            .clone()
            .expect("commitment should be set after commit phase");

        (
            commitment,
            WhirDeferredProverData {
                merkle_data,
                statement,
                proof,
            },
        )
    }

    /// Commit several base-field initial polynomials under one MMCS root.
    ///
    /// The returned data can only be opened through
    /// [`open_grouped_batched_deferred`](Self::open_grouped_batched_deferred):
    /// the verifier authenticates one Merkle path per queried row and then
    /// recombines the opened matrix rows with Fiat-Shamir sampled
    /// coefficients. Unlike [`commit_deferred`](Self::commit_deferred), this
    /// does not sample OOD points or mutate the opening transcript; the virtual
    /// polynomial is not known until the grouped opening is requested.
    pub fn commit_base_batch_deferred(
        &self,
        evaluations: Vec<RowMajorMatrix<F>>,
        _challenger: &mut Challenger,
    ) -> (
        MT::Commitment,
        Arc<WhirSharedBaseDeferredProverData<F, EF, MT, DIGEST_ELEMS>>,
    )
    where
        MT::Commitment: Clone,
        Challenger: CanObserve<MT::Commitment> + Clone,
        Dft: TwoAdicSubgroupDft<F>,
    {
        assert!(
            !evaluations.is_empty(),
            "shared WHIR commitment requires at least one polynomial",
        );

        let mut polys = Vec::with_capacity(evaluations.len());
        let mut folded_matrices = Vec::with_capacity(evaluations.len());
        for evaluations in evaluations {
            assert_eq!(
                evaluations.width(),
                1,
                "shared WHIR currently supports single-column polynomials",
            );
            assert_eq!(
                evaluations.height(),
                1 << self.config.num_variables,
                "evaluation vector length must be 2^num_variables",
            );

            let poly = Poly::new(evaluations.values);
            let num_vars = poly.num_variables();
            let mut padded = RowMajorMatrixView::new(
                poly.as_slice(),
                1 << (num_vars - self.config.folding_factor.at_round(0)),
            )
            .transpose();
            padded.pad_to_height(
                1 << (num_vars + self.config.starting_log_inv_rate
                    - self.config.folding_factor.at_round(0)),
                F::ZERO,
            );
            let folded_matrix = self.dft.dft_batch(padded).to_row_major_matrix();
            polys.push(poly);
            folded_matrices.push(folded_matrix);
        }

        let (commitment, merkle_data) = self.config.mmcs.commit(folded_matrices);

        (
            commitment.clone(),
            Arc::new(WhirSharedBaseDeferredProverData {
                merkle_data: Arc::new(merkle_data),
                polys,
                commitment,
                _phantom: core::marker::PhantomData,
            }),
        )
    }

    /// Commit to an extension-field polynomial before its opening points are
    /// known.
    ///
    /// The initial WHIR oracle is committed with [`ExtensionMmcs`], so one Merkle
    /// root and one WHIR proof bind all base-field limbs of each extension
    /// element. This is the same commitment shape WHIR already uses for folded
    /// rounds.
    pub fn commit_extension_deferred(
        &self,
        evaluations: RowMajorMatrix<EF>,
        challenger: &mut Challenger,
    ) -> (
        MT::Commitment,
        WhirExtensionDeferredProverData<F, EF, MT, DIGEST_ELEMS>,
    )
    where
        MT::Commitment: Clone,
        Challenger: CanObserve<MT::Commitment> + Clone,
        Dft: TwoAdicSubgroupDft<F>,
    {
        assert_eq!(
            evaluations.width(),
            1,
            "WHIR currently supports committing a single polynomial",
        );
        assert_eq!(
            evaluations.height(),
            1 << self.config.num_variables,
            "evaluation vector length must be 2^num_variables",
        );

        let poly = Poly::new(evaluations.values);
        let ds = self.build_domain_separator();
        ds.observe_domain_separator(challenger);

        let mut proof =
            WhirProof::from_protocol_parameters(&self.protocol_params, self.config.num_variables);

        let padded = {
            let num_vars = poly.num_variables();
            let mut mat = p3_matrix::dense::RowMajorMatrixView::new(
                poly.as_slice(),
                1 << (num_vars - self.config.folding_factor.at_round(0)),
            )
            .transpose();
            mat.pad_to_height(
                1 << (num_vars + self.config.starting_log_inv_rate
                    - self.config.folding_factor.at_round(0)),
                EF::ZERO,
            );
            mat
        };
        let folded_matrix = self.dft.dft_algebra_batch(padded).to_row_major_matrix();
        let extension_mmcs = ExtensionMmcs::new(self.config.mmcs.clone());
        let (commitment, merkle_data) = extension_mmcs.commit_matrix(folded_matrix);

        proof.initial_commitment = Some(commitment.clone());
        challenger.observe(commitment.clone());

        let mut ood_statement = EqStatement::initialize(self.config.num_variables);
        for _ in 0..self.config.commitment_ood_samples {
            let point = Point::expand_from_univariate(
                challenger.sample_algebra_element(),
                self.config.num_variables,
            );
            let eval = poly.eval_ext::<F>(&point);
            proof.initial_ood_answers.push(eval);
            challenger.observe_algebra_element(eval);
            ood_statement.add_evaluated_constraint(point, eval);
        }

        (
            commitment,
            WhirExtensionDeferredProverData {
                merkle_data,
                poly,
                ood_statement,
                proof,
            },
        )
    }

    /// Prove openings for a polynomial committed with
    /// [`commit_deferred`](Self::commit_deferred).
    pub fn open_deferred(
        &self,
        mut prover_data: WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>,
        opening_points: &[Vec<Point<EF>>],
        challenger: &mut Challenger,
    ) -> (MultilinearOpenedValues<EF>, WhirProof<F, EF, MT>)
    where
        Challenger: CanObserve<MT::Commitment>,
        Dft: TwoAdicSubgroupDft<F>,
    {
        assert_eq!(
            opening_points.len(),
            1,
            "WHIR currently supports opening a single polynomial",
        );

        let ood_statement = prover_data.statement.normalize();
        let mut statement = self
            .config
            .initial_statement(prover_data.statement.poly.clone(), self.sumcheck_strategy);

        let mut values = Vec::with_capacity(opening_points[0].len());
        for point in &opening_points[0] {
            values.push(statement.evaluate(point));
        }
        for (point, &expected) in ood_statement.iter() {
            let actual = statement.evaluate(point);
            debug_assert_eq!(actual, expected);
        }

        let prover = WhirProver(&self.config);
        prover
            .prove(
                &self.dft,
                &mut prover_data.proof,
                challenger,
                &statement,
                prover_data.merkle_data,
            )
            .expect("proving phase failed");

        (vec![values], prover_data.proof)
    }

    /// Prove openings for an extension-field polynomial committed with
    /// [`commit_extension_deferred`](Self::commit_extension_deferred).
    pub fn open_extension_deferred(
        &self,
        mut prover_data: WhirExtensionDeferredProverData<F, EF, MT, DIGEST_ELEMS>,
        opening_points: &[Vec<Point<EF>>],
        challenger: &mut Challenger,
    ) -> (MultilinearOpenedValues<EF>, WhirProof<F, EF, MT>)
    where
        Challenger: CanObserve<MT::Commitment>,
        Dft: TwoAdicSubgroupDft<F>,
    {
        assert_eq!(
            opening_points.len(),
            1,
            "WHIR currently supports opening a single polynomial",
        );

        let mut statement = EqStatement::initialize(self.config.num_variables);
        let mut values = Vec::with_capacity(opening_points[0].len());
        for point in &opening_points[0] {
            let eval = prover_data.poly.eval_ext::<F>(point);
            statement.add_evaluated_constraint(point.clone(), eval);
            values.push(eval);
        }
        statement.concatenate(&prover_data.ood_statement);

        let prover = WhirProver(&self.config);
        prover
            .prove_extension_initial(
                &self.dft,
                &mut prover_data.proof,
                challenger,
                &prover_data.poly,
                &statement,
                prover_data.merkle_data,
            )
            .expect("extension proving phase failed");

        (vec![values], prover_data.proof)
    }

    /// Prove one opening of a virtual initial oracle
    /// `g = sum_i coeff_i * f_i`, where each `f_i` has already been committed
    /// independently.
    ///
    /// The proof does not commit a new root for `g`. Its initial STIR queries
    /// open the original roots and recombine the answers with the supplied
    /// coefficients.
    pub fn open_batched_deferred(
        &self,
        oracles: Vec<WhirBatchedDeferredProverData<F, EF, MT, DIGEST_ELEMS>>,
        coeffs: &[EF],
        opening_point: Point<EF>,
        opening_value: EF,
        challenger: &mut Challenger,
    ) -> Result<WhirProof<F, EF, MT>, LinearSigmaReductionError>
    where
        Challenger: CanObserve<MT::Commitment>,
        Dft: TwoAdicSubgroupDft<F>,
    {
        if oracles.is_empty() || oracles.len() != coeffs.len() {
            return Err(LinearSigmaReductionError::ArityMismatch {
                expected: coeffs.len(),
                actual: oracles.len(),
            });
        }

        let oracles = oracles
            .into_iter()
            .zip(coeffs.iter().copied())
            .map(|(oracle, coeff)| match oracle {
                WhirBatchedDeferredProverData::Base(data) => {
                    WhirBatchedDeferredProverOracle::Base { coeff, data }
                }
                WhirBatchedDeferredProverData::Extension(data) => {
                    WhirBatchedDeferredProverOracle::Extension { coeff, data }
                }
            })
            .collect();
        self.open_grouped_batched_deferred(oracles, opening_point, opening_value, challenger)
    }

    /// Prove one opening of a virtual initial oracle where some terms may
    /// share one MMCS root.
    pub fn open_grouped_batched_deferred(
        &self,
        oracles: Vec<WhirBatchedDeferredProverOracle<F, EF, MT, DIGEST_ELEMS>>,
        opening_point: Point<EF>,
        opening_value: EF,
        challenger: &mut Challenger,
    ) -> Result<WhirProof<F, EF, MT>, LinearSigmaReductionError>
    where
        Challenger: CanObserve<MT::Commitment>,
        Dft: TwoAdicSubgroupDft<F>,
    {
        if oracles.is_empty() {
            return Err(LinearSigmaReductionError::EmptyStatement);
        }

        let ds = self.build_domain_separator();
        ds.observe_domain_separator(challenger);

        let mut virtual_values = EF::zero_vec(1 << self.config.num_variables);
        let mut prover_data = Vec::with_capacity(oracles.len());
        for oracle in oracles {
            match oracle {
                WhirBatchedDeferredProverOracle::Base { coeff, data } => {
                    if data.statement.poly.num_variables() != self.config.num_variables {
                        return Err(LinearSigmaReductionError::ArityMismatch {
                            expected: self.config.num_variables,
                            actual: data.statement.poly.num_variables(),
                        });
                    }
                    add_scaled_base_poly::<F, EF>(
                        &mut virtual_values,
                        coeff,
                        data.statement.poly.as_slice(),
                    );
                    let root = data
                        .proof
                        .initial_commitment
                        .as_ref()
                        .ok_or(LinearSigmaReductionError::FinalCheckFailed)?;
                    challenger.observe(root.clone());
                    prover_data.push(WhirBatchedInitialMerkleProverData::Base {
                        coeff,
                        data: data.merkle_data,
                    });
                }
                WhirBatchedDeferredProverOracle::Extension { coeff, data } => {
                    if data.poly.num_variables() != self.config.num_variables {
                        return Err(LinearSigmaReductionError::ArityMismatch {
                            expected: self.config.num_variables,
                            actual: data.poly.num_variables(),
                        });
                    }
                    add_scaled_extension_poly::<F, EF>(
                        &mut virtual_values,
                        coeff,
                        data.poly.as_slice(),
                    );
                    let root = data
                        .proof
                        .initial_commitment
                        .as_ref()
                        .ok_or(LinearSigmaReductionError::FinalCheckFailed)?;
                    challenger.observe(root.clone());
                    prover_data.push(WhirBatchedInitialMerkleProverData::Extension {
                        coeff,
                        data: data.merkle_data,
                    });
                }
                WhirBatchedDeferredProverOracle::SharedBase { coeffs, data } => {
                    if data.polys.len() != coeffs.len() {
                        return Err(LinearSigmaReductionError::ArityMismatch {
                            expected: data.polys.len(),
                            actual: coeffs.len(),
                        });
                    }
                    for poly in data.polys.iter() {
                        if poly.num_variables() != self.config.num_variables {
                            return Err(LinearSigmaReductionError::ArityMismatch {
                                expected: self.config.num_variables,
                                actual: poly.num_variables(),
                            });
                        }
                    }
                    add_scaled_shared_base_polys::<F, EF>(
                        &mut virtual_values,
                        &coeffs,
                        &data.polys,
                    );
                    challenger.observe(data.commitment.clone());
                    prover_data.push(WhirBatchedInitialMerkleProverData::SharedBase {
                        coeffs,
                        data: data.merkle_data.clone(),
                    });
                }
            }
        }

        let virtual_poly = Poly::new(virtual_values);
        if virtual_poly.eval_ext::<F>(&opening_point) != opening_value {
            return Err(LinearSigmaReductionError::FinalCheckFailed);
        }

        let mut proof =
            WhirProof::from_protocol_parameters(&self.protocol_params, self.config.num_variables);
        let mut statement = EqStatement::initialize(self.config.num_variables);
        statement.add_evaluated_constraint(opening_point, opening_value);
        for _ in 0..self.config.commitment_ood_samples {
            let point = Point::expand_from_univariate(
                challenger.sample_algebra_element(),
                self.config.num_variables,
            );
            let eval = virtual_poly.eval_ext::<F>(&point);
            proof.initial_ood_answers.push(eval);
            challenger.observe_algebra_element(eval);
            statement.add_evaluated_constraint(point, eval);
        }

        let prover = WhirProver(&self.config);
        prover
            .prove_batched_initial(
                &self.dft,
                &mut proof,
                challenger,
                &virtual_poly,
                &statement,
                prover_data,
            )
            .expect("batched WHIR proving phase failed");

        Ok(proof)
    }

    /// Prove that a deferred commitment satisfies a linear Sigma statement.
    ///
    /// Transcript order:
    ///
    /// 1. [`commit_deferred`](Self::commit_deferred) has already observed the
    ///    WHIR domain separator, commitment, and OOD answers.
    /// 2. The linear Sigma reduction samples its batching/sumcheck challenges.
    /// 3. The residual point produced by that reduction is opened by the same
    ///    WHIR commitment.
    ///
    /// The returned residual claim is not trusted by itself; the proof includes
    /// the WHIR opening that binds it to the committed oracle.
    pub fn open_linear_sigma_deferred(
        &self,
        prover_data: WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>,
        statement: &LinearSigmaStatement<EF>,
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<
        (LinearSigmaOpeningClaim<EF>, WhirLinearSigmaProof<F, EF, MT>),
        LinearSigmaReductionError,
    >
    where
        Challenger: CanObserve<MT::Commitment>,
        Dft: TwoAdicSubgroupDft<F>,
    {
        let (reduction, residual_claim) = statement.prove_reduction_base::<F, _>(
            &prover_data.statement.poly,
            challenger,
            reduction_pow_bits,
        )?;
        let (opened_values, opening) = self.open_deferred(
            prover_data,
            &[vec![residual_claim.point.clone()]],
            challenger,
        );
        if opened_values[0][0] != residual_claim.value {
            return Err(LinearSigmaReductionError::FinalCheckFailed);
        }

        Ok((residual_claim, WhirLinearSigmaProof { reduction, opening }))
    }

    /// Verify a deferred opening proof.
    pub fn verify_deferred(
        &self,
        commitment: &MT::Commitment,
        opening_claims: &[Vec<(Point<EF>, EF)>],
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> Result<(), VerifierError>
    where
        MT::Commitment: PartialEq,
        Challenger: CanObserve<MT::Commitment>,
        Dft: TwoAdicSubgroupDft<F>,
    {
        let parsed_commitment = self.parse_initial_commitment(commitment, proof, challenger)?;
        self.verify_deferred_after_commitment(&parsed_commitment, opening_claims, proof, challenger)
    }

    /// Verify a deferred opening after the commitment phase has already been
    /// replayed into the transcript.
    ///
    /// This is needed by composed protocols whose verifier must run additional
    /// Fiat-Shamir work between the commitment phase and the residual WHIR
    /// opening proof.
    pub fn verify_deferred_after_commitment(
        &self,
        parsed_commitment: &ParsedCommitment<EF, MT::Commitment>,
        opening_claims: &[Vec<(Point<EF>, EF)>],
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> Result<(), VerifierError>
    where
        Challenger: CanObserve<MT::Commitment>,
    {
        assert_eq!(
            opening_claims.len(),
            1,
            "WHIR currently supports verifying a single polynomial"
        );

        let mut eq_statement = EqStatement::initialize(self.config.num_variables);
        for (point, value) in &opening_claims[0] {
            eq_statement.add_evaluated_constraint(point.clone(), *value);
        }

        let verifier = WhirVerifier::new(&self.config);
        verifier.verify(proof, challenger, parsed_commitment, eq_statement)?;

        Ok(())
    }

    /// Verify a deferred opening proof whose initial oracle was committed over
    /// the extension field.
    pub fn verify_extension_deferred(
        &self,
        commitment: &MT::Commitment,
        opening_claims: &[Vec<(Point<EF>, EF)>],
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> Result<(), VerifierError>
    where
        MT::Commitment: PartialEq,
        Challenger: CanObserve<MT::Commitment>,
        Dft: TwoAdicSubgroupDft<F>,
    {
        let parsed_commitment = self.parse_initial_commitment(commitment, proof, challenger)?;
        self.verify_deferred_after_commitment(&parsed_commitment, opening_claims, proof, challenger)
    }

    /// Verify one opening proof for a virtual initial oracle
    /// `g = sum_i coeff_i * f_i`, where every `f_i` has its own independently
    /// committed WHIR root.
    ///
    /// This verifier does not accept a commitment to `g`. It replays a WHIR
    /// proof whose first STIR layer opens all original roots and recombines the
    /// answers with the supplied coefficients. This keeps the batching
    /// transcript-bound without introducing an uncommitted virtual root.
    pub fn verify_batched_deferred(
        &self,
        oracles: &[WhirBatchedDeferredVerifierOracle<EF, MT::Commitment>],
        opening_point: Point<EF>,
        opening_value: EF,
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> Result<(), VerifierError>
    where
        Challenger: CanObserve<MT::Commitment>,
        Dft: TwoAdicSubgroupDft<F>,
    {
        let ds = self.build_domain_separator();
        ds.observe_domain_separator(challenger);

        let initial_oracles = oracles
            .iter()
            .map(|oracle| match oracle {
                WhirBatchedDeferredVerifierOracle::Base { coeff, commitment } => {
                    WhirBatchedInitialVerifierOracle::Base {
                        coeff: *coeff,
                        root: commitment.clone(),
                    }
                }
                WhirBatchedDeferredVerifierOracle::Extension { coeff, commitment } => {
                    WhirBatchedInitialVerifierOracle::Extension {
                        coeff: *coeff,
                        root: commitment.clone(),
                    }
                }
                WhirBatchedDeferredVerifierOracle::SharedBase { coeffs, commitment } => {
                    WhirBatchedInitialVerifierOracle::SharedBase {
                        coeffs: coeffs.clone(),
                        root: commitment.clone(),
                    }
                }
            })
            .collect::<Vec<_>>();

        let mut statement = EqStatement::initialize(self.config.num_variables);
        statement.add_evaluated_constraint(opening_point, opening_value);

        let verifier = WhirVerifier::new(&self.config);
        verifier.verify_batched_initial(proof, challenger, &initial_oracles, statement)?;

        Ok(())
    }

    /// Verify a WHIR-backed linear Sigma proof for a deferred commitment.
    pub fn verify_linear_sigma_deferred(
        &self,
        commitment: &MT::Commitment,
        statement: &LinearSigmaStatement<EF>,
        proof: &WhirLinearSigmaProof<F, EF, MT>,
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<LinearSigmaOpeningClaim<EF>, WhirLinearSigmaError>
    where
        MT::Commitment: PartialEq,
        Challenger: CanObserve<MT::Commitment>,
        Dft: TwoAdicSubgroupDft<F>,
    {
        let parsed_commitment =
            self.parse_initial_commitment(commitment, &proof.opening, challenger)?;
        let residual_claim =
            statement.verify_reduction::<F, _>(&proof.reduction, challenger, reduction_pow_bits)?;
        self.verify_deferred_after_commitment(
            &parsed_commitment,
            &[vec![(residual_claim.point.clone(), residual_claim.value)]],
            &proof.opening,
            challenger,
        )?;

        Ok(residual_claim)
    }

    fn parse_initial_commitment(
        &self,
        commitment: &MT::Commitment,
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> Result<ParsedCommitment<EF, MT::Commitment>, VerifierError>
    where
        MT::Commitment: PartialEq,
        Challenger: CanObserve<MT::Commitment>,
    {
        let ds: DomainSeparator<EF, F> = self.build_domain_separator();
        ds.observe_domain_separator(challenger);

        let commitment_reader = CommitmentReader::new(&self.config);
        let parsed_commitment =
            commitment_reader.parse_commitment::<F, DIGEST_ELEMS>(proof, challenger);
        if &parsed_commitment.root != commitment {
            return Err(VerifierError::CommitmentMismatch);
        }

        Ok(parsed_commitment)
    }
}

impl<EF, F, MT, Challenger, Dft, const DIGEST_ELEMS: usize> MultilinearPcs<EF, Challenger>
    for WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    MT::Commitment: PartialEq,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment> + Clone,
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
        commitment: &Self::Commitment,
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
        if &parsed_commitment.root != commitment {
            return Err(VerifierError::CommitmentMismatch);
        }

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
