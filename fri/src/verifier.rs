use alloc::borrow::Cow;
use alloc::collections::btree_map::{self, BTreeMap};
use alloc::vec;
use alloc::vec::Vec;

use itertools::{Itertools, izip};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{CommitmentOpening, MatrixOpening, Mmcs, MmcsArith, PointOpening};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    Algebra, Field, HornerIter, PrimeCharacteristicRing, PrimeField64, TwoAdicField, dot_product,
};
use p3_matrix::Dimensions;
use p3_util::log2_strict_usize;
use p3_verifier_arith::{Arith, Lift};
use thiserror::Error;

use crate::{
    BatchMultiOpening, FriFoldingStrategy, FriParameters, FriProof, FriShape, PcsTranscriptFailure,
    TranscriptFailure, VerifierTranscript, fold_schedule,
};

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FriError<CommitMmcsErr, InputError>
where
    CommitMmcsErr: core::fmt::Debug,
    InputError: core::fmt::Debug,
{
    /// The proof does not carry one commit-phase opening set per commit round.
    ///
    /// The round count comes from the folding schedule.
    ///
    /// The parameters and the claimed heights fix that schedule.
    ///
    /// This list is never absorbed.
    ///
    /// So the described transcript cannot see its length.
    ///
    /// The fold chain and the shared Merkle check walk it.
    ///
    /// Each of them zips the list against the schedule.
    ///
    /// A zip over a short list would quietly skip the trailing rounds.
    #[error("commit phase opening count mismatch: expected {expected}, got {got}")]
    CommitPhaseOpeningsCountMismatch {
        /// Opening-set count the folding schedule fixes.
        expected: usize,
        /// Opening-set count the proof carries.
        got: usize,
    },
    /// The proof does not carry one commit-phase commitment per commit round.
    ///
    /// The round count is a length of the described transcript.
    ///
    /// So this is the description rejecting a run of a different shape.
    #[error("commit round count mismatch: expected {expected}, got {got}")]
    CommitRoundCountMismatch {
        /// Round count the folding schedule fixes.
        expected: usize,
        /// Round count the proof carries.
        got: usize,
    },
    #[error("round {round}: opened query count mismatch: expected {expected}, got {got}")]
    CommitPhaseQueryCountMismatch {
        round: usize,
        expected: usize,
        got: usize,
    },
    #[error("batch {batch}: opened query count mismatch: expected {expected}, got {got}")]
    InputOpeningsQueryCountMismatch {
        batch: usize,
        expected: usize,
        got: usize,
    },
    /// The proof does not carry one commit-phase grinding witness per commit round.
    #[error("commit PoW witness count mismatch: expected {expected}, got {got}")]
    CommitPowWitnessCountMismatch {
        /// Witness count the folding schedule fixes.
        expected: usize,
        /// Witness count the proof carries.
        got: usize,
    },
    /// One claimed opening carries a count the PCS transcript was not described with.
    ///
    /// Both sides derive that count from the claims the verifier was handed.
    /// Reaching this means the description and the claims came from different inputs.
    #[error("opening {opening}: claimed evaluation count mismatch: expected {expected}, got {got}")]
    ClaimedEvaluationCountMismatch {
        /// Position of the opening in commitment, matrix and point order.
        opening: usize,
        /// Claimed-evaluation count the run was described with.
        expected: usize,
        /// Claimed-evaluation count the caller supplied.
        got: usize,
    },
    #[error("final polynomial length mismatch: expected {expected}, got {got}")]
    FinalPolyLengthMismatch { expected: usize, got: usize },
    /// The instance is configured with zero queries.
    ///
    /// - The query loop never runs, so any final polynomial would be accepted.
    /// - At least one query is required for the protocol to prove anything.
    #[error("FRI instance has zero queries; at least one is required for soundness")]
    ZeroQueries,
    /// The instance commits to no matrices.
    ///
    /// - Every derived quantity comes from the committed heights.
    /// - With none, each guard would compare against nothing and pass.
    #[error("FRI instance commits to no matrices; at least one is required")]
    NoCommittedMatrices,
    /// The folding cap exceeds what this verifier can fold.
    ///
    /// Circle FRI folds two points at a time and nothing else.
    ///
    /// The schedule is derived from the cap, never read out of a proof, so any other
    /// cap would have that derivation name an arity the fold cannot apply.
    #[error("folding cap 2^{max_log_arity} exceeds the supported arity 2")]
    UnsupportedFoldingCap {
        /// The configured cap, in log form.
        max_log_arity: usize,
    },
    /// The instance is configured with a folding cap of one point per round.
    ///
    /// A round that folds nothing away never reaches the final height.
    ///
    /// `fold_schedule` asserts a positive cap rather than returning such a schedule,
    /// so this rejection is what keeps a zero cap from reaching that assert.
    #[error("FRI instance has max_log_arity = 0; a positive folding cap is required")]
    ZeroFoldingArity,
    /// The instance is configured with `log_blowup == 0`.
    ///
    /// At rate 1 every length-N word is itself a degree-<N codeword, so the
    /// folding chain and final-polynomial check are satisfied by an
    /// arbitrary reduced-opening word: false claimed evaluations verify.
    #[error("FRI instance has log_blowup = 0; a positive blowup is required for soundness")]
    ZeroBlowup,
    #[error("missing initial reduced opening at log height {expected}")]
    MissingInitialReducedOpening { expected: usize },
    #[error("initial reduced opening height mismatch: expected {expected}, got {got}")]
    InitialReducedOpeningHeightMismatch { expected: usize, got: usize },
    /// The evaluation height the claims imply has no root of unity in this field.
    ///
    /// The query phase evaluates the final polynomial at a root of unity.
    ///
    /// Its order is the global evaluation height.
    ///
    /// No such root exists past the two-adicity.
    ///
    /// A claim may name a domain as tall as the two-adicity allows.
    ///
    /// Adding the blowup can then push the evaluation height past it.
    #[error(
        "global max height 2^{log_global_max_height} exceeds field two-adicity 2^{two_adicity}"
    )]
    GlobalMaxHeightTooLarge {
        /// Log of the evaluation height the claimed domains imply.
        log_global_max_height: usize,
        /// Two-adicity of the base field.
        two_adicity: usize,
    },
    #[error("round {round}: sibling values length mismatch: expected {expected}, got {got}")]
    SiblingValuesLengthMismatch {
        round: usize,
        expected: usize,
        got: usize,
    },
    #[error("final folded height mismatch: expected {expected}, got {got}")]
    FinalFoldHeightMismatch { expected: usize, got: usize },
    /// One schedule entry names an arity this fold cannot apply.
    ///
    /// A zero entry is a round that folds nothing away, so the chain never descends.
    ///
    /// An entry at or above `usize::BITS` overflows the `1 << log_arity` the fold needs.
    ///
    /// `fold_schedule` produces neither, so only a caller-supplied schedule reaches this.
    #[error("round {round}: fold schedule arity 2^{log_arity} is out of range")]
    FoldScheduleArityOutOfRange {
        /// Position of the offending entry in the schedule.
        round: usize,
        /// The entry itself, in log form.
        log_arity: usize,
    },
    /// The arity schedule folds past the final domain size.
    ///
    /// Reducing `log_global_max_height` by `total_log_reduction` bits would drop below
    /// `log_final_height`, so the schedule cannot describe a valid fold chain.
    #[error(
        "fold schedule reduces 2^{log_global_max_height} by {total_log_reduction} bits, past the final height 2^{log_final_height}"
    )]
    FoldScheduleTooLong {
        total_log_reduction: usize,
        log_global_max_height: usize,
        log_final_height: usize,
    },
    #[error(
        "unconsumed reduced openings remain after folding: next log height {next_log_height}, remaining {remaining}"
    )]
    UnconsumedReducedOpenings {
        next_log_height: usize,
        remaining: usize,
    },
    #[error("input proof batch count mismatch: expected {expected}, got {got}")]
    InputProofBatchCountMismatch { expected: usize, got: usize },
    #[error("batch {batch}: opened-values matrix count mismatch: expected {expected}, got {got}")]
    BatchOpenedValuesCountMismatch {
        batch: usize,
        expected: usize,
        got: usize,
    },
    #[error(
        "batch {batch}, matrix {matrix}: opened at no points; its width cannot be authenticated"
    )]
    MatrixWithoutOpeningPoints { batch: usize, matrix: usize },
    #[error(
        "batch {batch}, matrix {matrix}, point {point}: evaluation count mismatch: expected {expected}, got {got}"
    )]
    PointEvaluationCountMismatch {
        batch: usize,
        matrix: usize,
        point: usize,
        expected: usize,
        got: usize,
    },
    #[error("hiding PCS random opening round count mismatch: expected {expected}, got {got}")]
    HidingRandomOpeningRoundCountMismatch { expected: usize, got: usize },
    #[error(
        "hiding PCS round {round}: random opening matrix count mismatch: expected {expected}, got {got}"
    )]
    HidingRandomOpeningMatrixCountMismatch {
        round: usize,
        expected: usize,
        got: usize,
    },
    #[error(
        "hiding PCS round {round}, matrix {matrix}: random opening point count mismatch: expected {expected}, got {got}"
    )]
    HidingRandomOpeningPointCountMismatch {
        round: usize,
        matrix: usize,
        expected: usize,
        got: usize,
    },
    #[error(
        "hiding PCS round {round}, matrix {matrix}, point {point}: random opening value count mismatch: expected {expected}, got {got}"
    )]
    HidingRandomOpeningValueCountMismatch {
        round: usize,
        matrix: usize,
        point: usize,
        expected: usize,
        got: usize,
    },
    #[error("hiding PCS preprocessing index {index} is outside {num_rounds} commitment rounds")]
    HidingPreprocessedCommitmentOutOfBounds { index: usize, num_rounds: usize },
    #[error("hiding PCS requires at least {required} random codewords, got {got}")]
    InsufficientHidingRandomCodewords { required: usize, got: usize },
    #[error(
        "hiding PCS round {round}, matrix {matrix}: insufficient hiding budget: {mask_height} mask values, {num_queries} queries, {num_opening_points} opening points, extension degree {extension_degree}"
    )]
    HidingBudgetExceeded {
        round: usize,
        matrix: usize,
        mask_height: usize,
        num_queries: usize,
        num_opening_points: usize,
        extension_degree: usize,
    },
    #[error("commit phase MMCS error: {0:?}")]
    CommitPhaseMmcsError(CommitMmcsErr),
    #[error("input error: {0:?}")]
    InputError(InputError),
    #[error("final polynomial mismatch: evaluation does not match expected value")]
    FinalPolyMismatch,
    #[error("invalid proof-of-work witness for the {0} phase")]
    InvalidPowWitness(PowPhase),
    /// A grinding witness that is not the value a zero difficulty budget admits.
    ///
    /// Checked with the other proof-shape rejections, before any transcript work.
    //
    // Why: at zero bits neither side touches the sponge.
    //
    //     prover  : grind(0)  -> returns zero, absorbs nothing
    //     verifier: the Pow step is elided, so no witness is read at all
    //
    // The field is then bound to nothing: any value rides along and still verifies.
    // Zero is the only value an honest prover emits, so zero is the only value accepted.
    #[error("non-canonical grinding witness for the {phase} phase at zero difficulty")]
    NonCanonicalPowWitness {
        /// Phase whose witness is not the canonical zero.
        phase: PowPhase,
    },
    /// A query point coincides with an opening point, so the quotient denominator is zero.
    #[error(
        "batch {batch}, matrix {matrix}, point {point}: query point coincides with the opening point"
    )]
    OpeningPointMatchesQueryPoint {
        batch: usize,
        matrix: usize,
        point: usize,
    },
}

/// Which grinding phase a rejected proof-of-work witness belongs to.
///
/// A proof carries an independent witness per phase, each guarding a different
/// challenge, so a bare "invalid witness" cannot say which parameter is
/// mismatched. Naming the phase distinguishes a prover built against different
/// `*_proof_of_work_bits` from a genuinely forged witness.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PowPhase {
    /// Guards the challenge batching the openings into one FRI instance
    /// (`FriParameters::batch_proof_of_work_bits`).
    ///
    /// Replayed by the commitment scheme wrapped around the low-degree test.
    Batch,
    /// Guards one commit-phase folding challenge
    /// (`FriParameters::commit_proof_of_work_bits`).
    CommitPhase,
    /// Guards the query indices (`FriParameters::query_proof_of_work_bits`).
    Query,
}

impl core::fmt::Display for PowPhase {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Batch => "opening-batching",
            Self::CommitPhase => "commit",
            Self::Query => "query",
        })
    }
}

impl<CommitMmcsErr, InputError> From<TranscriptFailure> for FriError<CommitMmcsErr, InputError>
where
    CommitMmcsErr: core::fmt::Debug,
    InputError: core::fmt::Debug,
{
    fn from(failure: TranscriptFailure) -> Self {
        match failure {
            // A witness that is absent and one that is too weak fail the same step.
            TranscriptFailure::PowWitness(phase) | TranscriptFailure::MissingPowWitness(phase) => {
                Self::InvalidPowWitness(phase)
            }
            TranscriptFailure::FinalPolyLen { expected, got } => {
                Self::FinalPolyLengthMismatch { expected, got }
            }
            TranscriptFailure::CommitRoundCount { expected, got } => {
                Self::CommitRoundCountMismatch { expected, got }
            }
            TranscriptFailure::CommitPowWitnessCount { expected, got } => {
                Self::CommitPowWitnessCountMismatch { expected, got }
            }
        }
    }
}

impl<CommitMmcsErr, InputError> From<PcsTranscriptFailure> for FriError<CommitMmcsErr, InputError>
where
    CommitMmcsErr: core::fmt::Debug,
    InputError: core::fmt::Debug,
{
    fn from(failure: PcsTranscriptFailure) -> Self {
        match failure {
            // A witness that is absent and one that is too weak fail the same step.
            PcsTranscriptFailure::PowWitness { .. }
            | PcsTranscriptFailure::MissingPowWitness { .. } => {
                Self::InvalidPowWitness(PowPhase::Batch)
            }
            PcsTranscriptFailure::ClaimedEvaluationCount {
                opening,
                expected,
                got,
            } => Self::ClaimedEvaluationCountMismatch {
                opening,
                expected,
                got,
            },
        }
    }
}

/// A chain of FRI input openings allowing a verifier to check a sequence of
/// FRI folds and rolls. The first element of each pair indicates the round of
/// fri in which the input should be rolled in. The second element is the opening.
pub type FriOpenings<F> = Vec<(usize, F)>;

/// Checks that every grinding witness a zero difficulty leaves unread carries the one
/// value a zero budget admits, before any transcript operation.
///
/// A positive budget needs no check for its own phase: there the witness is absorbed and
/// its sampled bits are compared, so the difficulty itself pins the field.
///
/// The three phases are checked independently, because their difficulties are independent.
//
// Why: `GrindingChallenger::check_witness` returns `true` at `bits == 0` without
// absorbing, and the typed transcript elides the step altogether, which leaves the
// witness compared against nothing and free to be any value.
//
//     bits = 0 -> prover emits zero, verifier reads nothing -> pin the field here
//     bits > 0 -> prover grinds,     verifier resamples     -> the grind pins it
//
// The commit phase carries one witness per round, so every entry is pinned rather than
// only the vector's length: a count check is not a value check.
pub(crate) fn check_canonical_pow_witnesses<
    Val,
    Challenge,
    FriMmcs,
    CommitErr,
    InputErr,
    InputProof,
>(
    params: &FriParameters<FriMmcs>,
    proof: &FriProof<Challenge, FriMmcs, Val, InputProof>,
) -> Result<(), FriError<CommitErr, InputErr>>
where
    Val: Field,
    Challenge: Field,
    FriMmcs: Mmcs<Challenge>,
    CommitErr: core::fmt::Debug,
    InputErr: core::fmt::Debug,
{
    if params.batch_proof_of_work_bits == 0 && proof.batch_pow_witness != Val::ZERO {
        return Err(FriError::NonCanonicalPowWitness {
            phase: PowPhase::Batch,
        });
    }

    if params.commit_proof_of_work_bits == 0
        && proof.commit_pow_witnesses.iter().any(|w| *w != Val::ZERO)
    {
        return Err(FriError::NonCanonicalPowWitness {
            phase: PowPhase::CommitPhase,
        });
    }

    if params.query_proof_of_work_bits == 0 && proof.query_pow_witness != Val::ZERO {
        return Err(FriError::NonCanonicalPowWitness {
            phase: PowPhase::Query,
        });
    }

    Ok(())
}

/// Verifies a FRI proof within an arithmetic.
///
/// # Overview
///
/// The shape of a run is settled before the proof is read.
///
/// ```text
///     claimed domains + parameters  ->  global evaluation height
///                                   ->  folding schedule
///                                   ->  round count and per-round arity
/// ```
///
/// Every length the proof carries is then compared against that shape.
///
/// The transcript is described by it too.
///
/// A proof of a different shape is rejected, never replayed.
///
/// The arithmetic only sees values, so the same checks run natively and inside a circuit.
///
/// # Errors
///
/// - The instance is vacuous: no queries, rate 1, a zero cap, nothing committed.
/// - The evaluation height the claims imply has no root of unity in this field.
/// - A per-round list does not carry one entry per round, or one entry per query.
/// - An opened group is not exactly one value short of its round's arity.
/// - A grinding witness is missing, too weak, or outside what a zero budget admits.
/// - An opening fails to authenticate against its commitment.
/// - A query's fold chain misses the final polynomial's own evaluation.
///
/// # Arguments
/// - `arith`: The arithmetic the checks run in.
/// - `folding`: The FRI folding scheme used by the prover.
/// - `params`: The parameters for the specific FRI protocol instance.
/// - `proof`: The proof to verify.
/// - `challenger`: The Fiat-Shamir challenger.
/// - `commitments_with_opening_points`: The claims, lifted into the arithmetic.
/// - `input_mmcs`: The commitment scheme of the claimed matrices.
/// - `alpha`: The challenge batching every claimed opening into one low-degree claim.
///   The caller draws it, against a transcript that binds the claims it batches.
///   This function runs inside a bracket the caller opens around it.
// The argument list is the protocol's own shape.
// Grouping any of it into a struct would move the same fields behind another name.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn verify_fri<A, Folding, InputMmcs, FriMmcs, Challenger>(
    arith: &mut A,
    folding: &Folding,
    params: &FriParameters<FriMmcs>,
    proof: &FriProof<A::Challenge, FriMmcs, A::Val, Folding::InputProof>,
    challenger: &mut Challenger,
    commitments_with_opening_points: &[CommitmentOpening<
        A::EF,
        <InputMmcs as MmcsArith<A, A::F>>::Target,
        TwoAdicMultiplicativeCoset<A::Val>,
    >],
    input_mmcs: &InputMmcs,
    alpha: A::EF,
) -> Result<
    (),
    FriError<<FriMmcs as MmcsArith<A, A::EF>>::Error, <InputMmcs as MmcsArith<A, A::F>>::Error>,
>
where
    A: Lift<Val: TwoAdicField + PrimeField64, F: Clone, EF: Clone>,
    InputMmcs: Mmcs<A::Val>
        + MmcsArith<
            A,
            A::F,
            Commitment = <InputMmcs as Mmcs<A::Val>>::Commitment,
            MultiProof = <InputMmcs as Mmcs<A::Val>>::MultiProof,
        >,
    FriMmcs: Mmcs<A::Challenge>
        + MmcsArith<
            A,
            A::EF,
            Commitment = <FriMmcs as Mmcs<A::Challenge>>::Commitment,
            MultiProof = <FriMmcs as Mmcs<A::Challenge>>::MultiProof,
        >,
    Challenger: FieldChallenger<A::Val>
        + GrindingChallenger<Witness = A::Val>
        + CanObserve<<FriMmcs as Mmcs<A::Challenge>>::Commitment>,
    Folding: FriFoldingStrategy<
            A::Val,
            A::Challenge,
            InputError = <InputMmcs as MmcsArith<A, A::F>>::Error,
            InputProof = Vec<BatchMultiOpening<A::Val, InputMmcs>>,
        >,
{
    // Reject a vacuous instance before any transcript work.
    // With zero queries the per-query loop never runs.
    // Any final polynomial would then pass.
    if params.num_queries == 0 {
        return Err(FriError::ZeroQueries);
    }
    // Reject rate-1 FRI: every length-N word is a degree-<N codeword, so an
    // arbitrary reduced-opening word would pass the folding chain unchecked.
    if params.log_blowup == 0 {
        return Err(FriError::ZeroBlowup);
    }
    // Reject a cap that folds nothing away.
    //
    // `compute_log_arity_for_round` asserts a positive cap, so without this guard
    // `fold_schedule` panics on its first round rather than looping.
    //
    // A rejection is the right answer either way: a cap of zero leaves every round
    // folding by zero bits, and no schedule over it ever reaches the final height.
    if params.max_log_arity == 0 {
        return Err(FriError::ZeroFoldingArity);
    }

    // Reject an instance with nothing committed.
    //
    // Every derived quantity below comes from the committed heights.
    // With none, each guard would compare against nothing and pass.
    if commitments_with_opening_points
        .iter()
        .all(|CommitmentOpening { matrices: mats, .. }| mats.is_empty())
    {
        return Err(FriError::NoCommittedMatrices);
    }

    // The batching challenge arrives from the caller, drawn against the caller's transcript.
    //
    // Soundness Error: `|f|/|EF|` where `|f|` is the number of different functions of the form
    // `(f(zeta) - fi(x))/(zeta - x)` which need to be checked.
    // Explicitly, `|f|` is `commitments_with_opening_points.flatten().flatten().len()`
    // (i.e counting the number (point, claimed_evaluation) pairs).
    //
    // The grinding step the caller plays immediately before drawing it adds
    // `batch_proof_of_work_bits` to this round's round-by-round error;
    // `p3_security::GrindingSites::batch_combination` is where the soundness model credits it.

    // Phase 1: derive the shape of this run from the configuration alone.
    //
    // Nothing in this block reads the proof.
    //
    // Both sides reach the same numbers from the parameters and the claimed heights.
    //
    // The shape is settled before any proof-supplied length is looked at.

    // Heights of the folding inputs, tallest first.
    //
    // A claim is what the caller asked to have checked.
    //
    // So the claim is what fixes the geometry.
    //
    // Folding sees one input per distinct height.
    //
    // Equal heights therefore collapse into one.
    let mut input_log_heights: Vec<usize> = commitments_with_opening_points
        .iter()
        .flat_map(|CommitmentOpening { matrices: mats, .. }| {
            mats.iter().map(|MatrixOpening { domain, .. }| {
                log2_strict_usize(domain.size()) + params.log_blowup
            })
        })
        .collect();
    input_log_heights.sort_unstable_by(|a, b| b.cmp(a));
    input_log_heights.dedup();

    // The tallest input is where folding starts.
    //
    // Ref: Ben-Sasson et al., "Fast RS IOPP", ICALP 2018, section 2.1.1.
    //
    //     H = max claimed log_2(domain.size) + log_blowup
    let log_global_max_height = input_log_heights[0];

    // Bound the global height by the field two-adicity before using it.
    //
    // The query phase evaluates the final polynomial at a 2^H-th root of unity.
    //
    // No such root exists past the two-adicity.
    //
    // Asking for one panics.
    //
    // A claim may name a domain as tall as the two-adicity allows.
    //
    // Adding the blowup can then push the evaluation height past it.
    if log_global_max_height > A::Val::TWO_ADICITY {
        return Err(FriError::GlobalMaxHeightTooLarge {
            log_global_max_height,
            two_adicity: A::Val::TWO_ADICITY,
        });
    }

    // The folding schedule.
    //
    // It carries the round count and the arity of every round.
    //
    // Two properties come with the derivation:
    //
    //     1 <= entry <= max_log_arity        every round folds, none overshoots the cap
    //     sum(entries) == H - H_final        the chain lands on the final height
    //
    // No proof can contradict a property of the derivation.
    let log_arities = fold_schedule(
        &input_log_heights,
        params.log_blowup + params.log_final_poly_len,
        params.max_log_arity,
    );
    let num_rounds = log_arities.len();

    // Phase 2: check every proof-supplied length against the shape just derived.
    //
    // Each comparison has configuration on the left and untrusted input on the right.
    //
    // A length taken from the proof never sizes anything.

    // One commit-phase opening set per round.
    //
    // This list is never absorbed.
    //
    // So the described transcript cannot see its length.
    //
    // The fold chain and the shared Merkle check walk it.
    //
    // Each of them zips the list against the schedule.
    //
    // A zip over a short list would skip the trailing rounds instead of failing.
    if proof.commit_phase_openings.len() != num_rounds {
        return Err(FriError::CommitPhaseOpeningsCountMismatch {
            expected: num_rounds,
            got: proof.commit_phase_openings.len(),
        });
    }

    // Every round must open exactly the sampled queries.
    //
    // Every opened group must be one value short of its round's arity.
    //
    //     sibling_values          one row per query
    //     sibling_values[query]   arity - 1 values, the queried one omitted
    //
    // No fewer, and no unopened extras riding along.
    //
    // The arity comes from the derived schedule.
    //
    // So the width of every opened group is settled before the proof is looked at.
    for (round, (opening, &log_arity)) in
        izip!(&proof.commit_phase_openings, &log_arities).enumerate()
    {
        if opening.sibling_values.len() != params.num_queries {
            return Err(FriError::CommitPhaseQueryCountMismatch {
                round,
                expected: params.num_queries,
                got: opening.sibling_values.len(),
            });
        }
        let arity = 1 << log_arity;
        for siblings in &opening.sibling_values {
            if siblings.len() != arity - 1 {
                return Err(FriError::SiblingValuesLengthMismatch {
                    round,
                    expected: arity - 1,
                    got: siblings.len(),
                });
            }
        }
    }

    // A zero grinding budget leaves the witness unread.
    //
    // So the three fields are pinned here rather than by their grinds.
    check_canonical_pow_witnesses(params, proof)?;

    // Ensure that the final polynomial has the expected degree.
    //
    // The transcript describes it as one step of that width.
    // A different width is untrusted input, so it is rejected before absorbing.
    if proof.final_poly.len() != params.final_poly_len() {
        return Err(FriError::FinalPolyLengthMismatch {
            expected: params.final_poly_len(),
            got: proof.final_poly.len(),
        });
    }

    // Phase 3: replay the transcript the derived shape describes.
    //
    // Every value the native transcript absorbs is lifted and announced first, in absorb order.
    let commit_targets: Vec<_> = proof
        .commit_phase_commits
        .iter()
        .map(|commit| params.mmcs.load_commitment(arith, commit))
        .collect();
    for (round, target) in commit_targets.iter().enumerate() {
        params.mmcs.will_observe(arith, target);
        if params.commit_proof_of_work_bits > 0
            && let Some(&witness) = proof.commit_pow_witnesses.get(round)
        {
            let witness = arith.load(witness);
            arith.will_observe(vec![witness]);
        }
    }
    let final_poly = arith.load_ext_slice(&proof.final_poly).into_owned();
    for coefficient in &final_poly {
        arith.will_observe_ext(coefficient);
    }
    if params.query_proof_of_work_bits > 0 {
        let witness = arith.load(proof.query_pow_witness);
        arith.will_observe(vec![witness]);
    }

    // The batching challenge above belongs to the caller's transcript, not to
    // FRI's, so seeding starts here.
    let index_bits = log_global_max_height + folding.extra_query_index_bits();
    let mut transcript = arith.seeding(|| {
        VerifierTranscript::<Challenger, A::Val, A::Challenge>::new(
            challenger,
            FriShape::with_schedule(params, log_arities.clone(), index_bits),
        )
    });

    // One folding challenge per round, each guarded by its own grinding step.
    //
    // The round count and the witness count are lengths of the description.
    //
    // So the transcript is what compares them against what the proof carries.
    let betas =
        transcript.commit_rounds(&proof.commit_phase_commits, &proof.commit_pow_witnesses)?;
    let betas: Vec<A::EF> = betas
        .into_iter()
        .map(|beta| arith.sampled_ext(beta))
        .collect();

    // Bind the final polynomial, re-check the query grind, redraw the indices.
    let indices = transcript.query_phase(&proof.final_poly, Some(proof.query_pow_witness))?;
    let indices: Vec<A::Index> = indices
        .into_iter()
        .map(|index| arith.sampled_index(index, index_bits))
        .collect();

    // Every described step has now been replayed.
    transcript.finish();

    // The log of the final domain size.
    let log_final_height = params.log_blowup + params.log_final_poly_len;

    // Check all input openings against their commitments (one shared proof per
    // batch) and combine the opened values into each query's FRI inputs.
    let reduced_openings = open_inputs::<A, InputMmcs, <FriMmcs as MmcsArith<A, A::EF>>::Error>(
        arith,
        params.log_blowup,
        log_global_max_height,
        &indices,
        &proof.input_openings,
        alpha,
        input_mmcs,
        commitments_with_opening_points,
    )?;

    // Walk every query's fold chain, which is pure arithmetic.
    //
    // Each round's full evaluation row is reconstructed as the chain passes it.
    //
    // The rows are authenticated afterwards, one shared check per round.
    //
    // Both collectors are sized by the derived round count, never by the proof.
    let mut group_indices_by_round: Vec<Vec<A::Index>> =
        vec![Vec::with_capacity(params.num_queries); num_rounds];
    // Each entry holds the opened rows of one round's single committed matrix.
    //
    // The per-query shape matches what the multi-opening verification expects.
    let mut rows_by_round: Vec<Vec<Vec<Cow<'_, [A::EF]>>>> =
        vec![Vec::with_capacity(params.num_queries); num_rounds];

    for (query, (index, ro)) in izip!(&indices, reduced_openings).enumerate() {
        // If we queried extra bits, shift them off now.
        let domain_index = arith.index_shr(index, folding.extra_query_index_bits());

        // The siblings this query opened in every round.
        let siblings: Vec<Vec<A::EF>> = proof
            .commit_phase_openings
            .iter()
            .map(|opening| {
                arith
                    .load_ext_slice(&opening.sibling_values[query])
                    .into_owned()
            })
            .collect();

        // Starting at the evaluation at `index` of the initial domain,
        // perform FRI folds until the domain size reaches the final domain size,
        // recording the reconstructed evaluation row of every round.
        let (folded_eval, rows) = fold_query::<A, Folding, _, _>(
            arith,
            folding,
            &domain_index,
            &betas,
            &log_arities,
            &siblings,
            ro,
            log_global_max_height,
            log_final_height,
        )?;

        // Hand this query's group indices and reconstructed rows to the collectors.
        //
        // The group index of round `r` drops the bits every round up to `r` folded away.
        let mut log_folded = 0;
        for (round, (&log_arity, row)) in izip!(&log_arities, rows).enumerate() {
            log_folded += log_arity;
            group_indices_by_round[round].push(arith.index_shr(&domain_index, log_folded));
            rows_by_round[round].push(vec![Cow::Owned(row)]);
        }

        // The final polynomial is opened at the folded index.
        //
        //     x = g^rev(folded_index),  g of order 2^log_global_max_height
        let folded_index = arith.index_shr(&domain_index, log_folded);
        let x = arith.reversed_index_power(
            A::Val::two_adic_generator(log_global_max_height),
            &folded_index,
            log_global_max_height,
        );

        // Assuming all the checks passed, the final check is to ensure that the folded evaluation
        // matches the evaluation of the final polynomial sent by the prover.
        let eval: A::EF = final_poly.iter().cloned().horner(x);
        arith.assert_zero(eval - folded_eval, || FriError::FinalPolyMismatch)?;
    }

    // Verify the commitment to the evaluations of every queried group, one
    // shared amortized check per round. Paths that share a parent reuse a
    // single compression instead of recomputing it once per query.
    let mut log_current_height = log_global_max_height;
    for (round, ((commit, opening), &log_arity)) in commit_targets
        .iter()
        .zip(&proof.commit_phase_openings)
        .zip(&log_arities)
        .enumerate()
    {
        let arity = 1 << log_arity;
        let log_folded_height = log_current_height - log_arity;
        let dims = &[Dimensions {
            width: arity,
            height: 1 << log_folded_height,
        }];
        params
            .mmcs
            .verify_rows(
                arith,
                commit,
                dims,
                &group_indices_by_round[round],
                &rows_by_round[round],
                &opening.opening_proof,
            )
            .map_err(FriError::CommitPhaseMmcsError)?;
        log_current_height = log_folded_height;
    }

    Ok(())
}

/// Verifies a single query chain in the FRI proof. This is the verifier complement
/// to the prover's `answer_queries` function.
///
/// Given an initial `index` corresponding to a point in the initial domain
/// and a series of `reduced_openings` corresponding to evaluations of
/// polynomials to be added in at specific domain sizes, perform the standard
/// sequence of FRI folds, reconstructing at each step the full evaluation row
/// the prover committed to.
///
/// This pass is pure arithmetic: the reconstructed rows are returned with the folded value.
///
/// The caller authenticates them afterwards, one shared amortized check per round.
///
/// With variable arity, each round may fold by a different factor.
///
/// Those factors come from the schedule the caller supplies.
///
/// # Security
///
/// The returned value is **not authenticated**.
///
/// It is the folded evaluation implied by the rows this pass reconstructed.
///
/// It becomes meaningful only once the caller authenticates those rows.
///
/// The shared per-round Merkle check is what does that, after every query is folded.
///
/// On its own the value proves nothing about what the prover committed to.
///
/// # Errors
///
/// Every shape obligation on proof-controlled input is checked here.
///
/// So this pass is total on well-typed input:
///
/// - Each schedule entry must name an arity this pass can shift and fold by.
/// - Each round must open this query with one value fewer than its arity.
/// - The schedule must not fold past the final height.
///
/// Each schedule entry must be at most the configured `max_log_arity`, as `fold_schedule` produces.
///
/// This pass does not see that cap, so it bounds each entry only by what it can itself apply.
///
/// Arguments:
/// - `arith`: The arithmetic the fold runs in.
/// - `folding`: The FRI folding scheme used by the prover.
/// - `index`: The opening index for the unfolded polynomial.
/// - `betas`: The folding challenge of each round.
/// - `log_arities`: The validated log-arity schedule, one entry per round.
/// - `siblings`: This query's opened siblings, one group per round.
/// - `reduced_openings`: Pairs of a log height and an opening. The opening is a linear combination
///   of all input polynomials of that size opened at the appropriate index. Each opening is added into
///   the FRI folding chain once the domain size reaches the size specified in the pair.
/// - `log_global_max_height`: The log of the maximum domain size.
/// - `log_final_height`: The log of the final domain size.
#[expect(clippy::too_many_arguments, clippy::type_complexity)]
#[inline]
pub fn fold_query<A, Folding, CommitErr, InputErr>(
    arith: &mut A,
    folding: &Folding,
    index: &A::Index,
    betas: &[A::EF],
    log_arities: &[usize],
    siblings: &[Vec<A::EF>],
    reduced_openings: FriOpenings<A::EF>,
    log_global_max_height: usize,
    log_final_height: usize,
) -> Result<(A::EF, Vec<Vec<A::EF>>), FriError<CommitErr, InputErr>>
where
    A: Arith<EF: Clone>,
    Folding: FriFoldingStrategy<A::Val, A::Challenge>,
    CommitErr: core::fmt::Debug,
    InputErr: core::fmt::Debug,
{
    // Each entry must name a real arity before anything shifts or sums by it.
    //
    //     log_arity == 0   ->  a round that folds nothing away
    //     log_arity >= 64  ->  `1 << log_arity` overflows the shift below
    //
    // `verify_fri` derives its schedule from `fold_schedule`, whose entries are already
    // in `1..=max_log_arity`. A caller replaying one query on its own is not assumed to
    // have done that, and this pass is the last place to catch it.
    for (round, &log_arity) in log_arities.iter().enumerate() {
        if log_arity == 0 || log_arity >= usize::BITS as usize {
            return Err(FriError::FoldScheduleArityOutOfRange { round, log_arity });
        }
    }

    // Shape checks on the proof-controlled openings, before any indexing into them.
    // `verify_fri` establishes these for every query up front; a caller replaying a single
    // query cannot be assumed to have, and the row reconstruction below indexes each group.
    for (round, (group, &log_arity)) in izip!(siblings, log_arities).enumerate() {
        let arity = 1 << log_arity;
        if group.len() != arity - 1 {
            return Err(FriError::SiblingValuesLengthMismatch {
                round,
                expected: arity - 1,
                got: group.len(),
            });
        }
    }

    // The schedule may still fold past the final height, which would underflow
    // `log_current_height - log_arity` below. Under-folding is caught by the terminal
    // height check instead, which reports the height actually reached.
    //
    // Every entry is under `usize::BITS` by now, but the sum over enough rounds is not,
    // so it saturates. A saturated total exceeds any reachable reduction and is rejected
    // right here, which is where an overflowing schedule belongs anyway.
    let total_log_reduction: usize = log_arities.iter().copied().fold(0, usize::saturating_add);
    if total_log_reduction > log_global_max_height.saturating_sub(log_final_height) {
        return Err(FriError::FoldScheduleTooLong {
            total_log_reduction,
            log_global_max_height,
            log_final_height,
        });
    }

    let mut ro_iter = reduced_openings.into_iter().peekable();

    // These checks are not essential to security,
    // but they should be satisfied by any non malicious prover.
    // ro_iter being empty means that we have committed to no polynomials at all and
    // we need to roll in a polynomial initially otherwise we are just folding a zero polynomial.
    let Some((first_log_height, _)) = ro_iter.peek() else {
        return Err(FriError::MissingInitialReducedOpening {
            expected: log_global_max_height,
        });
    };
    if *first_log_height != log_global_max_height {
        return Err(FriError::InitialReducedOpeningHeightMismatch {
            expected: log_global_max_height,
            got: *first_log_height,
        });
    }
    let mut folded_eval = ro_iter.next().expect("peeked above").1;

    // Track the current log_height as we fold down
    let mut log_current_height = log_global_max_height;
    let mut index = index.clone();
    let mut rows = Vec::with_capacity(log_arities.len());

    // We start with evaluations over a domain of size (1 << log_global_max_height). We fold
    // using FRI until the domain size reaches (1 << log_final_height).
    for (beta, &log_arity, group) in izip!(betas, log_arities, siblings) {
        // Reconstruct the full evaluation row from self + siblings.
        //
        // The queried value sits at the position the low `log_arity` index bits name.
        //
        // The row is authenticated by the round's shared opening proof,
        // which binds this query's folded value to the committed codeword.
        let row = arith.insert_at(&index, log_arity, folded_eval, group);

        // Compute the new height after folding
        let log_folded_height = log_current_height - log_arity;

        // Replace index with the index of the parent FRI node.
        index = arith.index_shr(&index, log_arity);

        // Fold the sibling group into the parent FRI node's evaluation.
        folded_eval = folding.fold_row(
            arith,
            &index,
            log_folded_height,
            log_arity,
            beta.clone(),
            &row,
        );
        rows.push(row);

        // Update current height
        log_current_height = log_folded_height;

        // If there are new polynomials to roll in at the folded height, do so.
        //
        // Each element of `ro_iter` is the evaluation of a reduced opening polynomial, which is itself
        // a random linear combination `f_{i, 0}(x) + alpha f_{i, 1}(x) + ...`, but when we add it
        // to the current folded polynomial evaluation claim, we need to multiply by a new random factor
        // since `f_{i, 0}` has no leading coefficient.
        //
        // We use `beta^arity` as the random factor to maintain independence.
        if let Some((_, ro)) = ro_iter.next_if(|(lh, _)| *lh == log_folded_height) {
            folded_eval += beta.exp_power_of_2(log_arity) * ro;
        }
    }

    // Verify we reached the expected final height
    if log_current_height != log_final_height {
        return Err(FriError::FinalFoldHeightMismatch {
            expected: log_final_height,
            got: log_current_height,
        });
    }

    // If ro_iter is not empty, we failed to fold in some polynomial evaluations.
    if let Some((next_log_height, _)) = ro_iter.next() {
        return Err(FriError::UnconsumedReducedOpenings {
            next_log_height,
            remaining: 1 + ro_iter.count(),
        });
    }

    // If we reached this point, the chain of folds starting at the initial
    // index produces folded_eval, contingent on the per-round row
    // authentication performed by the caller.
    Ok((folded_eval, rows))
}

/// The openings of one height at one point, which share the denominator `z - x`.
struct OpeningGroup<EF, Challenge> {
    /// Log height of the matrices, blowup included.
    log_height: usize,
    /// The opening point `z`.
    point: EF,
    /// The native value of `z`, which identifies the group.
    value: Challenge,
    /// `S(z)`: every claimed value of the group, combined with its rung of the alpha ladder.
    sum_at_z: EF,
    /// The first opening point of the group, as (batch, matrix, point).
    position: (usize, usize, usize),
}

/// Given the query indices and one shared opening per input batch, check every
/// opening proof and combine the opened values into each query's FRI inputs
/// along the path specified by its index.
///
/// In cases where the maximum height of a batch of matrices is smaller than the
/// global max height, shift the indices down to compensate.
///
/// We combine the functions by mapping each function and opening point pair to `(f(z) - f(x))/(z - x)`
/// and then combining functions of the same degree using the challenge alpha.
///
/// ## Arguments:
/// - `arith`: The arithmetic the checks run in.
/// - `log_blowup`: The log of the evaluation-domain blowup.
/// - `log_global_max_height`: The log of the maximum height of the input matrices.
/// - `indices`: The index at which to open the functions, one per query.
/// - `input_openings`: One multi-opening per input batch, containing the opened
///   values of every query for a collection of matrices along with one shared proof.
/// - `alpha`: The challenge used to combine the functions.
/// - `input_mmcs`: The input multi-matrix commitment scheme.
/// - `commitments_with_opening_points`: The claims, lifted into the arithmetic.
#[expect(clippy::type_complexity, clippy::too_many_arguments)]
#[inline]
pub fn open_inputs<A, InputMmcs, CommitErr>(
    arith: &mut A,
    log_blowup: usize,
    log_global_max_height: usize,
    indices: &[A::Index],
    input_openings: &[BatchMultiOpening<A::Val, InputMmcs>],
    alpha: A::EF,
    input_mmcs: &InputMmcs,
    commitments_with_opening_points: &[CommitmentOpening<
        A::EF,
        <InputMmcs as MmcsArith<A, A::F>>::Target,
        TwoAdicMultiplicativeCoset<A::Val>,
    >],
) -> Result<Vec<FriOpenings<A::EF>>, FriError<CommitErr, <InputMmcs as MmcsArith<A, A::F>>::Error>>
where
    A: Lift<Val: TwoAdicField, F: Clone, EF: Clone>,
    InputMmcs:
        Mmcs<A::Val> + MmcsArith<A, A::F, MultiProof = <InputMmcs as Mmcs<A::Val>>::MultiProof>,
    CommitErr: core::fmt::Debug,
{
    if input_openings.len() != commitments_with_opening_points.len() {
        return Err(FriError::InputProofBatchCountMismatch {
            expected: commitments_with_opening_points.len(),
            got: input_openings.len(),
        });
    }

    // The opened rows of every batch, lifted once and read by both passes below.
    //
    // `opened_rows[batch][query][matrix]` is one opened row.
    let mut opened_rows = Vec::with_capacity(input_openings.len());

    // Check every batch's openings against its commitment first, one shared
    // amortized verification per batch, so the arithmetic below only ever
    // reads authenticated values.
    for (
        batch,
        (
            batch_opening,
            CommitmentOpening {
                commitment: batch_commit,
                matrices: mats,
            },
        ),
    ) in input_openings
        .iter()
        .zip(commitments_with_opening_points.iter())
        .enumerate()
    {
        // One opened row set per query.
        if batch_opening.opened_values.len() != indices.len() {
            return Err(FriError::InputOpeningsQueryCountMismatch {
                batch,
                expected: indices.len(),
                got: batch_opening.opened_values.len(),
            });
        }

        // The opened rows must pair one-to-one with the committed matrices.
        // Why: the width derivation below zips rows with matrix metadata.
        for opened_values in &batch_opening.opened_values {
            if opened_values.len() != mats.len() {
                return Err(FriError::BatchOpenedValuesCountMismatch {
                    batch,
                    expected: mats.len(),
                    got: opened_values.len(),
                });
            }
        }

        // Find the height of each matrix in the batch.
        // Currently we only check domain.size() as the shift is
        // assumed to always be Val::GENERATOR.
        let batch_heights = mats
            .iter()
            .map(|MatrixOpening { domain, .. }| domain.size() << log_blowup)
            .collect_vec();
        let batch_dims = batch_heights
            .iter()
            .zip(mats)
            .enumerate()
            .map(
                |(
                    matrix,
                    (
                        &height,
                        MatrixOpening {
                            points: points_and_values,
                            ..
                        },
                    ),
                )| {
                    // Pin each matrix width to its claimed evaluation count, never to the proof.
                    //
                    // Why: the leaf hash flattens all same-height rows into one stream.
                    //   - `check_widths` (in the MMCS) is the only authenticator of row boundaries
                    //   - a matrix opened at no points carries no claim to pin its width
                    //   - reading the width from the proof-supplied row leaves that boundary unchecked
                    // Every input matrix is opened at >= 1 point, so reject the degenerate case.
                    let PointOpening { values, .. } = points_and_values
                        .first()
                        .ok_or(FriError::MatrixWithoutOpeningPoints { batch, matrix })?;
                    Ok(Dimensions {
                        width: values.len(),
                        height,
                    })
                },
            )
            .collect::<Result<Vec<_>, FriError<CommitErr, _>>>()?;

        // If the maximum height of the batch is smaller than the global max height,
        // we need to correct the indices by right shifting them.
        let bits_reduced = batch_heights
            .iter()
            .max()
            .map_or(0, |&h| log_global_max_height - log2_strict_usize(h));
        let reduced_indices: Vec<A::Index> = indices
            .iter()
            .map(|index| arith.index_shr(index, bits_reduced))
            .collect();

        let rows: Vec<Vec<Cow<'_, [A::F]>>> = batch_opening
            .opened_values
            .iter()
            .map(|query_rows| query_rows.iter().map(|row| arith.load_slice(row)).collect())
            .collect();

        input_mmcs
            .verify_rows(
                arith,
                batch_commit,
                &batch_dims,
                &reduced_indices,
                &rows,
                &batch_opening.opening_proof,
            )
            .map_err(FriError::InputError)?;
        opened_rows.push(rows);
    }

    // The point every matrix is queried at, one per query and per distinct height.
    //
    // A matrix of log height `h` reads the index with its low `H - h` bits dropped.
    //
    //     x = GENERATOR * w_h^rev(index >> (H - h))
    let mut query_points: Vec<BTreeMap<usize, A::F>> = Vec::with_capacity(indices.len());
    for index in indices {
        let mut points = BTreeMap::new();
        for CommitmentOpening { matrices: mats, .. } in commitments_with_opening_points {
            for MatrixOpening { domain, .. } in mats {
                let log_height = log2_strict_usize(domain.size()) + log_blowup;
                if let btree_map::Entry::Vacant(entry) = points.entry(log_height) {
                    let reduced = arith.index_shr(index, log_global_max_height - log_height);
                    let power = arith.reversed_index_power(
                        A::Val::two_adic_generator(log_height),
                        &reduced,
                        log_height,
                    );
                    entry.insert(power * A::Val::GENERATOR);
                }
            }
        }
        query_points.push(points);
    }

    // Within one `log_height`, the powers of `alpha` form a ladder.
    //
    // It advances once per opened column, walking (batch, matrix, point, column) in a fixed order.
    //
    // The column counts come from the claimed evaluations, which bind every opened row.
    //
    // So the ladder is the same for every query.
    //
    // The longest ladder over all heights bounds how many powers of alpha are ever needed.
    let mut ladder_lens = BTreeMap::<usize, usize>::new();
    for CommitmentOpening { matrices: mats, .. } in commitments_with_opening_points {
        for MatrixOpening { domain, points } in mats {
            let log_height = log2_strict_usize(domain.size()) + log_blowup;
            *ladder_lens.entry(log_height).or_insert(0) +=
                points.iter().map(|p| p.values.len()).sum::<usize>();
        }
    }
    let alpha_powers: Vec<A::EF> = alpha
        .powers()
        .take(ladder_lens.into_values().max().unwrap_or(0))
        .collect();

    // Every matrix of one height opened at one point shares the denominator `z - x`.
    //
    // So the openings are grouped by height and point, and each query divides once per group.
    //
    //     reduced(x) = sum_g (S_g(z) - S_g(x)) / (z_g - x)
    //
    // `S_g` combines every column of group `g` with its rung of the alpha ladder.
    //
    // The `S_g(z)` half depends only on the claimed values, so every query shares it.
    let mut groups: Vec<OpeningGroup<A::EF, A::Challenge>> = Vec::new();
    // The group and ladder start of every opening point, in (batch, matrix, point) order.
    let mut placements: Vec<(usize, usize)> = Vec::new();
    let mut ladder_offsets = BTreeMap::<usize, usize>::new();
    for (batch, CommitmentOpening { matrices: mats, .. }) in
        commitments_with_opening_points.iter().enumerate()
    {
        for (matrix, MatrixOpening { domain, points }) in mats.iter().enumerate() {
            let log_height = log2_strict_usize(domain.size()) + log_blowup;
            for (
                point,
                PointOpening {
                    point: z,
                    values: ps_at_z,
                },
            ) in points.iter().enumerate()
            {
                let offset = ladder_offsets.entry(log_height).or_insert(0);
                let start = *offset;
                *offset += ps_at_z.len();
                let sum_at_z = dot_product(
                    alpha_powers[start..*offset].iter().cloned(),
                    ps_at_z.iter().cloned(),
                );
                let value = arith.value_ext(z);
                let group = if let Some(group) = groups
                    .iter()
                    .position(|g| g.log_height == log_height && g.value == value)
                {
                    groups[group].sum_at_z += sum_at_z;
                    group
                } else {
                    groups.push(OpeningGroup {
                        log_height,
                        point: z.clone(),
                        value,
                        sum_at_z,
                        position: (batch, matrix, point),
                    });
                    groups.len() - 1
                };
                placements.push((group, start));
            }
        }
    }

    // The denominators of every query and group, inverted together in one batch.
    //
    // A coinciding opening point makes its quotient undefined and is rejected by position.
    let mut denominators = Vec::with_capacity(query_points.len() * groups.len());
    for points in &query_points {
        for group in &groups {
            denominators.push(group.point.clone() - points[&group.log_height].clone());
        }
    }
    let inverses = arith.batch_inverse(&denominators, |at| {
        let (batch, matrix, point) = groups[at % groups.len()].position;
        FriError::OpeningPointMatchesQueryPoint {
            batch,
            matrix,
            point,
        }
    })?;

    // Combine each query's authenticated values into its reduced openings.
    (0..indices.len())
        .map(|query| {
            // `S_g(x)` of every group, a dot product of ladder powers against authenticated rows.
            let mut sums_at_x = vec![A::EF::ZERO; groups.len()];
            let mut placements = placements.iter();
            for (batch, (batch_rows, CommitmentOpening { matrices: mats, .. })) in opened_rows
                .iter()
                .zip(commitments_with_opening_points.iter())
                .enumerate()
            {
                for (matrix, (mat_opening, MatrixOpening { points, .. })) in
                    batch_rows[query].iter().zip(mats.iter()).enumerate()
                {
                    // Every point of a matrix combines the same row, one ladder rung further on.
                    //
                    // So the first point's combination, times `alpha^(start - first)`, is every other.
                    let mut first: Option<(usize, A::EF)> = None;
                    for (
                        point,
                        PointOpening {
                            values: ps_at_z, ..
                        },
                    ) in points.iter().enumerate()
                    {
                        if mat_opening.len() != ps_at_z.len() {
                            return Err(FriError::PointEvaluationCountMismatch {
                                batch,
                                matrix,
                                point,
                                expected: mat_opening.len(),
                                got: ps_at_z.len(),
                            });
                        }
                        let &(group, start) =
                            placements.next().expect("one placement per opening point");
                        let (first_start, combination) = first.get_or_insert_with(|| {
                            let combination = A::EF::batched_linear_combination(
                                &alpha_powers[start..start + mat_opening.len()],
                                mat_opening,
                            );
                            (start, combination)
                        });
                        sums_at_x[group] += if start == *first_start {
                            combination.clone()
                        } else {
                            combination.clone() * alpha_powers[start - *first_start].clone()
                        };
                    }
                }
            }

            // log_height -> reduced_opening
            let mut reduced_openings = BTreeMap::<usize, A::EF>::new();
            let query_inverses = &inverses[query * groups.len()..(query + 1) * groups.len()];
            for ((group, sum_at_x), inverse) in groups.iter().zip(sums_at_x).zip(query_inverses) {
                *reduced_openings
                    .entry(group.log_height)
                    .or_insert(A::EF::ZERO) +=
                    (group.sum_at_z.clone() - sum_at_x) * inverse.clone();
            }

            // The blowup-height entry exists only for a height-1 (constant) trace matrix.
            // Its quotient `(f(zeta) - f(x)) / (zeta - x)` must then be zero.
            // One check after all batches suffices: the random combining challenge rules out cancellation.
            if let Some(ro) = reduced_openings.get(&log_blowup) {
                arith.assert_zero(ro.clone(), || FriError::FinalPolyMismatch)?;
            }

            // Return reduced openings descending by log_height.
            Ok(reduced_openings.into_iter().rev().collect())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use core::marker::PhantomData;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::{ExtensionMmcs, Mmcs, Pcs};
    use p3_dft::Radix2Dit;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, HornerIter, PrimeCharacteristicRing, TwoAdicField};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::{MerkleTreeError, MerkleTreeMmcs};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use p3_verifier_arith::{Native, insert_by_bits};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::{
        CommitmentWithOpeningPoints, FriParameters, PcsShape, PcsVerifierTranscript,
        TwoAdicFriFolding, TwoAdicFriFoldingForMmcs, TwoAdicFriPcs,
    };

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type Proof = FriProof<Challenge, ChallengeMmcs, Val, Vec<BatchMultiOpening<Val, ValMmcs>>>;
    type Folding = TwoAdicFriFoldingForMmcs<Val, ValMmcs>;
    type ChallengeMmcsError = <ChallengeMmcs as Mmcs<Challenge>>::Error;
    type TestError = FriError<ChallengeMmcsError, <ValMmcs as Mmcs<Val>>::Error>;
    type NativeArith = Native<Val, Challenge>;

    /// Replay one query against a caller-supplied schedule, with everything else empty.
    ///
    /// The schedule is validated before any opening is indexed, so no proof is needed.
    fn fold_query_with_schedule(log_arities: &[usize]) -> Result<Challenge, TestError> {
        fold_query::<NativeArith, Folding, _, _>(
            &mut NativeArith::new(),
            &TwoAdicFriFolding(PhantomData),
            &0,
            &[],
            log_arities,
            &[],
            vec![],
            8,
            1,
        )
        .map(|(folded, _)| folded)
    }

    #[test]
    fn rows_hold_the_value_at_its_position_and_the_siblings_in_order() {
        // Every arity up to the largest a schedule uses, and every position within it.
        for log_arity in 1..=4 {
            let arity = 1 << log_arity;
            let siblings: Vec<Challenge> = (1..arity)
                .map(|i| Challenge::from(Val::from_usize(i)))
                .collect();
            let value = Challenge::from(Val::from_usize(1000));
            for position in 0..arity {
                // The reference row splices the value in front of sibling `position`.
                let mut expected = siblings.clone();
                expected.insert(position, value);

                // The native row and the selection network both build it.
                let mut arith = NativeArith::new();
                assert_eq!(
                    arith.insert_at(&position, log_arity, value, &siblings),
                    expected
                );
                let bits = arith.index_bits(&position, log_arity);
                assert_eq!(
                    insert_by_bits(&mut arith, &bits, value, &siblings),
                    expected
                );
            }
        }
    }

    #[test]
    fn a_caller_schedule_naming_an_unusable_arity_is_rejected() {
        // Invariant: `fold_query` is public and takes the schedule from its caller.
        //
        // `verify_fri` derives its own from `fold_schedule`, so entries are sane there.
        //
        // A caller replaying one query supplies its own, and these three would have
        // reached a shift, a sum, or a fold that cannot make progress.
        //
        // Fixture state: three schedules, each unusable for its own reason.
        //
        //     [64]             ->  `1 << 64` overflows the shift
        //     [usize::MAX, 1]  ->  the total overflows the sum
        //     [0, 2]           ->  a round that folds nothing away

        // An entry at the word width cannot be shifted by.
        assert!(matches!(
            fold_query_with_schedule(&[usize::BITS as usize]),
            Err(FriError::FoldScheduleArityOutOfRange {
                round: 0,
                log_arity: 64
            } | FriError::FoldScheduleArityOutOfRange {
                round: 0,
                log_arity: 32
            })
        ));

        // A saturating entry is caught by the same bound, before the sum runs.
        assert!(matches!(
            fold_query_with_schedule(&[usize::MAX, 1]),
            Err(FriError::FoldScheduleArityOutOfRange { round: 0, .. })
        ));

        // A zero entry is a round that never descends.
        assert!(matches!(
            fold_query_with_schedule(&[0, 2]),
            Err(FriError::FoldScheduleArityOutOfRange {
                round: 0,
                log_arity: 0
            })
        ));

        // A schedule of usable entries that still folds too far is a different rejection.
        assert!(matches!(
            fold_query_with_schedule(&[4, 4]),
            Err(FriError::FoldScheduleTooLong { .. })
        ));
    }

    /// All the data needed to invoke the top-level FRI verification.
    struct TestFixture {
        /// Protocol parameters (blowup, arity, queries, etc.).
        fri_params: FriParameters<ChallengeMmcs>,
        /// Base-field commitment scheme used for input polynomials.
        input_mmcs: ValMmcs,
        /// A valid proof produced by a real prover run.
        proof: Proof,
        /// Fiat-Shamir challenger already advanced past the PCS phase,
        /// ready for the verification entry point.
        challenger: Challenger,
        /// The batching challenge that phase drew, which the PCS hands to FRI.
        alpha: Challenge,
        /// Commitment and opening-point data the verifier checks against.
        commitments_with_opening_points: Vec<
            CommitmentWithOpeningPoints<
                Challenge,
                <ValMmcs as Mmcs<Val>>::Commitment,
                TwoAdicMultiplicativeCoset<Val>,
            >,
        >,
    }

    /// Build the default fixture: the minimal instance with 2 queries.
    ///
    /// Two queries are the minimum for the arity-consistency checks.
    fn make_test_fixture() -> TestFixture {
        make_test_fixture_with_queries(2)
    }

    /// Build a deterministic, minimal test fixture with the given query count.
    ///
    /// Commits a single 8-row, 2-column trace, opens it at one
    /// random challenge point, and produces a valid FRI proof with:
    /// - binary folding (log arity 1)
    /// - blowup factor 2
    /// - zero proof-of-work bits
    ///
    /// # Proof Shape
    ///
    /// With degree 8 and blowup 2, the evaluation domain has 16 points.
    /// Binary folding halves the domain each round, producing 3 COMMIT
    /// rounds before reaching the final constant polynomial:
    ///
    /// ```text
    ///     Round 0: |L^(0)| = 16  -->  |L^(1)| = 8   (commit + fold)
    ///     Round 1: |L^(1)| = 8   -->  |L^(2)| = 4   (commit + fold)
    ///     Round 2: |L^(2)| = 4   -->  |L^(3)| = 2   (commit + fold)
    ///     Final:   1 coefficient (the constant polynomial)
    /// ```
    ///
    /// The resulting proof contains:
    /// - 3 commit-phase commitments (one per round)
    /// - 3 proof-of-work witnesses (one per round)
    /// - 3 commit-phase multi-openings (one per round, each covering every query)
    /// - 1 input batch multi-opening (covering every query)
    /// - 1 final polynomial coefficient
    ///
    /// The final round folds a 4-point codeword onto a 2-node layer.
    /// So any 3 or more queries must collide on one of those 2 nodes.
    ///
    /// The challenger is advanced past the opened-values observation,
    /// ready for the verification entry point.
    fn make_test_fixture_with_queries(num_queries: usize) -> TestFixture {
        make_test_fixture_with(num_queries, 1, 3)
    }

    /// Build a fixture with a chosen query count, folding cap, and trace height.
    ///
    /// A cap above one lets a single input fold non-uniformly.
    /// That is what separates two schedules sharing a height sum.
    fn make_test_fixture_with(
        num_queries: usize,
        max_log_arity: usize,
        log_degree: usize,
    ) -> TestFixture {
        // Use a fixed seed so every test run is deterministic.
        let mut rng = SmallRng::seed_from_u64(42);

        // Build the permutation, hash, and compression for the Merkle tree.
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());

        // - One commitment scheme for base-field inputs,
        // - Another for extension-field FRI commitments.
        let input_mmcs = ValMmcs::new(hash.clone(), compress.clone(), 0);
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress, 0));

        // Minimal parameters that exercise all verifier paths cheaply.
        //
        // - blowup 2: evaluation domain is 2x the polynomial degree,
        //   the smallest blowup for a sound protocol
        // - final poly length 1: fold down to f^(r) with degree 0
        // - binary folding: each round halves the domain (arity 2)
        // - queries: chosen by the caller
        // - 1 batch PoW bit: cheap to grind, but enough that the witness is
        //   actually observed and so genuinely bound to the transcript (a
        //   0-bit check short-circuits without reading the witness at all)
        // - 0 commit/query PoW bits: those witnesses are trivially valid, so
        //   tests that tamper with proof shape reach the check they target
        let fri_params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            max_log_arity,
            num_queries,
            batch_proof_of_work_bits: 1,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
            mmcs: challenge_mmcs,
        };

        // Wrap the parameters and commitment scheme into a PCS instance.
        let pcs = TwoAdicFriPcs::new(Radix2Dit::default(), input_mmcs.clone(), fri_params.clone());

        // Commit to a single 2-column trace matrix of the requested height.
        // With blowup 2 the evaluation domain is twice as tall.
        // The default of 8 rows gives log height 4, folded binary in 3 rounds.
        let width = 2;
        let domain = <TwoAdicFriPcs<Val, Radix2Dit<Val>, ValMmcs, ChallengeMmcs> as Pcs<
            Challenge,
            Challenger,
        >>::natural_domain_for_degree(&pcs, 1 << log_degree);
        let trace = RowMajorMatrix::<Val>::rand_nonzero(&mut rng, 1 << log_degree, width);

        // Produce the Merkle commitment to the low-degree-extended trace.
        let (commitment, prover_data) =
            <TwoAdicFriPcs<Val, Radix2Dit<Val>, ValMmcs, ChallengeMmcs> as Pcs<
                Challenge,
                Challenger,
            >>::commit(&pcs, [(domain, trace)])
            .unwrap();

        // Prover side:
        // Observe the commitment, sample an opening point, and produce the FRI proof.
        let mut p_challenger = Challenger::new(perm.clone());
        p_challenger.observe(&commitment);
        let zeta: Challenge = p_challenger.sample_algebra_element();
        let (opened_values, proof) = pcs
            .open(
                vec![p3_commit::OpeningRequest {
                    prover_data: &prover_data,
                    points: vec![vec![zeta]],
                }],
                &mut p_challenger,
            )
            .unwrap();

        // Verifier side:
        // Replay the transcript up to the point where the top-level FRI verification begins.
        let mut v_challenger = Challenger::new(perm);
        v_challenger.observe(&commitment);
        let v_zeta: Challenge = v_challenger.sample_algebra_element();
        assert_eq!(
            v_zeta, zeta,
            "prover and verifier must sample the same point"
        );

        // Assemble the commitment-with-opening-points structure that the
        // verifier checks the proof against.
        let cwop: Vec<CommitmentWithOpeningPoints<_, _, _>> = vec![
            (
                commitment,
                vec![(domain, vec![(zeta, opened_values[0][0][0].clone())])],
            )
                .into(),
        ];

        // Replay the PCS phase: bind the claims, replay the grind, draw the batching challenge.
        // This is the last transcript work before FRI verification begins.
        let alpha = replay_pcs_phase(
            &fri_params,
            &mut v_challenger,
            &cwop,
            proof.batch_pow_witness,
        );

        TestFixture {
            fri_params,
            input_mmcs,
            proof,
            challenger: v_challenger,
            alpha,
            commitments_with_opening_points: cwop,
        }
    }

    /// Advance a verifier challenger through the PCS phase and hand back its challenge.
    ///
    /// The driver seeds itself, replays the claims, replays the grind, and draws `alpha`.
    /// Its delegation bracket records no sponge operation.
    /// So the state left here is exactly the one the PCS lends to `verify_fri`.
    fn replay_pcs_phase(
        params: &FriParameters<ChallengeMmcs>,
        challenger: &mut Challenger,
        cwop: &[CommitmentWithOpeningPoints<
            Challenge,
            <ValMmcs as Mmcs<Val>>::Commitment,
            TwoAdicMultiplicativeCoset<Val>,
        >],
        batch_pow_witness: Val,
    ) -> Challenge {
        let mut transcript = PcsVerifierTranscript::<Challenger, Val, Challenge>::new(
            challenger,
            PcsShape::from_claims(params, cwop),
        );
        transcript
            .claimed_openings(cwop)
            .expect("the claims describe their own shape");
        let alpha = transcript
            .batch_phase(Some(batch_pow_witness))
            .expect("the fixture's witness meets the difficulty it was ground at");
        transcript.delegate(|_| ());
        transcript.finish();
        alpha
    }

    /// Convenience wrapper that constructs the folding strategy and
    /// invokes the top-level FRI verification.
    fn run_verify_fri(
        params: &FriParameters<ChallengeMmcs>,
        proof: &Proof,
        challenger: &mut Challenger,
        cwop: &[CommitmentWithOpeningPoints<
            Challenge,
            <ValMmcs as Mmcs<Val>>::Commitment,
            TwoAdicMultiplicativeCoset<Val>,
        >],
        input_mmcs: &ValMmcs,
        alpha: Challenge,
    ) -> Result<(), TestError> {
        let folding: Folding = TwoAdicFriFolding(PhantomData);
        verify_fri(
            &mut NativeArith::new(),
            &folding,
            params,
            proof,
            challenger,
            cwop,
            input_mmcs,
            alpha,
        )
    }

    #[test]
    fn colliding_queries_are_collapsed() {
        // The last round folds a 4-point codeword onto a 2-node layer.
        // Three queries into two nodes must collide by pigeonhole.
        // The shared node is opened once and reused by both queries.
        //
        // Fixture state (seed 42, 3 queries): sampled indices [2, 4, 11].
        // Final group index = index >> 3.
        //
        //     query 0 : idx 2  -> node 0
        //     query 1 : idx 4  -> node 0   (collides with query 0)
        //     query 2 : idx 11 -> node 1
        //
        // An honest proof must still verify.
        // The multiproof authenticates the shared node once.
        // Both colliding queries fold through it.
        let f = make_test_fixture_with_queries(3);
        let mut challenger = f.challenger.clone();
        let result = run_verify_fri(
            &f.fri_params,
            &f.proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        );
        assert!(
            result.is_ok(),
            "honest proof with colliding queries should pass: {result:?}"
        );
    }

    #[test]
    fn tampered_shared_node_is_rejected() {
        let f = make_test_fixture_with_queries(3);
        let mut proof = f.proof.clone();

        // Two queries collide on one node in the last round.
        // The multiproof opens that shared node once for both.
        // Tampering the shared node's values must still be caught.
        //
        // Fixture state (seed 42, 3 queries): sampled indices [2, 4, 11].
        // Last round folds a 4-point codeword onto a 2-node layer.
        //
        //     query 0 : idx 2  -> node 0, position 0
        //     query 1 : idx 4  -> node 0, position 1
        //     query 2 : idx 11 -> node 1
        //
        // Mutation: bump query 0's sibling in the shared node by one.
        //
        //     honest : the row folds to the committed final value
        //     bumped : the row folds to a different value
        //
        // The reconstructed row feeds the fold.
        // A tampered value changes the folded evaluation.
        // The final-polynomial check then rejects the proof.
        let last_round = proof.commit_phase_openings.len() - 1;
        proof.commit_phase_openings[last_round].sibling_values[0][0] += Challenge::ONE;

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("should reject a tampered shared node");

        // The fold arithmetic pins the shared node's values.
        // A collapsed opening grants no escape from the final-polynomial check.
        match err {
            FriError::FinalPolyMismatch => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn valid_proof_passes() {
        // Baseline: an unmodified proof must pass all checks.
        let f = make_test_fixture();
        let mut challenger = f.challenger.clone();
        let result = run_verify_fri(
            &f.fri_params,
            &f.proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        );
        assert!(result.is_ok(), "valid proof should pass: {result:?}");
    }

    #[test]
    fn commit_phase_openings_count_mismatch() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // In FRI, each COMMIT round produces one oracle f^(i). During the
        // QUERY phase, the verifier opens that oracle at every queried index,
        // all authenticated by the round's shared multi-opening.
        // So the proof must carry exactly one multi-opening per round.
        //
        // Fixture state: the claim fixes 3 rounds, so 3 multi-openings are expected.
        //
        // Mutation: append a duplicate multi-opening.
        //
        //     openings:     [round_0, round_1, round_2, EXTRA]
        //     commitments:  [round_0, round_1, round_2]
        //     → 4 != 3 → error
        let extra = proof.commit_phase_openings[0].clone();
        proof.commit_phase_openings.push(extra);

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("should reject mismatched commit-phase opening count");

        let expected_rounds = f.proof.commit_phase_commits.len();
        match err {
            FriError::CommitPhaseOpeningsCountMismatch { expected, got } => {
                assert_eq!(expected, expected_rounds);
                assert_eq!(got, expected_rounds + 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn matrix_without_opening_points() {
        let f = make_test_fixture();

        // A matrix's width feeds the authenticated MMCS dimensions. With opening
        // points it is pinned to the claimed evaluation count; with none it could
        // only be read back from the proof, so the verifier must reject instead.
        //
        // Mutation: strip the sole matrix's opening points, leaving it in the batch.
        //
        //     before: mats[0] points = [(zeta, values)]   → width pinned by claim
        //     after:  mats[0] points = []                 → no claim → reject
        let mut cwop = f.commitments_with_opening_points.clone();
        cwop[0].matrices[0].points = vec![];

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &f.proof,
            &mut challenger,
            &cwop,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("should reject a matrix opened at zero points");

        match err {
            FriError::MatrixWithoutOpeningPoints { batch, matrix } => {
                // The lone batch holds the lone matrix.
                assert_eq!(batch, 0);
                assert_eq!(matrix, 0);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn commit_phase_query_count_mismatch() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // Each round's multi-opening must open the codeword at every query:
        // one sibling row per query. The log-arity schedule lives once per
        // round, so a cross-query schedule mismatch is unrepresentable; the
        // remaining per-query shape to check is the round's query coverage.
        //
        // Fixture state: 2 queries → each round carries 2 sibling rows.
        //
        // Mutation: drop query 1's sibling row from round 0.
        //
        //     round 0 sibling rows: [q_0]        (was [q_0, q_1])
        //     → 1 != 2 → error at round 0
        proof.commit_phase_openings[0].sibling_values.pop();

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("should reject a round that does not open every query");

        match err {
            FriError::CommitPhaseQueryCountMismatch {
                round,
                expected,
                got,
            } => {
                assert_eq!(round, 0);
                assert_eq!(expected, f.fri_params.num_queries);
                assert_eq!(got, f.fri_params.num_queries - 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn commit_pow_witness_count_mismatch() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // Each COMMIT round includes a proof-of-work grinding step: the
        // prover finds a hash preimage satisfying a difficulty target.
        // There must be exactly one witness per round — one per commitment.
        //
        // Fixture state: 3 rounds → 3 commitments → expect 3 witnesses.
        //
        // Mutation: push a dummy witness → 4 witnesses vs 3 commitments.
        proof.commit_pow_witnesses.push(Val::ZERO);

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("should reject proof with extra PoW witness");

        let expected_count = f.proof.commit_phase_commits.len();
        match err {
            FriError::CommitPowWitnessCountMismatch { expected, got } => {
                assert_eq!(expected, expected_count);
                assert_eq!(got, expected_count + 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn final_poly_length_mismatch() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // After all COMMIT rounds, f^(r) should be a polynomial of degree < 2^log_final_poly_len.
        // The prover sends its coefficients.
        // In this fixture, log_final_poly_len = 0 → exactly 1 coefficient.
        //
        // Mutation: append a zero coefficient → 2 coefficients vs 1 expected.
        proof.final_poly.push(Challenge::ZERO);

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("should reject proof with wrong final polynomial length");

        let expected_len = f.fri_params.final_poly_len();
        match err {
            FriError::FinalPolyLengthMismatch { expected, got } => {
                assert_eq!(expected, expected_len);
                assert_eq!(got, expected_len + 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn input_openings_query_count_mismatch() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // The number of queries determines soundness: each independent
        // query multiplies the cheating prover's probability of escaping
        // detection. Every input batch must open all of them.
        //
        // Fixture state: num_queries = 2 → batch 0 carries 2 opened row sets.
        //
        // Mutation: pop query 1's row set from batch 0 → 1 vs 2 expected.
        proof.input_openings[0].opened_values.pop();

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("should reject an input batch that does not open every query");

        match err {
            FriError::InputOpeningsQueryCountMismatch {
                batch,
                expected,
                got,
            } => {
                assert_eq!(batch, 0);
                assert_eq!(expected, f.fri_params.num_queries);
                assert_eq!(got, f.fri_params.num_queries - 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn instance_with_nothing_committed_rejected() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // Invariant: an instance that commits to nothing proves nothing.
        //
        // Every derived guard compares against the committed heights.
        // With none, each would compare against nothing and pass.
        //
        // Fixture state: 1 committed polynomial.
        //
        // Mutation: clear input openings and commitment data.
        //
        //     input openings:    []   (was [batch_0])
        //     commitment data:   []   (was [(commit, mats)])
        proof.input_openings = vec![];
        let empty_cwop: Vec<
            CommitmentWithOpeningPoints<
                Challenge,
                <ValMmcs as Mmcs<Val>>::Commitment,
                TwoAdicMultiplicativeCoset<Val>,
            >,
        > = vec![];

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &empty_cwop,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("should reject proof with no committed polynomials");

        assert!(
            matches!(err, FriError::NoCommittedMatrices),
            "wrong error variant: {err:?}"
        );
    }

    #[test]
    fn sibling_values_length_mismatch() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // In each fold round the verifier reads the queried evaluation
        // plus (arity - 1) siblings. Binary folding → arity 2 → 1 sibling.
        // Siblings are not in the Fiat-Shamir transcript → safe to mutate.
        // The row shape is checked upfront for every query of every round.
        //
        // Fixture state: binary fold → expect 1 sibling per query per round.
        //
        // Mutation: push an extra sibling into query 0's row of round 0.
        //
        //     Before: sibling_values[0] = [s0]          (length 1 = arity - 1)
        //     After:  sibling_values[0] = [s0, ZERO]    (length 2 = arity)
        proof.commit_phase_openings[0].sibling_values[0].push(Challenge::ZERO);

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("should reject proof with wrong number of sibling values");

        match err {
            FriError::SiblingValuesLengthMismatch {
                round,
                expected,
                got,
            } => {
                // Binary fold: arity = 2, so expect 1 sibling, got 2.
                assert_eq!(round, 0);
                assert_eq!(expected, 1);
                assert_eq!(got, 2);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    /// Build a fixture whose proof folded on `forged` instead of the derived schedule.
    ///
    /// Every commitment, sibling row and opening proof agrees with `forged`.
    /// That is what a malicious prover would emit, not a field tamper.
    fn make_forged_schedule_fixture(
        num_queries: usize,
        max_log_arity: usize,
        log_degree: usize,
        forged: Vec<usize>,
    ) -> TestFixture {
        let mut rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());

        let input_mmcs = ValMmcs::new(hash.clone(), compress.clone(), 0);
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress, 0));

        let fri_params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            max_log_arity,
            num_queries,
            batch_proof_of_work_bits: 0,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
            mmcs: challenge_mmcs,
        };

        // The seam that makes this proof forged rather than merely tampered.
        let mut pcs =
            TwoAdicFriPcs::new(Radix2Dit::default(), input_mmcs.clone(), fri_params.clone());
        pcs.forged_fold_schedule = Some(forged);

        let width = 2;
        let domain = <TwoAdicFriPcs<Val, Radix2Dit<Val>, ValMmcs, ChallengeMmcs> as Pcs<
            Challenge,
            Challenger,
        >>::natural_domain_for_degree(&pcs, 1 << log_degree);
        let trace = RowMajorMatrix::<Val>::rand_nonzero(&mut rng, 1 << log_degree, width);

        let (commitment, prover_data) =
            <TwoAdicFriPcs<Val, Radix2Dit<Val>, ValMmcs, ChallengeMmcs> as Pcs<
                Challenge,
                Challenger,
            >>::commit(&pcs, [(domain, trace)])
            .unwrap();

        let mut p_challenger = Challenger::new(perm.clone());
        p_challenger.observe(&commitment);
        let zeta: Challenge = p_challenger.sample_algebra_element();
        let (opened_values, proof) = pcs
            .open(
                vec![p3_commit::OpeningRequest {
                    prover_data: &prover_data,
                    points: vec![vec![zeta]],
                }],
                &mut p_challenger,
            )
            .unwrap();

        let mut v_challenger = Challenger::new(perm);
        v_challenger.observe(&commitment);
        let _: Challenge = v_challenger.sample_algebra_element();

        let cwop: Vec<CommitmentWithOpeningPoints<_, _, _>> = vec![
            (
                commitment,
                vec![(domain, vec![(zeta, opened_values[0][0][0].clone())])],
            )
                .into(),
        ];
        let alpha = replay_pcs_phase(
            &fri_params,
            &mut v_challenger,
            &cwop,
            proof.batch_pow_witness,
        );

        TestFixture {
            fri_params,
            input_mmcs,
            proof,
            challenger: v_challenger,
            alpha,
            commitments_with_opening_points: cwop,
        }
    }

    #[test]
    fn forged_fold_schedule_rejected() {
        // Invariant: the folding schedule is derived from the claim.
        //
        // It is never read from the proof.
        //
        // Fixture state: one input at log height 5, final height 1, folding cap 3.
        //
        //     derived:  [3, 1]   folds 5 -> 2 -> 1
        //     forged:   [1, 3]   folds 5 -> 4 -> 1
        //
        // Both have two rounds.
        //
        // Both fold the same total distance.
        //
        // So a round count and a height sum agree on the two alike.
        //
        // The prover here really folded on the forgery.
        //
        // Every commitment, sibling row and opening proof agrees with it.
        //
        // The transcript it ground was seeded with it too.
        //
        // Mutation: the derived arity of round 0 is 8, the forged one is 2.
        //
        //     round 0 derived arity 2^3 = 8  ->  expects 7 siblings
        //     round 0 forged  arity 2^1 = 2  ->  carries  1 sibling
        //
        // Rejected before the transcript is even seeded.
        let f = make_forged_schedule_fixture(2, 3, 4, vec![1, 3]);

        // The forgery lives in the proof's geometry, not in a declared number.
        //
        // One sibling per query in round 0 is what folding by two produces.
        assert_eq!(
            f.proof.commit_phase_openings[0].sibling_values[0].len(),
            1,
            "the prover must have folded round 0 by two"
        );

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &f.proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("a forged fold schedule must be rejected");

        match err {
            FriError::SiblingValuesLengthMismatch {
                round,
                expected,
                got,
            } => {
                assert_eq!(round, 0);
                assert_eq!(expected, 7);
                assert_eq!(got, 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn honest_high_arity_proof_accepted() {
        // Invariant: the derivation accepts what an honest prover folds.
        //
        // Guards the forgery test above from passing for the wrong reason.
        let f = make_test_fixture_with(2, 3, 4);
        let mut challenger = f.challenger.clone();
        run_verify_fri(
            &f.fri_params,
            &f.proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect("an honest non-uniform schedule must verify");
    }

    #[test]
    fn a_uniformly_shortened_commit_phase_is_rejected() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // Invariant: the round count is a length of the described transcript.
        //
        // Fixture state: one input matrix at log height 4, final height 1, cap 1.
        //
        //     one bit folded per round  ->  the claim fixes [1, 1, 1]
        //
        // Mutation: drop the last round from all three per-round lists.
        //
        //     described:  [round_0, round_1, round_2]
        //     proof:      [round_0, round_1]
        //
        //     2 != 3
        //
        // The openings list is checked against the schedule first.
        //
        // So that is where a uniformly shortened proof stops.
        proof.commit_phase_commits.pop();
        proof.commit_pow_witnesses.pop();
        proof.commit_phase_openings.pop();

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("a short commit phase must be rejected");

        match err {
            FriError::CommitPhaseOpeningsCountMismatch { expected, got } => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn an_extra_commit_round_is_rejected() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // Invariant: an extra commit round is a shape the description does not fix.
        //
        // The description is what rejects it.
        //
        // Fixture state: the claim fixes a schedule of [1, 1, 1], so 3 rounds.
        //
        // Mutation: clone the last round into the commitment and witness lists.
        //
        // The openings list is left at the described length.
        //
        //     openings:     [r_0, r_1, r_2]        ->  3 == 3, accepted
        //     commitments:  [r_0, r_1, r_2, r_2]   ->  4 != 3, rejected
        //
        // A fourth replay would drive the player past the end of the description.
        //
        // The driver reports that by panicking.
        //
        // Reaching a structured error instead is the point of this test.
        let extra_commit = proof.commit_phase_commits.last().unwrap().clone();
        proof.commit_phase_commits.push(extra_commit);

        let extra_witness = *proof.commit_pow_witnesses.last().unwrap();
        proof.commit_pow_witnesses.push(extra_witness);

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("an extra commit round must be rejected");

        match err {
            FriError::CommitRoundCountMismatch { expected, got } => {
                assert_eq!(expected, 3);
                assert_eq!(got, 4);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn global_max_height_exceeds_two_adicity() {
        // Invariant: the global evaluation height must have a root of unity here.
        //
        // The query phase evaluates the final polynomial at a root of that order.
        //
        // Asking for one past the two-adicity panics.
        //
        // This guard runs before the height is used, so it rejects instead.
        //
        // Fixture state: one claimed matrix over a domain of log size 3, blowup 1.
        //
        //     H = 3 + 1 = 4, well inside a two-adicity of 27
        //
        // Mutation: restate the claim over the tallest domain the field admits.
        //
        //     claimed log size:  3  ->  TWO_ADICITY
        //     H = TWO_ADICITY + 1   ->  past the two-adicity
        //
        // The height is derived from the claim.
        //
        // So restating the claim is the only way to inflate it.
        let f = make_test_fixture();
        let mut cwop = f.commitments_with_opening_points.clone();

        // The tallest coset the field's two-adic subgroup contains.
        //
        // Nothing is evaluated over it, so its size costs no memory here.
        cwop[0].matrices[0].domain =
            TwoAdicMultiplicativeCoset::new(Val::ONE, Val::TWO_ADICITY).expect("coset exists");

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &f.proof,
            &mut challenger,
            &cwop,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("height above two-adicity must be rejected");

        match err {
            FriError::GlobalMaxHeightTooLarge {
                log_global_max_height,
                two_adicity,
            } => {
                assert_eq!(two_adicity, Val::TWO_ADICITY);
                assert_eq!(log_global_max_height, Val::TWO_ADICITY + 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn input_proof_batch_count_mismatch() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // Each batch of committed polynomials needs exactly one Merkle
        // multi-opening. This check fires before any cryptographic work.
        //
        // Fixture state: 1 batch commitment → expect 1 batch multi-opening.
        //
        // Mutation: push a duplicate batch multi-opening.
        //
        //     batch openings:  [batch_0, EXTRA]
        //     commitments:     [commit_0]
        //     → 2 != 1 → error
        let extra_batch = proof.input_openings[0].clone();
        proof.input_openings.push(extra_batch);

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("should reject proof with extra input batch");

        match err {
            FriError::InputProofBatchCountMismatch { expected, got } => {
                assert_eq!(expected, 1);
                assert_eq!(got, 2);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn batch_opened_values_count_mismatch() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // Each committed matrix needs exactly one row of opened column
        // values per query. This check fires before Merkle verification.
        //
        // Fixture state: 1 matrix per batch → expect 1 opened-values entry
        // in every query's row set.
        //
        // Mutation: pop query 0's only opened-values entry.
        //
        //     query 0 opened_values:  []           (was [row_0])
        //     matrices:               [matrix_0]
        //     → 0 != 1 → error
        proof.input_openings[0].opened_values[0].pop();

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("should reject proof with missing opened-values entry");

        match err {
            FriError::BatchOpenedValuesCountMismatch {
                batch,
                expected,
                got,
            } => {
                assert_eq!(batch, 0);
                assert_eq!(expected, 1);
                assert_eq!(got, 0);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn point_evaluation_count_mismatch() {
        let f = make_test_fixture();
        let mut cwop = f.commitments_with_opening_points.clone();

        // The verifier pairs opened column values f_i(x) with claimed
        // evaluations f_i(z) to form (f_i(z) - f_i(x)) / (z - x).
        // These two vectors must have the same length.
        //
        // Fixture state: 2 trace columns → 2 opened values, 2 claims at point 0.
        //
        // Mutation: add a second opening point with one claim too many.
        // Point 0 still matches the matrix width, so the Merkle shape
        // checks pass and the pairing check fires at point 1. Claims are
        // not in the transcript, so the original challenger is reused.
        //
        //     opened values:   [f_0(x), f_1(x)]            (length 2)
        //     point 0 claims:  [f_0(z), f_1(z)]            (length 2)
        //     point 1 claims:  [0, 0, 0]                   (length 3)
        //     → 2 != 3 → error at point 1
        cwop[0].matrices[0].points.push(PointOpening {
            point: Challenge::ZERO,
            values: vec![Challenge::ZERO; 3],
        });

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &f.proof,
            &mut challenger,
            &cwop,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("should reject proof with extra claimed evaluation");

        match err {
            FriError::PointEvaluationCountMismatch {
                batch,
                matrix,
                point,
                expected,
                got,
            } => {
                assert_eq!(batch, 0);
                assert_eq!(matrix, 0);
                assert_eq!(point, 1);
                // 2 trace columns opened, but 3 claimed evaluations.
                assert_eq!(expected, 2);
                assert_eq!(got, 3);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn claimed_width_mismatch_rejected_by_input_mmcs() {
        let f = make_test_fixture();
        let mut cwop = f.commitments_with_opening_points.clone();

        // The matrix width the verifier passes to the input commitment
        // scheme comes from the first point's claimed evaluations.
        //
        //     opened values:  [f_0(x), f_1(x)]             (length 2)
        //     claims:         [f_0(z), f_1(z), EXTRA]      (length 3)
        //     → expected width 3, opened row has 2 → error
        cwop[0].matrices[0].points[0].values.push(Challenge::ZERO);

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &f.proof,
            &mut challenger,
            &cwop,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("should reject opened rows narrower than the claimed width");

        assert!(matches!(
            err,
            FriError::InputError(MerkleTreeError::WrongWidth {
                matrix: 0,
                expected: 3,
                got: 2,
            })
        ));
    }

    #[test]
    fn initial_reduced_opening_height_mismatch() {
        let folding = TwoAdicFriFolding::<(), ()>(PhantomData);

        // The fold chain starts at the global maximum domain height.
        // Its seed (the first reduced opening) must live at that height.
        //
        //     global max height = 5   (domain |L^(0)| = 2^5 = 32)
        //     opening height    = 3   (domain |L^(?)| = 2^3 = 8)
        //     → mismatch: 5 != 3
        let log_global_max_height = 5;
        let wrong_height = 3;
        let reduced_openings: FriOpenings<Challenge> =
            vec![(wrong_height, Challenge::from(Val::from_u8(7)))];

        // Empty fold-data: the error fires before any folding.
        let betas: Vec<Challenge> = vec![];
        let log_arities: Vec<usize> = vec![];
        let siblings: Vec<Vec<Challenge>> = vec![];

        let log_final_height = 1;

        let err = fold_query::<NativeArith, TwoAdicFriFolding<(), ()>, ChallengeMmcsError, ()>(
            &mut NativeArith::new(),
            &folding,
            &0,
            &betas,
            &log_arities,
            &siblings,
            reduced_openings,
            log_global_max_height,
            log_final_height,
        )
        .expect_err("should reject opening at wrong initial height");

        match err {
            FriError::InitialReducedOpeningHeightMismatch { expected, got } => {
                assert_eq!(expected, log_global_max_height);
                assert_eq!(got, wrong_height);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn final_fold_height_mismatch() {
        let folding = TwoAdicFriFolding::<(), ()>(PhantomData);

        // After all COMMIT rounds, the domain must have been folded down to
        // exactly the final height. This defense-in-depth check guards
        // against implementation bugs.
        //
        //     global max height = 5   → fold chain starts at 2^5
        //     final height      = 1   → fold chain should end at 2^1
        //     fold rounds       = 0   → no folding happens
        //     → current height = 5 != final height 1 → error
        let log_global_max_height = 5;
        let log_final_height = 1;
        let reduced_openings: FriOpenings<Challenge> =
            vec![(log_global_max_height, Challenge::from(Val::from_u8(42)))];

        // No fold rounds: jump straight to the final-height check.
        let betas: Vec<Challenge> = vec![];
        let log_arities: Vec<usize> = vec![];
        let siblings: Vec<Vec<Challenge>> = vec![];

        let err = fold_query::<NativeArith, TwoAdicFriFolding<(), ()>, ChallengeMmcsError, ()>(
            &mut NativeArith::new(),
            &folding,
            &0,
            &betas,
            &log_arities,
            &siblings,
            reduced_openings,
            log_global_max_height,
            log_final_height,
        )
        .expect_err("should reject when final height is not reached");

        match err {
            FriError::FinalFoldHeightMismatch { expected, got } => {
                assert_eq!(expected, log_final_height);
                assert_eq!(got, log_global_max_height);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn unconsumed_reduced_openings() {
        let folding = TwoAdicFriFolding::<(), ()>(PhantomData);

        // The fold chain rolls in each committed polynomial at the round
        // where the domain shrinks to that polynomial's height. After all
        // rounds, every opening must have been consumed.
        //
        // Fixture state: global max = final = 5 → zero fold rounds needed.
        //
        // Mutation: provide an extra opening at height 3 that no round visits.
        //
        //     reduced_openings = [(height=5, v1), (height=3, v2)]
        //     fold rounds       = 0 → height 3 is never reached
        //     → opening at height 3 left over → error
        let log_global_max_height = 5;
        let log_final_height = 5;
        let reduced_openings: FriOpenings<Challenge> = vec![
            // Seed at the max height — consumed immediately.
            (log_global_max_height, Challenge::from(Val::from_u8(42))),
            // Extra opening at height 3 — never reached, becomes leftover.
            (3, Challenge::from(Val::from_u8(99))),
        ];

        // No fold rounds: the leftover is detected right after the loop.
        let betas: Vec<Challenge> = vec![];
        let log_arities: Vec<usize> = vec![];
        let siblings: Vec<Vec<Challenge>> = vec![];

        let err = fold_query::<NativeArith, TwoAdicFriFolding<(), ()>, ChallengeMmcsError, ()>(
            &mut NativeArith::new(),
            &folding,
            &0,
            &betas,
            &log_arities,
            &siblings,
            reduced_openings,
            log_global_max_height,
            log_final_height,
        )
        .expect_err("should reject proof with leftover reduced openings");

        match err {
            FriError::UnconsumedReducedOpenings {
                next_log_height,
                remaining,
            } => {
                assert_eq!(next_log_height, 3);
                assert_eq!(remaining, 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn invalid_pow_witness() {
        let f = make_test_fixture();

        // Before each COMMIT round's folding challenge is sampled, the
        // verifier checks a proof-of-work witness against a difficulty
        // target. Witnesses that don't satisfy the target are rejected.
        let mut params = f.fri_params.clone();
        params.commit_proof_of_work_bits = 20;

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &params,
            &f.proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("should reject proof with invalid PoW witness");

        match err {
            FriError::InvalidPowWitness(PowPhase::CommitPhase) => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn noncanonical_pow_witnesses_at_zero_difficulty_are_rejected() {
        // Invariant: at zero difficulty only a canonical-value check binds the witness.
        //
        //     bits = 0 -> check_witness returns true without absorbing -> unbound
        //     bits > 0 -> absorbed, bits resampled                     -> the grind binds
        //
        // The commit-phase mutation moves one element and leaves the length alone: the
        // length has its own rejection, and a count check is not a value check.
        let f = make_test_fixture();
        assert_eq!(f.fri_params.commit_proof_of_work_bits, 0);
        assert_eq!(f.fri_params.query_proof_of_work_bits, 0);

        // The honest prover writes zero into every slot it pays no work for.
        assert!(f.proof.commit_pow_witnesses.iter().all(|w| *w == Val::ZERO));
        assert_eq!(f.proof.query_pow_witness, Val::ZERO);

        // One commit round's witness, with the vector's length left untouched.
        let mut proof = f.proof.clone();
        proof.commit_pow_witnesses[0] = Val::ONE;
        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("a rewritten commit-phase witness must be rejected");
        match err {
            FriError::NonCanonicalPowWitness {
                phase: PowPhase::CommitPhase,
            } => {}
            other => panic!("wrong error variant: {other:?}"),
        }

        // The query witness, which is a scalar rather than a vector entry.
        let mut proof = f.proof.clone();
        proof.query_pow_witness = Val::ONE;
        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("a rewritten query witness must be rejected");
        match err {
            FriError::NonCanonicalPowWitness {
                phase: PowPhase::Query,
            } => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn commit_phase_mmcs_error() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // Each fold round verifies one shared Merkle multi-opening for the
        // committed oracle f^(i). A corrupted sibling digest makes the
        // recomputed root diverge from the commitment.
        //
        // Merkle proofs are NOT in the Fiat-Shamir transcript → safe to
        // mutate without desyncing the challenger.
        proof.commit_phase_openings[0].opening_proof.sibling_hashes[0] = Default::default();

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("should reject proof with corrupted commit-phase Merkle proof");

        match err {
            FriError::CommitPhaseMmcsError(_) => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn input_error() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // Before the fold chain, the verifier checks each input batch's
        // shared Merkle multi-opening. A corrupted sibling digest makes
        // the recomputed root diverge from the input commitment.
        //
        // Input Merkle proofs are NOT in the Fiat-Shamir transcript →
        // safe to mutate without desyncing the challenger.
        proof.input_openings[0].opening_proof.sibling_hashes[0] = Default::default();

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("should reject proof with corrupted input Merkle proof");

        match err {
            FriError::InputError(_) => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn final_poly_mismatch() {
        let mut f = make_test_fixture();
        let proof = &mut f.proof;

        // After the fold chain, the verifier evaluates the final polynomial
        // at the folded point (Horner's method) and compares with the fold
        // chain's output. If the coefficients are wrong, the two disagree.
        //
        // The final polynomial is observed into the Fiat-Shamir transcript
        // before queries are sampled, so corrupting it desyncs the challenger
        // and causes Merkle failures before this check is reached. Because
        // this error path cannot be triggered through the full pipeline, we
        // test the underlying Horner evaluation in isolation.

        // Evaluate the honest polynomial at an arbitrary point.
        let x = Val::TWO;
        let honest_eval: Challenge = proof.final_poly.iter().copied().horner(x);

        // Corrupt the constant coefficient.
        proof.final_poly[0] += Challenge::ONE;

        // Evaluate the corrupted polynomial at the same point.
        let corrupted_eval: Challenge = proof.final_poly.iter().copied().horner(x);

        // The two must differ — this is what the verifier would catch.
        assert_ne!(honest_eval, corrupted_eval);
    }

    #[test]
    fn rejects_with_zero_queries() {
        // Invariant: a zero-query FRI instance proves nothing.
        // The per-query loop never runs.
        // Without the guard the verifier returns Ok for any final polynomial.
        //
        // Fixture state: an honest proof built with 2 queries.
        //
        // Mutation: verify it under params with num_queries = 0.
        let f = make_test_fixture();
        let mut params = f.fri_params.clone();
        params.num_queries = 0;

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &params,
            &f.proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("zero-query instance must be rejected");

        assert!(
            matches!(err, FriError::ZeroQueries),
            "expected ZeroQueries, got {err:?}"
        );
    }

    #[test]
    fn rejects_with_zero_blowup() {
        // Invariant: at log_blowup = 0 every length-N word is a degree-<N
        // codeword, so the folding chain accepts an arbitrary word.
        //
        // Fixture state: an honest proof built with log_blowup >= 1.
        //
        // Mutation: verify it under params with log_blowup = 0.
        let f = make_test_fixture();
        let mut params = f.fri_params.clone();
        params.log_blowup = 0;

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &params,
            &f.proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("zero-blowup instance must be rejected");

        assert!(
            matches!(err, FriError::ZeroBlowup),
            "expected ZeroBlowup, got {err:?}"
        );
    }

    #[test]
    fn rejects_with_zero_folding_arity() {
        // Invariant: the verifier rejects a degenerate configuration.
        //
        // It rejects rather than panicking on one.
        //
        // A cap of zero would make every round fold by zero bits.
        //
        //     cap 1:  4 -> 3 -> 2 -> 1     reaches the final height
        //     cap 0:  no round makes progress
        //
        // `fold_schedule` does not run such a walk: `compute_log_arity_for_round`
        // asserts a positive cap, so the guard here is what stands between a zero
        // cap and that assert.
        //
        // Fixture state: an honest proof built with a folding cap of 1.
        //
        // Mutation: verify it under params with a cap of 0.
        let f = make_test_fixture();
        let mut params = f.fri_params.clone();
        params.max_log_arity = 0;

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &params,
            &f.proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
            f.alpha,
        )
        .expect_err("a zero folding cap must be rejected");

        assert!(
            matches!(err, FriError::ZeroFoldingArity),
            "expected ZeroFoldingArity, got {err:?}"
        );
    }

    #[test]
    #[should_panic(expected = "num_queries must be at least 1")]
    fn prover_rejects_zero_queries() {
        // The prover must refuse to build a vacuous proof.
        // The verifier guards the same config, so the failure is symmetric.
        let mut rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let input_mmcs = ValMmcs::new(hash.clone(), compress.clone(), 0);
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress, 0));

        // Zero queries; every other parameter is otherwise valid.
        let fri_params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 0,
            batch_proof_of_work_bits: 0,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
            mmcs: challenge_mmcs,
        };
        let pcs = TwoAdicFriPcs::new(Radix2Dit::default(), input_mmcs, fri_params);

        // Commit succeeds; the assert fires inside the opening (FRI prover).
        let log_degree = 3;
        let domain = <TwoAdicFriPcs<Val, Radix2Dit<Val>, ValMmcs, ChallengeMmcs> as Pcs<
            Challenge,
            Challenger,
        >>::natural_domain_for_degree(&pcs, 1 << log_degree);
        let trace = RowMajorMatrix::<Val>::rand_nonzero(&mut rng, 1 << log_degree, 2);
        let (commitment, prover_data) =
            <TwoAdicFriPcs<Val, Radix2Dit<Val>, ValMmcs, ChallengeMmcs> as Pcs<
                Challenge,
                Challenger,
            >>::commit(&pcs, [(domain, trace)])
            .unwrap();

        let mut challenger = Challenger::new(perm);
        challenger.observe(&commitment);
        let zeta: Challenge = challenger.sample_algebra_element();
        let _ = pcs
            .open(
                vec![p3_commit::OpeningRequest {
                    prover_data: &prover_data,
                    points: vec![vec![zeta]],
                }],
                &mut challenger,
            )
            .unwrap();
    }

    #[test]
    fn opening_point_equal_to_query_point_is_rejected() {
        // open_inputs forms (f(z) - f(x)) / (z - x) for each opening point z, where x is
        // the query point derived from the index. At index 0 with no height reduction,
        //     x = GENERATOR * g^reverse_bits(0) = GENERATOR.
        // Setting z = GENERATOR makes the denominator zero.
        // Without the guard the inverse of zero panics; with it we get a typed error.
        let f = make_test_fixture();

        // Commit one width-1 matrix at the global max height, then open it at index 0.
        // Heights: domain 2^3, blowup 2^1, so the committed LDE height is 2^4 = 16.
        let log_blowup = f.fri_params.log_blowup;
        let log_domain = 3;
        let log_global_max_height = log_domain + log_blowup;
        let domain = TwoAdicMultiplicativeCoset::new(Val::ONE, log_domain).unwrap();
        let height = 1 << log_global_max_height;

        let mut rng = SmallRng::seed_from_u64(7);
        let mat = RowMajorMatrix::<Val>::rand_nonzero(&mut rng, height, 1);
        let (commit, prover_data) = f.input_mmcs.commit(vec![mat]);
        let (opened_values, opening_proof) = f.input_mmcs.open_multi_batch(&[0], &prover_data);
        let input_openings = vec![BatchMultiOpening {
            opened_values,
            opening_proof,
        }];

        // One opening point placed exactly on the query point x = GENERATOR.
        //     opening point z : GENERATOR   (claimed value is irrelevant; the denominator fails first)
        //     query point   x : GENERATOR
        let z = Challenge::from(Val::GENERATOR);
        let cwop: Vec<CommitmentWithOpeningPoints<_, _, _>> =
            vec![(commit, vec![(domain, vec![(z, vec![Challenge::ZERO])])]).into()];

        let err = open_inputs::<NativeArith, ValMmcs, ChallengeMmcsError>(
            &mut NativeArith::new(),
            f.fri_params.log_blowup,
            log_global_max_height,
            &[0],
            &input_openings,
            Challenge::ONE,
            &f.input_mmcs,
            &cwop,
        )
        .expect_err("opening point equal to the query point must be rejected");

        match err {
            FriError::OpeningPointMatchesQueryPoint {
                batch,
                matrix,
                point,
            } => {
                assert_eq!((batch, matrix, point), (0, 0, 0));
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }
}
