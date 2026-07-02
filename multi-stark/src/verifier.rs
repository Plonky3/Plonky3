//! Verify a multilinear AIR SNARK against a trace commitment.

use core::fmt::{self, Debug, Display, Formatter};

use p3_air::{Air, AirLayout, BaseAir, SymbolicAirBuilder};
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_sumcheck::PrescribedPointPcs;

use crate::config::{Commitment, MultiStarkConfig, PcsError};
use crate::folder::MultilinearFolder;
use crate::keys::VerifyingKey;
use crate::opening::TableOpening;
use crate::proof::{MultiStarkProof, single_table_protocol};
use crate::zerocheck::{AirZerocheck, ZerocheckError};

/// Reasons the multilinear AIR verifier rejects a proof.
#[derive(Debug)]
pub enum VerificationError<E> {
    /// The zerocheck reduction or its closing constraint check failed.
    Zerocheck(ZerocheckError),
    /// The commitment opening failed to verify.
    Opening(E),
    /// The verifying key expects a preprocessed opening, but the proof carries none.
    MissingPreprocessedOpening,
    /// The proof carries a preprocessed opening, but the verifying key expects none.
    UnexpectedPreprocessedOpening,
    /// The preprocessed key height disagrees with the proof's trace height.
    PreprocessedHeightMismatch {
        /// Trace arity the preprocessed commitment was built for.
        expected: usize,
        /// Trace arity the proof's sumcheck fixes.
        actual: usize,
    },
}

impl<E: Debug> Display for VerificationError<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zerocheck(e) => write!(f, "zerocheck: {e}"),
            Self::Opening(e) => write!(f, "opening: {e:?}"),
            Self::MissingPreprocessedOpening => {
                write!(f, "preprocessed opening expected but absent")
            }
            Self::UnexpectedPreprocessedOpening => {
                write!(f, "preprocessed opening present but not expected")
            }
            Self::PreprocessedHeightMismatch { expected, actual } => write!(
                f,
                "preprocessed height mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl<E: Debug> core::error::Error for VerificationError<E> {}

/// Verify a complete multilinear AIR proof.
///
/// The verifier replays the prover's transcript in the same order:
///
/// ```text
///     1. absorb preprocessed commitment (if any)
///     2. absorb main commitment
///     3. verify zerocheck sumcheck -> bound point r, reduced sum
///     4. open main at r            -> main values bound to the main commitment
///     5. open preprocessed at r    -> preprocessed values bound to its commitment
///     6. recompute g at r          -> match the reduced sum
/// ```
///
/// # Soundness
///
/// - Opened values come from the commitment proofs.
/// - The proof body never supplies those values directly.
/// - The closing check therefore uses committed trace values.
/// - The point is the bound point returned by zerocheck.
///
/// # Arguments
///
/// - Proof configuration selecting the commitment schemes.
/// - Verifying key carrying the reusable preprocessed commitment.
/// - AIR whose constraints are checked.
/// - Proof to verify.
/// - Public inputs forwarded to the AIR.
/// - Grinding difficulty per sumcheck round.
/// - Fiat-Shamir transcript.
///
/// # Errors
///
/// Returns an error when the sumcheck fails.
/// Returns an error when the closing check fails.
/// Returns an error when either commitment opening fails.
/// Returns an error when the proof and key disagree on the preprocessed trace.
///
/// # Panics
///
/// Panics if the AIR declares periodic columns.
pub fn verify<C, A>(
    config: &C,
    verifying_key: &VerifyingKey<C>,
    air: &A,
    proof: &MultiStarkProof<C>,
    public_values: &[C::Val],
    pow_bits: usize,
    challenger: &mut C::Challenger,
) -> Result<(), VerificationError<PcsError<C>>>
where
    C: MultiStarkConfig,
    C::Pcs: PrescribedPointPcs<C::Challenge, C::Challenger>,
    C::Challenger: FieldChallenger<C::Val>
        + GrindingChallenger<Witness = C::Val>
        + CanSampleUniformBits<C::Val>
        + CanObserve<Commitment<C>>,
    Commitment<C>: Clone,
    A: for<'b> Air<MultilinearFolder<'b, C::Val, C::Challenge, C::Challenge>>
        + Air<SymbolicAirBuilder<C::Val, C::Challenge>>
        + BaseAir<C::Val>,
{
    // The sumcheck has one round per trace variable, so it fixes the trace arity.
    let log_height = proof.sumcheck.num_rounds();
    let width = AirLayout::from_air::<C::Val>(air).main_width;
    let next_columns = air.main_next_row_columns();
    let preprocessed = verifying_key.preprocessed.as_ref();

    // The proof's preprocessed opening must match what the key expects.
    match (preprocessed, &proof.preprocessed_opening) {
        (Some(_), None) => return Err(VerificationError::MissingPreprocessedOpening),
        (None, Some(_)) => return Err(VerificationError::UnexpectedPreprocessedOpening),
        _ => {}
    }

    // The preprocessed commitment was built for a fixed height.
    // That height must match the height this proof fixes.
    if let Some(preprocessed) = preprocessed
        && preprocessed.log_height != log_height
    {
        return Err(VerificationError::PreprocessedHeightMismatch {
            expected: preprocessed.log_height,
            actual: log_height,
        });
    }

    // 1. Absorb the preprocessed commitment, matching the prover's first absorption.
    if let Some(preprocessed) = preprocessed {
        challenger.observe(preprocessed.commitment.clone());
    }

    // 2. Absorb the main commitment, matching the prover's commit phase.
    challenger.observe(proof.commitment.clone());

    // 3. Verify the zerocheck sumcheck, yielding the bound point and the reduced sum.
    let zerocheck = AirZerocheck::new(air, pow_bits);
    let reduction = zerocheck
        .verify_reduction::<C::Val, C::Challenge, _>(
            &proof.sumcheck,
            log_height,
            public_values,
            challenger,
        )
        .map_err(VerificationError::Zerocheck)?;

    // 4. Open the committed main columns at the bound point.
    // The returned values are bound to the main commitment.
    let protocol = single_table_protocol(log_height, width, &next_columns);
    let main_evals = config
        .pcs()
        .verify_at(
            &proof.commitment,
            &proof.opening,
            &protocol,
            core::slice::from_ref(&reduction.point),
            challenger,
        )
        .map_err(VerificationError::Opening)?;

    // Single table, single point: one batch of current values then successor values.
    let main = &main_evals[0];

    // 5. Open the preprocessed columns at the same point against the reused commitment.
    // The owned batch is bound to a local so the closing check can borrow its values.
    let preprocessed_evals = match (preprocessed, &proof.preprocessed_opening) {
        (Some(preprocessed), Some(opening)) => {
            let protocol = single_table_protocol(
                preprocessed.log_height,
                preprocessed.width,
                &preprocessed.next_columns,
            );
            let evals = config
                .preprocessed_pcs()
                .verify_at(
                    &preprocessed.commitment,
                    opening,
                    &protocol,
                    core::slice::from_ref(&reduction.point),
                    challenger,
                )
                .map_err(VerificationError::Opening)?;
            Some(evals)
        }
        _ => None,
    };

    // Build the preprocessed table view, borrowing the opened values.
    // The view is empty when the AIR declares no preprocessed trace.
    let preprocessed_opening = match (preprocessed, &preprocessed_evals) {
        (Some(preprocessed), Some(evals)) => {
            let batch = &evals[0];
            TableOpening::new(batch.current(), &preprocessed.next_columns, batch.next())
        }
        _ => TableOpening::empty(),
    };

    // 6. Close the zerocheck: recompute g from the commitment-bound values and match the sum.
    zerocheck
        .check_constraint(
            &reduction,
            TableOpening::new(main.current(), &next_columns, main.next()),
            preprocessed_opening,
            public_values,
        )
        .map_err(VerificationError::Zerocheck)
}
