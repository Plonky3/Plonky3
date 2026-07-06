//! Verify a multilinear AIR SNARK against a trace commitment.

use core::fmt::{self, Debug, Display, Formatter};

use p3_air::{Air, AirLayout, BaseAir, SymbolicAirBuilder};
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_sumcheck::PrescribedPointPcs;

use crate::config::{Commitment, MultiStarkConfig, PcsError};
use crate::folder::MultilinearFolder;
use crate::proof::{MultiStarkProof, single_table_protocol};
use crate::zerocheck::{AirZerocheck, ZerocheckError};

/// Reasons the multilinear AIR verifier rejects a proof.
#[derive(Debug)]
pub enum VerificationError<E> {
    /// The zerocheck reduction or its closing constraint check failed.
    Zerocheck(ZerocheckError),
    /// The commitment opening failed to verify.
    Opening(E),
}

impl<E: Debug> Display for VerificationError<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zerocheck(e) => write!(f, "zerocheck: {e}"),
            Self::Opening(e) => write!(f, "opening: {e:?}"),
        }
    }
}

impl<E: Debug> core::error::Error for VerificationError<E> {}

/// Verify a complete multilinear AIR proof.
///
/// The verifier replays the prover's transcript in the same order:
///
/// ```text
///     1. absorb commitment
///     2. verify zerocheck sumcheck -> bound point r, reduced sum
///     3. open columns at r         -> opened values bound to the commitment
///     4. recompute g at r          -> match the reduced sum
/// ```
///
/// # Soundness
///
/// - Opened values come from the commitment proof.
/// - The proof body never supplies those values directly.
/// - The closing check therefore uses committed trace values.
/// - The point is the bound point returned by zerocheck.
///
/// # Arguments
///
/// - Proof configuration selecting the commitment scheme.
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
/// Returns an error when the commitment opening fails.
///
/// # Panics
///
/// Panics if the AIR declares preprocessed columns.
/// Panics if the AIR declares periodic columns.
pub fn verify<C, A>(
    config: &C,
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
    A: for<'b> Air<MultilinearFolder<'b, C::Val, C::Challenge, C::Challenge>>
        + Air<SymbolicAirBuilder<C::Val, C::Challenge>>
        + BaseAir<C::Val>,
{
    // The sumcheck has one round per trace variable, so it fixes the trace arity.
    let log_height = proof.sumcheck.num_rounds();
    let width = AirLayout::from_air::<C::Val>(air).main_width;
    let next_columns = air.main_next_row_columns();

    // 1. Absorb the commitment; the prover absorbed it during the commit phase.
    challenger.observe(proof.commitment.clone());

    // 2. Verify the zerocheck sumcheck, yielding the bound point and the reduced sum.
    let zerocheck = AirZerocheck::new(air, pow_bits);
    let reduction = zerocheck
        .verify_reduction::<C::Val, C::Challenge, _>(
            &proof.sumcheck,
            log_height,
            public_values,
            challenger,
        )
        .map_err(VerificationError::Zerocheck)?;

    // 3. Open the committed columns at the bound point; the returned values are commitment-bound.
    let protocol = single_table_protocol(log_height, width, &next_columns);
    let evals = config
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
    let batch = &evals[0];

    // 4. Close the zerocheck: recompute g from the commitment-bound values and match the sum.
    zerocheck
        .check_constraint(
            &reduction,
            batch.current(),
            &next_columns,
            batch.next(),
            public_values,
        )
        .map_err(VerificationError::Zerocheck)
}
