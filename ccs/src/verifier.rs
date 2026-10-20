//! Verifier of the two-phase reduction, leaving one witness evaluation claim.

use alloc::vec::Vec;

use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, dot_product};
use p3_multilinear_util::point::Point;
use tracing::instrument;

use crate::error::VerificationError;
use crate::operator::LinearOperator;
use crate::proof::{CcsProof, WitnessClaim};
use crate::relation::{Ccs, combine};
use crate::transcript::{ReductionShape, VerifierTranscript};

/// Check a reduction and return the witness claim the caller must still open.
///
/// Soundness error at most `(2 + (d + 2) * log_rows + 2 * log_cols) / |EF|`, so `EF` must be large.
///
/// The bound counts the homogeneous relation only; public cells need the committed proof.
///
/// # Errors
///
/// Returns the first check the proof fails.
pub fn verify<F, EF, O, C>(
    ccs: &Ccs<F, O>,
    proof: &CcsProof<F, EF>,
    pow_bits: usize,
    challenger: &mut C,
) -> Result<WitnessClaim<EF>, VerificationError>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    O: LinearOperator<F>,
    C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    check(ccs, proof, ReductionShape::new(ccs, pow_bits), challenger).map(|(claim, _)| claim)
}

/// Check a reduction under a shape, returning the claim and the public points.
#[instrument(skip_all, name = "ccs verify", fields(log_rows = ccs.log_rows(), log_cols = ccs.log_cols()))]
pub(crate) fn check<F, EF, O, C>(
    ccs: &Ccs<F, O>,
    proof: &CcsProof<F, EF>,
    shape: ReductionShape,
    challenger: &mut C,
) -> Result<(WitnessClaim<EF>, Vec<Vec<EF>>), VerificationError>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    O: LinearOperator<F>,
    C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let (log_rows, log_cols) = (ccs.log_rows(), ccs.log_cols());
    let pow_bits = shape.pow_bits();
    let mut transcript = VerifierTranscript::<C, F, EF>::new(challenger, shape);

    // Release the transcript completeness check before rejecting.
    let reject = |transcript: &mut VerifierTranscript<'_, C, F, EF>, error| {
        transcript.abort();
        Err(error)
    };

    // Phase 1: zero claim, then reduce to the row point.
    let alphas = transcript.alphas();
    let tau = transcript.tau();
    if proof.zerocheck.claimed_sum != EF::ZERO {
        return reject(&mut transcript, VerificationError::NonzeroZerocheckClaim);
    }
    let zerocheck = transcript.zerocheck(|challenger| {
        proof
            .zerocheck
            .verify(challenger, log_rows, ccs.degree() + 1, pow_bits)
    });
    let (row_point, zerocheck_final) = match zerocheck {
        Ok(reduced) => reduced,
        Err(error) => return reject(&mut transcript, VerificationError::Zerocheck(error)),
    };

    // Bind the matrix outputs; a wrong count never reaches the sponge.
    if transcript.operand_evals(&proof.operand_evals).is_err() {
        return Err(VerificationError::OperandCount {
            expected: ccs.operators().len(),
            actual: proof.operand_evals.len(),
        });
    }

    // final = eq(tau, r) * sum_k alpha_k * selector_k(r) * sum_i c_{k,i} * prod_{j in S_{k,i}} v_j.
    let row = row_point.as_slice();
    let batched: EF = ccs
        .constraints()
        .iter()
        .zip(&alphas)
        .map(|(k, &alpha)| {
            alpha * k.selector().evaluate(row) * combine(k.terms(), &proof.operand_evals)
        })
        .sum();
    if zerocheck_final != Point::eval_eq(&tau, row) * batched {
        return reject(&mut transcript, VerificationError::ZerocheckMismatch);
    }

    // Phase 2: the linear claim is the batched outputs.
    let batching = transcript.batching();
    let linear_claim = dot_product(
        batching.iter().copied(),
        proof.operand_evals.iter().copied(),
    );
    if proof.linear.claimed_sum != linear_claim {
        return reject(&mut transcript, VerificationError::LinearClaimMismatch);
    }
    let linear =
        transcript.linear(|challenger| proof.linear.verify(challenger, log_cols, 2, pow_bits));
    let (col_point, linear_final) = match linear {
        Ok(reduced) => reduced,
        Err(error) => return reject(&mut transcript, VerificationError::Linear(error)),
    };

    // w(s) = sum_j beta_j * M_j~(r, s), from each closed form.
    let weights_at: EF = ccs
        .operators()
        .iter()
        .zip(&batching)
        .map(|(op, &beta)| beta * op.evaluate(row, col_point.as_slice()))
        .sum();
    if linear_final != weights_at * proof.witness_eval {
        return reject(&mut transcript, VerificationError::LinearMismatch);
    }

    let public_points = transcript.finish(proof.witness_eval);
    let claim = WitnessClaim {
        point: col_point,
        value: proof.witness_eval,
    };
    Ok((claim, public_points))
}
