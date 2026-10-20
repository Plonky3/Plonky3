//! The reduction bound to a committed witness and its public cells, closed by one opening.

use alloc::vec;

use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_field::ExtensionField;
use p3_sumcheck::{OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableShape, TableSpec};

use crate::error::CommittedVerificationError;
use crate::operator::LinearOperator;
use crate::proof::CommittedProof;
use crate::public::PublicRegion;
use crate::relation::Ccs;
use crate::transcript::ReductionShape;
use crate::{prover, verifier};

/// The opening schedule of the witness: one column, opened at `points` points.
pub(crate) fn witness_protocol(log_cols: usize, points: usize) -> OpeningProtocol {
    let batches = (0..points)
        .map(|_| OpeningBatch::new(vec![0], vec![]))
        .collect();
    OpeningProtocol::new(vec![TableSpec::new(TableShape::new(log_cols, 1), batches)])
}

/// Prove a committed witness satisfies the system and holds the public cells.
///
/// The prover data must come from committing the witness on this challenger, which binds the commitment.
///
/// The reduction point and one point per public region are opened together.
///
/// # Errors
///
/// Returns the commitment scheme's configuration error.
///
/// # Panics
///
/// - The witness has the wrong length, or a public region lies outside it.
pub fn prove_committed<F, EF, O, P, C>(
    ccs: &Ccs<F, O>,
    pcs: &P,
    z: &[F],
    public: &[PublicRegion<F>],
    prover_data: P::ProverData,
    pow_bits: usize,
    challenger: &mut C,
) -> Result<CommittedProof<F, EF, P::Proof>, P::ProverError>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    O: LinearOperator<F>,
    P: PrescribedPointPcs<EF, C, Val = F>,
    C: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<P::Commitment>,
{
    let log_cols = ccs.log_cols();
    assert!(
        public.iter().all(|region| region.fits(log_cols)),
        "a public region lies outside the witness"
    );

    // Reduce satisfiability to one evaluation, then draw a point per public region.
    let shape = ReductionShape::new(ccs, pow_bits).with_public(public);
    let (reduction, claim, inner) = prover::reduce::<F, EF, O, C>(ccs, z, shape, challenger);

    // Open the reduction point and every public point; the prover knows every value.
    let mut known = vec![Some(OpeningBatch::new(vec![claim.value], vec![]))];
    known.extend(
        public
            .iter()
            .zip(&inner)
            .map(|(region, r)| Some(OpeningBatch::new(vec![region.evaluate(r)], vec![]))),
    );
    let mut points = vec![claim.point];
    points.extend(
        public
            .iter()
            .zip(&inner)
            .map(|(region, r)| region.witness_point(log_cols, r)),
    );
    let opening = pcs.open_at_known(
        prover_data,
        &witness_protocol(log_cols, points.len()),
        &points,
        &known,
        challenger,
    )?;
    Ok(CommittedProof { reduction, opening })
}

/// Verify a committed proof against the witness commitment and the public cells.
///
/// The commitment is bound here, before any challenge, as committing bound it for the prover.
///
/// # Errors
///
/// Returns the first check the reduction, the public cells or the opening fails.
pub fn verify_committed<F, EF, O, P, C>(
    ccs: &Ccs<F, O>,
    pcs: &P,
    commitment: &P::Commitment,
    public: &[PublicRegion<F>],
    proof: &CommittedProof<F, EF, P::Proof>,
    pow_bits: usize,
    challenger: &mut C,
) -> Result<(), CommittedVerificationError<P::Error>>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    O: LinearOperator<F>,
    P: PrescribedPointPcs<EF, C, Val = F>,
    C: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<P::Commitment>,
{
    let log_cols = ccs.log_cols();
    if let Some(region) = public.iter().position(|region| !region.fits(log_cols)) {
        return Err(CommittedVerificationError::PublicOutOfRange { region, log_cols });
    }

    // The witness is fixed before the reduction draws anything.
    pcs.observe_commitment(commitment, challenger);

    // The reduction leaves the point and value the opening must confirm.
    let shape = ReductionShape::new(ccs, pow_bits).with_public(public);
    let (claim, inner) = verifier::check(ccs, &proof.reduction, shape, challenger)?;

    // One opening returns the committed values at every point.
    let mut points = vec![claim.point];
    points.extend(
        public
            .iter()
            .zip(&inner)
            .map(|(region, r)| region.witness_point(log_cols, r)),
    );
    let opened = pcs
        .verify_at(
            commitment,
            &proof.opening,
            &witness_protocol(log_cols, points.len()),
            &points,
            challenger,
        )
        .map_err(CommittedVerificationError::Opening)?;

    // The first value closes the reduction.
    let [witness, cells @ ..] = opened.as_slice() else {
        return Err(CommittedVerificationError::WitnessMismatch);
    };
    if witness.current() != [claim.value] || cells.len() != public.len() {
        return Err(CommittedVerificationError::WitnessMismatch);
    }

    // Every other value is a public region's extension, computed here from its values.
    for (region, ((batch, values), r)) in cells.iter().zip(public).zip(&inner).enumerate() {
        if batch.current() != [values.evaluate(r)] {
            return Err(CommittedVerificationError::PublicMismatch { region });
        }
    }
    Ok(())
}
