//! The Flock verifier: bind the statement and the commitment, replay the PIOP, check the opening.

use p3_binary_pcs::whir::preset::hash64;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs, MmcsArith};
use p3_verifier_arith::Native;
use tracing::instrument;

use crate::config::{FlockChallenger, FlockConfig, FlockHash};
use crate::error::VerificationError;
use crate::opening::{Commitment, FlockMmcs};
use crate::piop::{self, Placement};
use crate::ports::PublicPorts;
use crate::proof::FlockProof;
use crate::{Batch, Challenge, FlockArith, Val};

/// Verify that a proof's committed witness satisfies every instance of a batch and holds its public port words.
///
/// Without public words the statement is existential: some witness satisfies every instance.
///
/// # Errors
///
/// Returns the first phase that rejects.
pub fn verify<H: FlockHash>(
    config: &FlockConfig,
    batch: &Batch<'_>,
    public: &PublicPorts,
    proof: &FlockProof<H>,
    challenger: &mut FlockChallenger<H>,
) -> Result<(), VerificationError> {
    verify_in(&mut Native::new(), config, batch, public, proof, challenger)
}

/// As [`verify`], within an arithmetic.
///
/// The statement is a constant of the arithmetic, and the proof's values are lifted into it.
///
/// # Errors
///
/// As [`verify`].
#[instrument(skip_all, name = "flock verify", fields(log_n = batch.log_n()))]
pub fn verify_in<A, Ch, H>(
    arith: &mut A,
    config: &FlockConfig,
    batch: &Batch<'_>,
    public: &PublicPorts,
    proof: &FlockProof<H>,
    challenger: &mut Ch,
) -> Result<(), VerificationError>
where
    A: FlockArith,
    Ch: FieldChallenger<Val>
        + GrindingChallenger<Witness = Val>
        + CanSampleUniformBits<Val>
        + CanObserve<Commitment>,
    H: FlockHash,
    FlockMmcs<H>: MmcsArith<
            A,
            A::F,
            Commitment = Commitment,
            MultiProof = <FlockMmcs<H> as Mmcs<Val>>::MultiProof,
        >,
    for<'s> ExtensionMmcs<Val, Challenge, &'s FlockMmcs<H>>: MmcsArith<
            A,
            A::EF,
            MultiProof = <FlockMmcs<H> as Mmcs<Val>>::MultiProof,
            Target = <FlockMmcs<H> as MmcsArith<A, A::F>>::Target,
        >,
{
    // Bind the statement, then the commitment, where the prover did.
    arith.bind_statement(batch, public, challenger)?;
    let pcs = hash64::witness_commitment::<H>(batch.log_n(), &config.pcs, 1)?;
    let commitment = pcs.mmcs().load_commitment(arith, &proof.commitment);
    pcs.observe_in(arith, &proof.commitment, &commitment, challenger);

    // Replay the reductions down to one witness claim, then check its opening.
    let claim = piop::verify_in(
        arith,
        config.pow_bits,
        batch,
        public,
        &proof.piop,
        &Placement::whole(&pcs),
        challenger,
    )?;
    pcs.verify_in(arith, &commitment, &[claim], &proof.opening, challenger)?;
    Ok(())
}
