//! Polynomial commitment scheme trait for multilinear polynomials.

use core::fmt::Debug;

use p3_field::{ExtensionField, Field};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Polynomial commitment scheme for multilinear polynomials over the Boolean hypercube.
///
/// A multilinear polynomial in m variables is defined by its 2^m evaluations
/// on {0,1}^m. This trait abstracts the three phases of a PCS:
///
/// - **Commit**: bind to a witness and return a public commitment plus
///   prover-only auxiliary data.
/// - **Open**: produce a proof for an agreed opening protocol using the
///   prover data from commitment.
/// - **Verify**: check the proof against the public commitment and opening
///   protocol.
pub trait MultilinearPcs<Challenge, Challenger>
where
    Challenge: ExtensionField<Self::Val>,
{
    /// Base field of the committed polynomials.
    type Val: Field;

    /// Succinct binding commitment sent to the verifier.
    type Commitment: Clone + Serialize + DeserializeOwned;

    /// Prover-side auxiliary data retained between commit and open.
    /// Never sent to the verifier.
    type ProverData;

    /// Opening proof checked by the verifier.
    type Proof: Clone + Serialize + DeserializeOwned;

    /// Verification failure type.
    type Error: Debug;

    /// Configuration or budget failure during commitment or opening.
    type ProverError: Debug;

    /// Committed witness.
    type Witness;

    /// Public opening shapes agreed before commit.
    type OpeningProtocol;

    /// Number of variables m of the committed polynomials.
    /// Every polynomial has 2^m evaluations.
    fn num_vars(&self) -> usize;

    /// Commit to a multilinear witness.
    ///
    /// The concrete witness representation is implementation-defined. It may
    /// be a flat polynomial, a table layout, or another structure that expands
    /// to multilinear evaluations over the Boolean hypercube.
    ///
    /// # Transcript
    ///
    /// The challenger is the sponge the whole proof shares.
    ///
    /// This phase owes it exactly one binding.
    ///
    /// ```text
    ///     required   ->  the commitment being returned, bound once
    ///     forbidden  ->  any other absorb, any sample, any grind
    /// ```
    ///
    /// This method performs that binding by calling the scheme's own binding method.
    ///
    /// A verifier never reaches this one, so it calls that same binding method instead.
    ///
    /// Routing both sides through one call is what makes them interchangeable.
    ///
    /// An absorbed table height, or a batching challenge drawn here, desyncs the two sides.
    ///
    /// Neither side has a step out of place, so the caller sees an unexplained rejection.
    ///
    /// # Returns
    ///
    /// - A succinct commitment (e.g. a Merkle root).
    /// - Opaque prover data consumed by `open`.
    ///
    /// Configuration and budget rejection must not mutate the challenger or consume
    /// private randomness.
    ///
    /// A successful call still binds the commitment exactly once.
    fn commit(
        &self,
        witness: Self::Witness,
        challenger: &mut Challenger,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::ProverError>;

    /// Bind a commitment into the transcript.
    ///
    /// # Overview
    ///
    /// The prover binds its commitment while producing it.
    ///
    /// A verifier never produces one.
    ///
    /// It calls this method instead, at the same point in the sponge stream.
    ///
    /// ```text
    ///     prover  : commit(..)  ->  binds the root it produced
    ///     verifier: this method ->  binds the root it was handed
    /// ```
    ///
    /// # Soundness
    ///
    /// Every implementation's commit phase binds by calling this method, and so
    /// does every verifier.
    ///
    /// Neither side can then drift from the other:
    ///
    /// - by absorbing a different value,
    /// - in a different encoding,
    /// - or under a different phase.
    ///
    /// A scheme whose binding is a typed phase keeps that phase here.
    ///
    /// The conformance tests pin the two against each other, so an implementation
    /// that binds inside its commit phase instead is caught rather than trusted.
    ///
    /// # Arguments
    ///
    /// - `commitment`: the commitment to bind.
    /// - `challenger`: sponge of the surrounding protocol, borrowed for the binding.
    fn observe_commitment(&self, commitment: &Self::Commitment, challenger: &mut Challenger);

    /// Produce an opening proof for the supplied opening protocol.
    ///
    /// Consumes the prover data returned by `commit`. The opening protocol is
    /// public metadata shared with the verifier and determines which committed
    /// values are opened.
    ///
    /// # Returns
    ///
    /// - The opening proof, including any implementation-specific claimed
    ///   evaluations needed by `verify`.
    ///
    /// Configuration and budget rejection leaves the challenger and private randomness
    /// unchanged; it does not undo the preceding successful commitment.
    fn open(
        &self,
        prover_data: Self::ProverData,
        protocol: Self::OpeningProtocol,
        challenger: &mut Challenger,
    ) -> Result<Self::Proof, Self::ProverError>;

    /// Verify an opening proof against a public commitment and opening protocol.
    ///
    /// The opening protocol must be the same public protocol used by the
    /// prover when constructing the proof.
    ///
    /// The challenger must be in the same transcript state as the prover's
    /// challenger was at the corresponding protocol step.
    fn verify(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut Challenger,
        protocol: Self::OpeningProtocol,
    ) -> Result<(), Self::Error>;
}
