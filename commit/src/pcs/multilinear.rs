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
    /// # Returns
    ///
    /// - A succinct commitment (e.g. a Merkle root).
    /// - Opaque prover data consumed by `open`.
    fn commit(
        &self,
        witness: Self::Witness,
        challenger: &mut Challenger,
    ) -> (Self::Commitment, Self::ProverData);

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
    fn open(
        &self,
        prover_data: Self::ProverData,
        protocol: Self::OpeningProtocol,
        challenger: &mut Challenger,
    ) -> Self::Proof;

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
