//! Error types shared by WARP root IOP recorders and compilers.

/// Root IOP recorder error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RootIopError {
    /// Codeword length must be a non-zero power of two.
    InvalidLength(usize),
    /// Codeword/commitment shape mismatch.
    ShapeMismatch,
    /// Oracle id was not found.
    UnknownOracle(usize),
    /// Claim id was not found.
    UnknownClaim(usize),
    /// Opened index is outside the oracle.
    IndexOutOfBounds { oracle_id: usize, index: usize },
    /// Claim id list length did not match opening list length.
    OpeningArityMismatch,
    /// Claim metadata disagreed between WARP and the recorded transcript.
    ClaimMetadataMismatch(usize),
    /// Commitment metadata disagreed between WARP and the recorded transcript.
    CommitmentMismatch(usize),
    /// Claim value disagreed with the recorded oracle.
    ClaimValueMismatch(usize),
    /// Base/extension opening was used against the wrong oracle field.
    OracleFieldMismatch,
    /// A prover recorder was asked to perform verifier-only work.
    ProverUsedAsVerifier,
    /// A verifier recorder was asked to perform prover-only work.
    VerifierUsedAsProver,
}

pub(super) fn checked_log2_len(len: usize) -> Result<usize, RootIopError> {
    if len == 0 || !len.is_power_of_two() {
        return Err(RootIopError::InvalidLength(len));
    }
    Ok(len.trailing_zeros() as usize)
}
