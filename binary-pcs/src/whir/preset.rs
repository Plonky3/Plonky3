//! Ready-made instantiations of the bit commitment.

/// 64 bits per committed element, challenges in `GF(2^192)`, Keccak-256 Merkle trees and sponge.
pub mod keccak64 {
    use p3_binary_field::{BinaryChallenger, Poly64, Poly192};
    use p3_challenger::HashChallenger;
    use p3_commit::Mmcs as MmcsTrait;
    use p3_keccak::Keccak256Hash;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};

    use crate::whir::{
        BitCommitment, BooleanWhirData, BooleanWhirDeferredProof, CommitmentError, CommitmentParams,
    };

    /// The committed field: one element packs 64 bits.
    pub type Val = Poly64;

    /// The challenge field, a cubic extension of the committed field.
    pub type Challenge = Poly192;

    /// Merkle leaf hash.
    pub type Hash = SerializingHasher<Keccak256Hash>;

    /// Merkle node compression.
    pub type Compress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;

    /// Merkle commitment to committed elements.
    pub type Mmcs = MerkleTreeMmcs<Val, u8, Hash, Compress, 2, 32>;

    /// The Fiat-Shamir sponge.
    pub type Challenger = BinaryChallenger<Val, HashChallenger<u8, Keccak256Hash, 32>>;

    /// A commitment.
    pub type Commitment = <Mmcs as MmcsTrait<Val>>::Commitment;

    /// What the prover keeps between committing and opening.
    pub type ProverData = BooleanWhirData<Val, Challenge, Mmcs>;

    /// One opening proof for every claim.
    pub type OpeningProof = BooleanWhirDeferredProof<Val, Challenge, Mmcs>;

    /// The bit commitment of this preset.
    pub type WitnessCommitment = BitCommitment<Val, Challenge, Mmcs, Challenger>;

    /// The bit commitment for `2^num_variables` bits, one opening batching `claims` weighted claims.
    ///
    /// # Errors
    ///
    /// As `BitCommitment::new`.
    pub fn witness_commitment(
        num_variables: usize,
        params: &CommitmentParams,
        claims: usize,
    ) -> Result<WitnessCommitment, CommitmentError> {
        WitnessCommitment::new(num_variables, params, claims, |cap_height| {
            Mmcs::new(
                Hash::new(Keccak256Hash),
                Compress::new(Keccak256Hash),
                cap_height,
            )
        })
    }
}
