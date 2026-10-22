//! Ready-made instantiations of the bit commitment.

/// 64 bits per committed element, challenges in `GF(2^192)`, Merkle trees and sponge over a 256-bit byte hash `H`.
pub mod hash64 {
    use p3_binary_field::{BinaryChallenger, Poly64, Poly192};
    use p3_challenger::HashChallenger;
    use p3_commit::Mmcs as MmcsTrait;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{CompressionFunctionFromHasher, CryptographicHasher, SerializingHasher};

    use crate::whir::{
        BitCommitment, BooleanWhirData, BooleanWhirDeferredProof, CommitmentError, CommitmentParams,
    };

    /// The committed field: one element packs 64 bits.
    pub type Val = Poly64;

    /// The challenge field, a cubic extension of the committed field.
    pub type Challenge = Poly192;

    /// Merkle leaf hash.
    pub type Hash<H> = SerializingHasher<H>;

    /// Merkle node compression.
    pub type Compress<H> = CompressionFunctionFromHasher<H, 2, 32>;

    /// Merkle commitment to committed elements.
    pub type Mmcs<H> = MerkleTreeMmcs<Val, u8, Hash<H>, Compress<H>, 2, 32>;

    /// The Fiat-Shamir sponge.
    pub type Challenger<H> = BinaryChallenger<Val, HashChallenger<u8, H, 32>>;

    /// A commitment.
    pub type Commitment<H> = <Mmcs<H> as MmcsTrait<Val>>::Commitment;

    /// What the prover keeps between committing and opening.
    pub type ProverData<H> = BooleanWhirData<Val, Challenge, Mmcs<H>>;

    /// One opening proof for every claim.
    pub type OpeningProof<H> = BooleanWhirDeferredProof<Val, Challenge, Mmcs<H>>;

    /// The bit commitment of this preset.
    pub type WitnessCommitment<H> = BitCommitment<Val, Challenge, Mmcs<H>, Challenger<H>>;

    /// The bit commitment for `2^num_variables` bits, one opening batching `claims` weighted claims.
    ///
    /// # Errors
    ///
    /// As `BitCommitment::new`.
    pub fn witness_commitment<H>(
        num_variables: usize,
        params: &CommitmentParams,
        claims: usize,
    ) -> Result<WitnessCommitment<H>, CommitmentError>
    where
        H: CryptographicHasher<u8, [u8; 32]> + Default + Clone + Send + Sync,
    {
        WitnessCommitment::new(num_variables, params, claims, |cap_height| {
            Mmcs::new(
                Hash::new(H::default()),
                Compress::new(H::default()),
                cap_height,
            )
        })
    }
}

/// 64 bits per committed element, challenges in `GF(2^192)`, Keccak-256 Merkle trees and sponge.
pub mod keccak64 {
    use p3_keccak::Keccak256Hash;

    use super::hash64;
    pub use super::hash64::{Challenge, Val};
    use crate::whir::{CommitmentError, CommitmentParams};

    /// Merkle leaf hash.
    pub type Hash = hash64::Hash<Keccak256Hash>;

    /// Merkle node compression.
    pub type Compress = hash64::Compress<Keccak256Hash>;

    /// Merkle commitment to committed elements.
    pub type Mmcs = hash64::Mmcs<Keccak256Hash>;

    /// The Fiat-Shamir sponge.
    pub type Challenger = hash64::Challenger<Keccak256Hash>;

    /// A commitment.
    pub type Commitment = hash64::Commitment<Keccak256Hash>;

    /// What the prover keeps between committing and opening.
    pub type ProverData = hash64::ProverData<Keccak256Hash>;

    /// One opening proof for every claim.
    pub type OpeningProof = hash64::OpeningProof<Keccak256Hash>;

    /// The bit commitment of this preset.
    pub type WitnessCommitment = hash64::WitnessCommitment<Keccak256Hash>;

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
        hash64::witness_commitment(num_variables, params, claims)
    }
}
