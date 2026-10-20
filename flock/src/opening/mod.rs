//! The witness commitment: packed bits under WHIR, opened at quirky claims through the ring switch.
//!
//! A quirky claim weighs the six absorbed bits by the skip domain's Lagrange basis instead of an equality table, as in DP ring switching.

#[cfg(test)]
mod tests;

use alloc::vec::Vec;

use p3_binary_pcs::whir::{
    BinaryWhirDomain, BinaryWhirProfile, BooleanWhirData, BooleanWhirError, BooleanWhirPcs,
    BooleanWhirProof, BooleanWhirProver, ProfileError, WeightedClaim, WeightedOpening,
    recommended_cap_height,
};
use p3_binary_pcs::{BooleanBackend, BooleanMultilinearPcs, PackedWords};
use p3_commit::Mmcs;
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use thiserror::Error;
use tracing::instrument;

use crate::config::{FlockChallenger, PcsParams};
use crate::point::QuirkyClaim;
use crate::skip::{K_SKIP, SKIP_SIZE, lagrange_weights};
use crate::{Challenge, Val, Witness};

/// Merkle leaf hash.
type Hash = SerializingHasher<Keccak256Hash>;

/// Merkle node compression.
type Compress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;

/// Merkle commitment to committed elements.
pub type FlockMmcs = MerkleTreeMmcs<Val, u8, Hash, Compress, 2, 32>;

/// A witness commitment.
pub type Commitment = <FlockMmcs as Mmcs<Val>>::Commitment;

/// What the prover keeps between committing and opening.
pub type ProverData = BooleanWhirData<Val, Challenge, FlockMmcs>;

/// One opening proof for every claim.
pub type OpeningProof = BooleanWhirProof<Val, Challenge, FlockMmcs>;

/// The bit commitment of a batch's witness: 64 bits per element, WHIR over the additive domain.
pub struct WitnessPcs(
    /// The WHIR commitment to the packed bits.
    BooleanWhirPcs<Val, Challenge, BinaryWhirDomain<Val>, FlockMmcs, FlockChallenger>,
);

/// Why committing or opening fails.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OpeningError {
    /// The witness is narrower than one committed element.
    #[error("a witness of {num_variables} variables is narrower than one element of {K_SKIP}")]
    TooFewVariables {
        /// The witness's variables.
        num_variables: usize,
    },
    /// The parameters admit no WHIR schedule.
    #[error(transparent)]
    Profile(#[from] ProfileError),
    /// The commitment refused.
    #[error(transparent)]
    Pcs(#[from] BooleanWhirError),
    /// The witness does not have the committed variables.
    #[error("the witness has {actual} words, the commitment {expected}")]
    WitnessLength {
        /// Words the commitment holds.
        expected: usize,
        /// Supplied.
        actual: usize,
    },
    /// A claim's point does not name the committed variables.
    #[error("a claim names {actual} coordinates past the skip, expected {expected}")]
    PointArity {
        /// The committed variables less the skipped ones.
        expected: usize,
        /// Supplied.
        actual: usize,
    },
    /// The prover was asked to open a claim the witness does not satisfy.
    #[error("claim {index} is not the witness's value")]
    FalseClaim {
        /// The claim's position.
        index: usize,
    },
}

impl WitnessPcs {
    /// The bit commitment for a witness of `2^num_variables` bits.
    ///
    /// # Errors
    ///
    /// The witness is narrower than one element, or the parameters admit no schedule.
    pub fn new(num_variables: usize, params: &PcsParams) -> Result<Self, OpeningError> {
        let packed = num_variables
            .checked_sub(K_SKIP)
            .ok_or(OpeningError::TooFewVariables { num_variables })?;
        let domain = BinaryWhirDomain::<Val>::default();
        let profile = BinaryWhirProfile::proven_list_decoding(
            params.security_bits,
            params.log_inv_rate,
            params.folding.min(packed),
        );
        let config = profile.config::<Challenge, Val, FlockChallenger, _>(packed, &domain)?;
        let cap_height = recommended_cap_height(&config);
        let mmcs = FlockMmcs::new(
            Hash::new(Keccak256Hash),
            Compress::new(Keccak256Hash),
            cap_height,
        );
        Ok(Self(BooleanWhirPcs::new(
            BooleanWhirProver::new(config, domain, mmcs),
            num_variables,
        )?))
    }

    /// Commit to a witness's packed bits, observing the commitment.
    ///
    /// # Errors
    ///
    /// The witness does not have the committed variables.
    #[instrument(skip_all, name = "flock commit")]
    pub fn commit(
        &self,
        witness: &Witness,
        challenger: &mut FlockChallenger,
    ) -> Result<(Commitment, ProverData), OpeningError> {
        let z = witness.z();
        let expected = 1usize << (BooleanBackend::<Challenge>::num_variables(&self.0) - K_SKIP);
        if z.len() != expected {
            return Err(OpeningError::WitnessLength {
                expected,
                actual: z.len(),
            });
        }
        // Bit j of word w is bit 64 w + j, the order the packing reads.
        let mut packed = PackedWords::<Val>::zeroed(z.len()).expect("Poly64 is one word");
        packed.words_mut().copy_from_slice(z);
        Ok(self.0.commit_packed(packed, challenger)?)
    }

    /// Observe a commitment on the verifier side, as committing does on the prover side.
    pub fn observe(&self, commitment: &Commitment, challenger: &mut FlockChallenger) {
        self.0.observe_commitment(commitment, challenger);
    }

    /// Open the committed witness at every claim, in one proof.
    ///
    /// # Errors
    ///
    /// A claim of the wrong width, a claim the witness does not satisfy, or a refused opening.
    #[instrument(skip_all, name = "flock open")]
    pub fn open(
        &self,
        data: ProverData,
        claims: &[QuirkyClaim],
        challenger: &mut FlockChallenger,
    ) -> Result<OpeningProof, OpeningError> {
        let openings = claims
            .iter()
            .map(|claim| self.weighted(claim))
            .collect::<Result<Vec<_>, _>>()?;
        let (values, proof) = self.0.open_weighted(data, &openings, challenger)?;
        if let Some(index) = values
            .iter()
            .zip(claims)
            .position(|(&value, claim)| value != claim.value)
        {
            return Err(OpeningError::FalseClaim { index });
        }
        Ok(proof)
    }

    /// Check one opening proof against every claim.
    ///
    /// # Errors
    ///
    /// A claim of the wrong width, or any refusal of the ring switch or WHIR.
    #[instrument(skip_all, name = "flock verify opening")]
    pub fn verify(
        &self,
        commitment: &Commitment,
        claims: &[QuirkyClaim],
        proof: &OpeningProof,
        challenger: &mut FlockChallenger,
    ) -> Result<(), OpeningError> {
        let claims = claims
            .iter()
            .map(|claim| {
                Ok(WeightedClaim {
                    opening: self.weighted(claim)?,
                    value: claim.value,
                })
            })
            .collect::<Result<Vec<_>, OpeningError>>()?;
        Ok(self
            .0
            .verify_weighted(commitment, &claims, proof, challenger)?)
    }

    /// A claim as a weighted opening: Lagrange weights at the skip coordinate on the absorbed bits.
    fn weighted(&self, claim: &QuirkyClaim) -> Result<WeightedOpening<Challenge>, OpeningError> {
        let expected = BooleanBackend::<Challenge>::num_variables(&self.0) - K_SKIP;
        if claim.point.rest.len() != expected {
            return Err(OpeningError::PointArity {
                expected,
                actual: claim.point.rest.len(),
            });
        }
        Ok(WeightedOpening {
            high: Point::new(claim.point.rest.clone()),
            low_weights: lagrange_weights(claim.point.skip, SKIP_SIZE),
        })
    }
}
