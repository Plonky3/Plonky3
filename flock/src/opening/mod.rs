//! The witness commitment: packed bits under WHIR, opened at quirky claims through the ring switch.
//!
//! A quirky claim weighs the six absorbed bits by the skip domain's Lagrange basis instead of an equality table, as in DP ring switching.

#[cfg(test)]
mod tests;

use alloc::vec::Vec;

use p3_binary_pcs::whir::{
    BinaryWhirDomain, BinaryWhirProfile, BitTensor, BooleanWhirData, BooleanWhirDeferredProof,
    BooleanWhirError, BooleanWhirPcs, BooleanWhirProver, ProfileError, SuffixTensors,
    WeightedClaim, WeightedOpening, recommended_cap_height,
};
use p3_binary_pcs::{BooleanBackend, BooleanMultilinearPcs, PackedWords};
use p3_commit::Mmcs;
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use thiserror::Error;
use tracing::instrument;

use crate::config::{FlockChallenger, PcsParams};
use crate::point::{QuirkyClaim, QuirkyPoint};
use crate::skip::{K_SKIP, SKIP_SIZE, lagrange_weights};
use crate::{Challenge, Val, Witness};

/// Bits past the target a single error term may be asked for, before the search gives up.
const MAX_TERM_SLACK: usize = 16;

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
pub type OpeningProof = BooleanWhirDeferredProof<Val, Challenge, FlockMmcs>;

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
    /// No per-term target composes to the requested bound.
    #[error("no WHIR schedule composes to {target} bits")]
    Security {
        /// Bits requested.
        target: usize,
    },
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
    /// Every error term is targeted a little past `security_bits`, so that their union bound reaches it.
    ///
    /// The bound is the one opening of one claim, which is what a Flock proof opens.
    ///
    /// # Errors
    ///
    /// The witness is narrower than one element, the parameters admit no schedule, or no target reaches the bound.
    pub fn new(num_variables: usize, params: &PcsParams) -> Result<Self, OpeningError> {
        for term_bits in params.security_bits..=params.security_bits + MAX_TERM_SLACK {
            // Grinding can cover a whole term, leaving no query; a higher target may still need some.
            let pcs = match Self::with_term_bits(num_variables, params, term_bits) {
                Err(OpeningError::Profile(ProfileError::ZeroQueries)) => continue,
                pcs => pcs?,
            };
            let composed = pcs
                .0
                .readings_security(1, false)
                .map(|security| security.error().bits());
            if composed.is_some_and(|bits| bits >= params.security_bits as f64) {
                return Ok(pcs);
            }
        }
        Err(OpeningError::Security {
            target: params.security_bits,
        })
    }

    /// The bit commitment with every error term targeted at `term_bits`.
    fn with_term_bits(
        num_variables: usize,
        params: &PcsParams,
        term_bits: usize,
    ) -> Result<Self, OpeningError> {
        let packed = num_variables
            .checked_sub(K_SKIP)
            .ok_or(OpeningError::TooFewVariables { num_variables })?;
        let domain = BinaryWhirDomain::<Val>::default();
        let profile = BinaryWhirProfile::proven_list_decoding(
            term_bits,
            params.log_inv_rate,
            params.folding.min(packed),
        )
        .with_first_round(params.first_folding.min(packed), params.first_reduction)
        .with_query_grinding(params.query_grinding);
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
        let packed = PackedWords::<Val>::from_words(z).expect("Poly64 is one word");
        Ok(self.0.commit_packed(packed, challenger)?)
    }

    /// Observe a commitment on the verifier side, as committing does on the prover side.
    pub fn observe(&self, commitment: &Commitment, challenger: &mut FlockChallenger) {
        self.0.observe_commitment(commitment, challenger);
    }

    /// A claim's element split over the trailing coordinates the commitment's first fold binds.
    ///
    /// Built from the witness folded over the instances, as the lincheck holds it before its rounds.
    ///
    /// # Returns
    ///
    /// Nothing when the inner coordinates past the skip are fewer than the first fold binds.
    #[must_use]
    pub fn suffix_tensors(
        &self,
        folded: &[Challenge],
        point: &QuirkyPoint,
        log_instances: usize,
    ) -> Option<SuffixTensors<Challenge, Val>> {
        let depth = self.0.first_folding();
        let inner = point.rest.get(log_instances..)?;
        let head = inner.len().checked_sub(depth)?;
        if folded.len() != SKIP_SIZE << inner.len() {
            return None;
        }
        let weights = Poly::new_from_point(&inner[..head], Challenge::ONE);
        let banks = (0..1usize << depth)
            .map(|bank| {
                let mut slices = [Challenge::ZERO; SKIP_SIZE];
                for (high, &weight) in weights.as_slice().iter().enumerate() {
                    let row = &folded[((high << depth) | bank) * SKIP_SIZE..][..SKIP_SIZE];
                    for (slice, &value) in slices.iter_mut().zip(row) {
                        *slice += weight * value;
                    }
                }
                BitTensor::from_columns(&slices)
            })
            .collect();
        SuffixTensors::new(banks)
    }

    /// Open the committed witness at every claim, in one proof.
    ///
    /// The ring switch's sum is proven by WHIR's first fold, which binds no separate point.
    ///
    /// Entry `i` of `splits`, when present, is claim `i`'s element split by [`Self::suffix_tensors`].
    ///
    /// # Errors
    ///
    /// A claim of the wrong width, a claim the witness does not satisfy, or a refused opening.
    #[instrument(skip_all, name = "flock open")]
    pub fn open(
        &self,
        data: ProverData,
        claims: &[QuirkyClaim],
        splits: &[Option<&SuffixTensors<Challenge, Val>>],
        challenger: &mut FlockChallenger,
    ) -> Result<OpeningProof, OpeningError> {
        let openings = claims
            .iter()
            .map(|claim| self.weighted(claim))
            .collect::<Result<Vec<_>, _>>()?;
        let (values, proof) = self
            .0
            .open_weighted_deferred(data, &openings, splits, challenger)?;
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
            .verify_weighted_deferred(commitment, &claims, proof, challenger)?)
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
