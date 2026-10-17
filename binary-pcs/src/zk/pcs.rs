//! Reference PCS implementation; see the public [`super`] protocol documentation.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::TowerLevel;
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_matrix::Dimensions;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeHidingMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, MerkleCap, SerializingHasher};
use rand::rngs::StdRng;
use rand::{CryptoRng, RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

use super::{Challenger, Val, ZkError, observe_bytes, observe_usize};

// Bound allocations and the deliberately quadratic Horner encoder's work.
const MAX_CODEWORD_LEN: usize = 1 << 16;
const MAX_QUERIES: usize = 4096;
const MAX_CELLS: usize = 1 << 20;
const MAX_ENCODING_WORK: usize = 1 << 27;
const VERSION: &[u8] = b"p3-binary-reference-zk-pcs-v1";

type SaltedMmcs = MerkleTreeHidingMmcs<
    Val,
    u8,
    SerializingHasher<Keccak256Hash>,
    CompressionFunctionFromHasher<Keccak256Hash, 2, 32>,
    StdRng,
    2,
    32,
    1,
>;
type Tree = <SaltedMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>;
type Root = MerkleCap<Val, [u8; 32]>;

const fn mmcs(rng: StdRng) -> SaltedMmcs {
    SaltedMmcs::new(
        SerializingHasher::new(Keccak256Hash),
        CompressionFunctionFromHasher::new(Keccak256Hash),
        0,
        rng,
    )
}

/// Checked geometry for the reference monomial PCS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferencePcsConfig {
    message_len: usize,
    num_queries: usize,
    coefficient_len: usize,
    codeword_len: usize,
}

impl ReferencePcsConfig {
    /// Select the message length and number of independent queries.
    ///
    /// Zero sizes, overflow, and geometries exceeding this reference encoder's
    /// allocation/work limits are rejected. The query count is not a certified
    /// security level; see the interactive bound in this module's documentation.
    pub fn new(message_len: usize, num_queries: usize) -> Result<Self, ZkError> {
        if message_len == 0 || num_queries == 0 || num_queries > MAX_QUERIES {
            return Err(ZkError::InvalidConfig("message/query count out of range"));
        }
        let coefficient_len = message_len
            .checked_add(num_queries)
            .and_then(|v| v.checked_add(1))
            .ok_or(ZkError::InvalidConfig("coefficient count overflow"))?;
        let codeword_len = coefficient_len
            .checked_mul(4)
            .and_then(usize::checked_next_power_of_two)
            .filter(|&n| n <= MAX_CODEWORD_LEN)
            .ok_or(ZkError::InvalidConfig("codeword too large"))?;
        let config = Self {
            message_len,
            num_queries,
            coefficient_len,
            codeword_len,
        };
        config.check_width(1)?;
        Ok(config)
    }

    /// Number of unmasked low monomial coefficients per message.
    pub const fn message_len(&self) -> usize {
        self.message_len
    }
    /// Independent tail coefficients per polynomial, q+1.
    pub const fn randomness_len(&self) -> usize {
        self.num_queries + 1
    }
    /// Committed polynomial coefficient count, M+q+1.
    pub const fn coefficient_len(&self) -> usize {
        self.coefficient_len
    }
    /// Size of the shifted nonzero evaluation domain.
    pub const fn codeword_len(&self) -> usize {
        self.codeword_len
    }
    /// Uniform queries sampled with replacement.
    pub const fn num_queries(&self) -> usize {
        self.num_queries
    }

    pub(super) fn check_width(&self, width: usize) -> Result<(), ZkError> {
        if width == 0
            || self
                .codeword_len
                .checked_mul(width)
                .is_none_or(|n| n > MAX_CELLS)
            || self
                .codeword_len
                .checked_mul(self.coefficient_len)
                .and_then(|n| n.checked_mul(width))
                .is_none_or(|n| n > MAX_ENCODING_WORK)
        {
            return Err(ZkError::InvalidShape(
                "polynomial batch exceeds reference resource limits",
            ));
        }
        Ok(())
    }
}

/// A salted Merkle root and the public geometry it commits to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkCommitment {
    pub root: Root,
    pub num_polynomials: usize,
    pub message_len: usize,
    pub num_queries: usize,
}

/// Private single-use state. Deliberately neither Clone nor serializable.
pub struct ZkProverData {
    config: ReferencePcsConfig,
    commitment: ZkCommitment,
    coefficients: Vec<Vec<Val>>,
    tree: Tree,
}

/// One salted leaf and its binary authentication path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkRowOpening {
    pub values: Vec<Val>,
    pub salt: Val,
    pub authentication_path: Vec<[u8; 32]>,
}

/// Authenticated h and u rows for one transcript-derived position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkQueryOpening {
    pub h: ZkRowOpening,
    pub u: ZkRowOpening,
}

/// Polynomial-size proof. No h coefficients or private masks are disclosed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkOpeningProof {
    pub claimed_values: Vec<Val>,
    pub mask_root: Root,
    pub sigma: Vec<Val>,
    /// One complete L-coefficient blend per polynomial, in monomial order.
    pub blended_coefficients: Vec<Vec<Val>>,
    /// Queries retain transcript order, including duplicates.
    pub queries: Vec<ZkQueryOpening>,
}

/// Fixed-field, fixed-hash reference PCS. Configuration is reusable; state is not.
#[derive(Debug, Clone)]
pub struct ReferenceZkPcs {
    config: ReferencePcsConfig,
}

impl ReferenceZkPcs {
    pub const fn new(config: ReferencePcsConfig) -> Self {
        Self { config }
    }
    pub const fn config(&self) -> &ReferencePcsConfig {
        &self.config
    }

    /// Commit columns of M monomial coefficients with independent random tails.
    /// Shape validation precedes every private random draw.
    pub fn commit<R: CryptoRng>(
        &self,
        messages: RowMajorMatrix<Val>,
        rng: &mut R,
    ) -> Result<(ZkCommitment, ZkProverData), ZkError> {
        let width = messages.width;
        self.config.check_width(width)?;
        if self.config.message_len.checked_mul(width) != Some(messages.values.len()) {
            return Err(ZkError::InvalidShape(
                "expected M rows of monomial coefficients",
            ));
        }
        let coefficients: Vec<Vec<_>> = (0..width)
            .map(|column| {
                (0..self.config.coefficient_len)
                    .map(|row| {
                        if row < self.config.message_len {
                            messages.values[row * width + column]
                        } else {
                            rng.random()
                        }
                    })
                    .collect()
            })
            .collect();
        // Release the consumed input before allocating the larger codeword.
        drop(messages);
        // A private seed forks the salt generator. Never seed from the transcript.
        let salted = mmcs(StdRng::from_rng(rng));
        let (root, tree) = salted.commit_matrix(self.encode(&coefficients));
        let commitment = ZkCommitment {
            root,
            num_polynomials: width,
            message_len: self.config.message_len,
            num_queries: self.config.num_queries,
        };
        Ok((
            commitment.clone(),
            ZkProverData {
                config: self.config.clone(),
                commitment,
                coefficients,
                tree,
            },
        ))
    }

    /// Consume state and open the low-coefficient polynomial at one common point.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "single-use prover state must be consumed"
    )]
    pub fn open<R: CryptoRng>(
        &self,
        data: ZkProverData,
        point: Val,
        challenger: &mut Challenger,
        rng: &mut R,
    ) -> Result<ZkOpeningProof, ZkError> {
        if data.config != self.config {
            return Err(ZkError::InvalidConfig(
                "prover state belongs to another configuration",
            ));
        }
        self.validate_commitment(&data.commitment)?;
        let width = data.commitment.num_polynomials;
        let claimed_values: Vec<_> = data
            .coefficients
            .iter()
            .map(|h| evaluate(&h[..self.config.message_len], point))
            .collect();
        let masks: Vec<Vec<Val>> = (0..width)
            .map(|_| {
                (0..self.config.coefficient_len)
                    .map(|_| rng.random())
                    .collect()
            })
            .collect();
        let sigma: Vec<_> = masks
            .iter()
            .map(|u| evaluate(&u[..self.config.message_len], point))
            .collect();
        let salted = mmcs(StdRng::from_rng(rng));
        let (mask_root, mask_tree) = salted.commit_matrix(self.encode(&masks));
        self.bind_opening(
            &data.commitment,
            point,
            &claimed_values,
            &mask_root,
            &sigma,
            challenger,
        );
        let gamma = sample_nonzero(challenger);
        let blended_coefficients: Vec<Vec<_>> = data
            .coefficients
            .iter()
            .zip(&masks)
            .map(|(h, u)| h.iter().zip(u).map(|(&h, &u)| gamma * h + u).collect())
            .collect();
        observe_blend(&blended_coefficients, challenger);
        let queries = self
            .sample_indices(challenger)
            .into_iter()
            .map(|index| ZkQueryOpening {
                h: open_row(&salted, &data.tree, index),
                u: open_row(&salted, &mask_tree, index),
            })
            .collect();
        Ok(ZkOpeningProof {
            claimed_values,
            mask_root,
            sigma,
            blended_coefficients,
            queries,
        })
    }

    /// Verify the proof and return the authenticated low-polynomial evaluations.
    /// Malformed geometry is rejected before the public transcript is mutated.
    pub fn verify(
        &self,
        commitment: &ZkCommitment,
        point: Val,
        proof: &ZkOpeningProof,
        challenger: &mut Challenger,
    ) -> Result<Vec<Val>, ZkError> {
        self.validate_commitment(commitment)?;
        self.validate_proof(commitment.num_polynomials, proof)?;
        self.bind_opening(
            commitment,
            point,
            &proof.claimed_values,
            &proof.mask_root,
            &proof.sigma,
            challenger,
        );
        let gamma = sample_nonzero(challenger);
        for ((g, &claim), &sigma) in proof
            .blended_coefficients
            .iter()
            .zip(&proof.claimed_values)
            .zip(&proof.sigma)
        {
            if evaluate(&g[..self.config.message_len], point) != gamma * claim + sigma {
                return Err(ZkError::InvalidOpening("low-coefficient blend identity"));
            }
        }
        observe_blend(&proof.blended_coefficients, challenger);
        let indices = self.sample_indices(challenger);
        // Verification does not use the MMCS RNG. The fixed seed is not used for commitment.
        let salted = mmcs(StdRng::from_seed([0; 32]));
        let dimensions = [Dimensions {
            width: commitment.num_polynomials,
            height: self.config.codeword_len,
        }];
        for (&index, query) in indices.iter().zip(&proof.queries) {
            verify_row(&salted, &commitment.root, &dimensions, index, &query.h)?;
            verify_row(&salted, &proof.mask_root, &dimensions, index, &query.u)?;
            let x = self.domain_point(index);
            for (column, g) in proof.blended_coefficients.iter().enumerate() {
                if evaluate(g, x) != gamma * query.h.values[column] + query.u.values[column] {
                    return Err(ZkError::InvalidOpening("queried blend identity"));
                }
            }
        }
        Ok(proof.claimed_values.clone())
    }

    /// Bind the commitment, configuration and batch width before an outer point challenge.
    pub fn observe_commitment(&self, commitment: &ZkCommitment, challenger: &mut Challenger) {
        observe_bytes(challenger, VERSION);
        observe_bytes(challenger, b"commitment");
        for value in [
            self.config.message_len,
            self.config.num_queries,
            self.config.randomness_len(),
            self.config.coefficient_len,
            self.config.codeword_len,
            commitment.message_len,
            commitment.num_queries,
            commitment.num_polynomials,
        ] {
            observe_usize(challenger, value);
        }
        // Include a length even though valid proofs always use exactly one root.
        observe_usize(challenger, commitment.root.num_roots());
        challenger.observe(&commitment.root);
    }

    pub(super) fn validate_commitment(&self, commitment: &ZkCommitment) -> Result<(), ZkError> {
        if commitment.message_len != self.config.message_len
            || commitment.num_queries != self.config.num_queries
        {
            return Err(ZkError::InvalidConfig("commitment geometry mismatch"));
        }
        self.config.check_width(commitment.num_polynomials)?;
        if commitment.root.num_roots() != 1 {
            return Err(ZkError::InvalidShape("expected a single commitment root"));
        }
        Ok(())
    }

    pub(super) fn validate_proof(
        &self,
        width: usize,
        proof: &ZkOpeningProof,
    ) -> Result<(), ZkError> {
        let path_len = self.config.codeword_len.ilog2() as usize;
        if proof.claimed_values.len() != width
            || proof.sigma.len() != width
            || proof.mask_root.num_roots() != 1
            || proof.blended_coefficients.len() != width
            || proof
                .blended_coefficients
                .iter()
                .any(|g| g.len() != self.config.coefficient_len)
            || proof.queries.len() != self.config.num_queries
            || proof.queries.iter().any(|q| {
                [&q.h, &q.u].into_iter().any(|row| {
                    row.values.len() != width || row.authentication_path.len() != path_len
                })
            })
        {
            return Err(ZkError::InvalidShape(
                "opening shape does not match configuration",
            ));
        }
        Ok(())
    }

    fn bind_opening(
        &self,
        commitment: &ZkCommitment,
        point: Val,
        claims: &[Val],
        mask_root: &Root,
        sigma: &[Val],
        challenger: &mut Challenger,
    ) {
        self.observe_commitment(commitment, challenger);
        observe_bytes(challenger, b"opening-point-and-claims");
        challenger.observe(point);
        observe_usize(challenger, claims.len());
        challenger.observe_slice(claims);
        observe_bytes(challenger, b"uniform-mask-commitment-and-functional");
        observe_usize(challenger, mask_root.num_roots());
        challenger.observe(mask_root);
        observe_usize(challenger, sigma.len());
        challenger.observe_slice(sigma);
    }

    fn sample_indices(&self, challenger: &mut Challenger) -> Vec<usize> {
        (0..self.config.num_queries)
            .map(|_| challenger.sample_bits(self.config.codeword_len.ilog2() as usize))
            .collect()
    }

    fn domain_point(&self, index: usize) -> Val {
        let log_n = self.config.codeword_len.ilog2() as usize;
        let mut x = Val::cantor_basis(log_n);
        for bit in 0..log_n {
            if index & (1 << bit) != 0 {
                x += Val::cantor_basis(bit);
            }
        }
        x
    }

    fn encode(&self, coefficients: &[Vec<Val>]) -> RowMajorMatrix<Val> {
        let mut values = Vec::with_capacity(self.config.codeword_len * coefficients.len());
        for index in 0..self.config.codeword_len {
            let x = self.domain_point(index);
            values.extend(coefficients.iter().map(|h| evaluate(h, x)));
        }
        RowMajorMatrix::new(values, coefficients.len())
    }
}

fn evaluate<F: Field>(coefficients: &[F], x: F) -> F {
    coefficients
        .iter()
        .rev()
        .fold(F::ZERO, |acc, &coefficient| acc * x + coefficient)
}

fn sample_nonzero(challenger: &mut Challenger) -> Val {
    loop {
        let gamma: Val = challenger.sample();
        if gamma != Val::ZERO {
            return gamma;
        }
    }
}

fn observe_blend(coefficients: &[Vec<Val>], challenger: &mut Challenger) {
    observe_bytes(challenger, b"complete-monomial-blend-before-queries");
    observe_usize(challenger, coefficients.len());
    for g in coefficients {
        observe_usize(challenger, g.len());
        challenger.observe_slice(g);
    }
}

fn open_row(salted: &SaltedMmcs, tree: &Tree, index: usize) -> ZkRowOpening {
    let (mut values, (salts, authentication_path)) = salted.open_batch(index, tree).unpack();
    ZkRowOpening {
        values: values.remove(0),
        salt: salts[0][0],
        authentication_path,
    }
}

fn verify_row(
    salted: &SaltedMmcs,
    root: &Root,
    dimensions: &[Dimensions],
    index: usize,
    row: &ZkRowOpening,
) -> Result<(), ZkError> {
    let values = vec![row.values.clone()];
    let proof = (vec![vec![row.salt]], row.authentication_path.clone());
    salted
        .verify_batch(
            root,
            dimensions,
            index,
            BatchOpeningRef::new(&values, &proof),
        )
        .map_err(|_| ZkError::InvalidOpening("salted Merkle authentication"))
}

#[cfg(test)]
mod tests;
