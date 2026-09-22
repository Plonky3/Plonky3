//! A multi-STARK proving harness for AIRs over `BinaryField128`, using the binary PCS.
//!
//! [`prove_binary_air`] proves and verifies one AIR instance end to end: it derives the
//! commitment scheme's arity from the trace shape, builds a [`BinaryStarkConfig`], times the
//! prove and verify phases, and reports the composed security bits from
//! [`p3_multi_stark::security_report`]. The PCS is binding but not hiding; proofs built here
//! carry no zero-knowledge guarantee.
//!
//! [`prove_boolean_air`] does the same for an AIR whose trace cells are all bits, committing the
//! trace as bits through a [`BooleanStarkConfig`] rather than one field element per cell. That
//! commitment opens both the current row and the next row of every column.

use core::fmt;
use std::time::Instant;

use p3_air::{Air, BaseAir};
use p3_binary_dft::{AdditiveNtt, AdditiveRsEncoder, PolyBasisNtt};
use p3_binary_field::{BinaryChallenger, BinaryField2, BinaryField128, Ghash128, poly_basis};
use p3_binary_pcs::{
    BinaryPcs, BinaryPcsConfig, BinaryPcsConfigError, BinaryPcsParams, BinaryPcsProverData,
    BooleanPcsError, BooleanTraceData, BooleanTraceError, BooleanTracePcs, GroupedCodewordMmcs,
};
use p3_blake3::Blake3;
use p3_bus::BusSymbolicBuilder;
use p3_challenger::{CanObserve, HashChallenger};
use p3_commit::MultilinearPcs;
use p3_field::RawDataSerializable;
use p3_keccak::Keccak256Hash;
use p3_lookup::InteractionSymbolicBuilder;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_multi_stark::config::{Commitment, MultiStarkConfig, PcsError, PcsProverError, ProverData};
use p3_multi_stark::folder::{InteractionMultilinearFolder, MultilinearFolder};
use p3_multi_stark::packed_ext::{PackedExt, PackedRepr};
use p3_multi_stark::sliced::SlicedFolder;
use p3_multi_stark::subfield::{SubfieldAcc, SubfieldVar};
use p3_multi_stark::{
    MultiStarkProof, ProverInstance, ProverInstances, ProvingError, ProvingKey, ReprBackend,
    SecurityError, SubfieldBackend, VerificationError, VerifierInstance, VerifierInstances,
    VerifyingKey, prove_with_backend, security_report, setup, verify,
};
use p3_sumcheck::layout::{Layout, SuffixProver, Table, Witness, plan_stacked_layout};
use p3_sumcheck::ring_switch::bits::BitRingSwitch;
use p3_sumcheck::{PrescribedPointPcs, TableShape};
use p3_symmetric::{CompressionFunctionFromHasher, CryptographicHasher, SerializingHasher};
use p3_util::log2_strict_usize;

mod whir;
use p3_binary_pcs::BooleanTraceCommitmentError;
pub use p3_binary_pcs::whir::{BinaryWhirBudget, BudgetError};
use p3_binary_pcs::whir::{BooleanWhirError, ProfileError};
pub use whir::{
    BooleanPcsChoice, BooleanWhirStarkConfig, WhirIncompatibility, WhirInteractionFamily,
    WhirOptions, WhirRegime, WhirSummary, boolean_whir_config,
};
use whir::{WhirErrorProjection, boolean_whir_config_with_schedule};

type F = BinaryField128;
/// `H` is the byte hash the Merkle leaves, the Merkle nodes, and the transcript all run.
type Hash<H> = SerializingHasher<H>;
/// `N` is the number of children each Merkle-tree node compresses.
type Compress<H, const N: usize> = CompressionFunctionFromHasher<H, N, 32>;
type MerkleMmcs<H, const N: usize> =
    p3_merkle_tree::MerkleTreeMmcs<F, u8, Hash<H>, Compress<H, N>, N, 32>;
type Mmcs<H, const N: usize> = GroupedCodewordMmcs<MerkleMmcs<H, N>>;
type Challenger<H> = BinaryChallenger<F, HashChallenger<u8, H, 32>>;

/// A byte hash the harness builds its Merkle trees and its Fiat-Shamir transcript from.
///
/// Every implementor is a unit type carrying no state, so [`Self::INSTANCE`] is the only value
/// a configuration ever needs to name one.
pub trait HarnessHash:
    CryptographicHasher<u8, [u8; 32]> + Clone + Copy + fmt::Debug + Send + Sync + 'static
{
    /// The hash itself.
    const INSTANCE: Self;

    /// Collision resistance of this hash, in bits, as the security report's cap.
    ///
    /// The digest is 32 bytes wide, so a hash whose construction is as strong as its output
    /// states the birthday bound, half of that. One that is weaker states the lower number, so
    /// every implementor names its own.
    const COLLISION_RESISTANCE_BITS: usize;
}

impl HarnessHash for Keccak256Hash {
    const INSTANCE: Self = Self;
    const COLLISION_RESISTANCE_BITS: usize = 128;
}

impl HarnessHash for Blake3 {
    const INSTANCE: Self = Self;
    const COLLISION_RESISTANCE_BITS: usize = 128;
}

/// The byte hash one run commits and transcribes with.
///
/// Both choices emit a 32-byte digest and are capped at the same collision resistance, so the
/// composed security a statement reports does not move between them. Proof bytes do: the two
/// hashes produce different roots and different challenges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HashFamily {
    /// Keccak-256.
    #[default]
    Keccak256,
    /// BLAKE3, which compresses a 64-byte block where Keccak-256 permutes a 136-byte rate.
    Blake3,
}

/// Commitment scheme selected for a proof run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcsIdentity {
    /// The existing folding-only binary commitment.
    Folding,
    /// The additive-domain WHIR commitment.
    Whir,
}

impl fmt::Display for HashFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Keccak256 => f.write_str("keccak-256"),
            Self::Blake3 => f.write_str("blake3"),
        }
    }
}

/// The grouped Merkle commitment the rounds of `pcs_config` commit and fold through.
///
/// `leaf_elements` packs that many field elements into one leaf; `None` packs exactly the coset
/// one fold batch opens, which is the grouping the schedule itself derives.
///
/// # Errors
///
/// `leaf_elements` is zero or not a power of two, so no grouping matches it.
fn grouped_mmcs<H: HarnessHash, const N: usize>(
    pcs_config: &BinaryPcsConfig,
    leaf_elements: Option<usize>,
) -> Result<Mmcs<H, N>, BinaryProofError> {
    let merkle = MerkleMmcs::<H, N>::new(
        Hash::new(H::INSTANCE),
        Compress::<H, N>::new(H::INSTANCE),
        0,
    );
    match leaf_elements {
        Some(elements) if !elements.is_power_of_two() => {
            Err(BinaryProofError::UnsupportedLeafElements(elements))
        }
        Some(elements) => Ok(Mmcs::with_group_size(merkle, pcs_config, elements)),
        None => Ok(Mmcs::for_folding(merkle, pcs_config)),
    }
}

/// Field elements one Merkle leaf of the base codeword packs under `mmcs`.
///
/// The commitment caps a leaf at each round's message length, so a request wider than the base
/// codeword's message is reported at the cap rather than at its face value.
///
/// # Panics
///
/// Panics if the base codeword carries no message, which a validated schedule never does.
fn leaf_elements_of<H, const N: usize>(pcs_config: &BinaryPcsConfig, mmcs: &Mmcs<H, N>) -> usize {
    let base_height = 1usize << (pcs_config.num_variables() + pcs_config.log_inv_rate());
    mmcs.group_size_at(base_height)
        .expect("a validated schedule blows its message up by the inverse rate")
}

/// Multi-STARK configuration proving AIRs over `BinaryField128` with the binary PCS.
///
/// `N` is the Merkle tree's child arity.
/// `Ntt` selects the additive transform used to encode the base codeword.
/// `H` is the byte hash the Merkle tree and the transcript share.
pub struct BinaryStarkConfig<const N: usize, Ntt = PolyBasisNtt, H = Keccak256Hash> {
    pcs: BinaryPcs<F, F, Mmcs<H, N>, Mmcs<H, N>, AdditiveRsEncoder<F, Ntt>>,
    leaf_elements: usize,
}

impl<const N: usize, Ntt, H> BinaryStarkConfig<N, Ntt, H> {
    /// Field elements each Merkle leaf of the base codeword packs.
    pub const fn leaf_elements(&self) -> usize {
        self.leaf_elements
    }
}

impl<const N: usize, Ntt, H> MultiStarkConfig for BinaryStarkConfig<N, Ntt, H>
where
    Ntt: AdditiveNtt<F> + Sync,
    H: HarnessHash,
{
    type Val = F;
    type Challenge = F;
    type Challenger = Challenger<H>;
    type Pcs = BinaryPcs<F, F, Mmcs<H, N>, Mmcs<H, N>, AdditiveRsEncoder<F, Ntt>>;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
        // One hash serves the transcript and the Merkle tree alike.
        Some(H::COLLISION_RESISTANCE_BITS)
    }

    fn min_num_variables(&self) -> usize {
        // The binary PCS folds at least one variable and does not pad individual tables.
        1
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Witness<F> {
        SuffixProver::<F, F>::new_witness(tables, 0)
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a BinaryPcsProverData<F, F, Mmcs<H, N>>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

/// Derives a [`BinaryStarkConfig`] for a stacked polynomial of `arity` variables, committing
/// through an `N`-ary Merkle tree of `H` and encoding its codeword through `ntt`.
///
/// `folding` batches up to that many sequential variable folds between PCS commitments; it is
/// clamped to `arity`, since a batch cannot fold more variables than the polynomial has.
/// `leaf_elements` sizes the Merkle leaves independently of that batch; `None` sizes each leaf
/// to exactly the coset a batch opens.
///
/// # Errors
///
/// - `leaf_elements` is zero or not a power of two.
/// - The PCS parameters do not describe a usable schedule for `arity`.
pub fn binary_config<const N: usize, Ntt, H>(
    arity: usize,
    params: BinaryPcsParams,
    folding: usize,
    leaf_elements: Option<usize>,
    ntt: Ntt,
) -> Result<BinaryStarkConfig<N, Ntt, H>, BinaryProofError>
where
    Ntt: AdditiveNtt<F> + Sync,
    H: HarnessHash,
{
    let pcs_config =
        BinaryPcsConfig::try_new_with_folding::<F, F>(arity, params, folding.min(arity))?;
    let mmcs = grouped_mmcs::<H, N>(&pcs_config, leaf_elements)?;
    let leaf_elements = leaf_elements_of(&pcs_config, &mmcs);
    Ok(BinaryStarkConfig {
        pcs: BinaryPcs::with_ntt(pcs_config, mmcs.clone(), mmcs, ntt)?,
        leaf_elements,
    })
}

/// Multi-STARK configuration proving Boolean-valued AIRs over `BinaryField128` with the Boolean
/// commitment.
///
/// The trace is committed as bits, one committed element per 128 of them, and its codeword is
/// encoded through the level's own additive NTT, [`PolyBasisNtt`].
///
/// `N` is the Merkle tree's child arity, and `H` the byte hash it shares with the transcript.
pub struct BooleanStarkConfig<const N: usize, H = Keccak256Hash> {
    pcs: BooleanTracePcs<F, Mmcs<H, N>, Mmcs<H, N>>,
    leaf_elements: usize,
}

impl<const N: usize, H> BooleanStarkConfig<N, H> {
    /// Field elements each Merkle leaf of the base codeword packs.
    pub const fn leaf_elements(&self) -> usize {
        self.leaf_elements
    }
}

impl<const N: usize, H: HarnessHash> MultiStarkConfig for BooleanStarkConfig<N, H> {
    type Val = F;
    type Challenge = F;
    type Challenger = Challenger<H>;
    type Pcs = BooleanTracePcs<F, Mmcs<H, N>, Mmcs<H, N>>;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
        // One hash serves the transcript and the Merkle tree alike.
        Some(H::COLLISION_RESISTANCE_BITS)
    }

    fn min_num_variables(&self) -> usize {
        // Every column keeps its own exact run of the bit witness, so no table is zero-extended.
        1
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Vec<Table<F>> {
        // The commitment packs the bits itself at commit time.
        tables
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a BooleanTraceData<F, Mmcs<H, N>>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

/// Derives a [`BooleanStarkConfig`] for one Boolean trace of `shape`, committing through an
/// `N`-ary Merkle tree of `H`.
///
/// Every column of the trace stacks into one bit witness. Each committed element absorbs
/// [`BitRingSwitch::ABSORBED`] of its variables, and the PCS schedule covers the rest. `folding`
/// and `leaf_elements` mean what they do in [`binary_config`], with `folding` clamped to the
/// committed arity.
///
/// # Errors
///
/// - The bit witness has fewer variables than one committed element absorbs.
/// - The PCS parameters do not describe a usable schedule for the committed arity.
/// - `leaf_elements` is zero or not a power of two.
pub fn boolean_config<const N: usize, H: HarnessHash>(
    shape: TableShape,
    params: BinaryPcsParams,
    folding: usize,
    leaf_elements: Option<usize>,
) -> Result<BooleanStarkConfig<N, H>, BinaryProofError> {
    let (arity, _) = plan_stacked_layout(&[shape]);
    let absorbed = BitRingSwitch::<F>::ABSORBED;
    let committed = arity
        .checked_sub(absorbed)
        .ok_or(BinaryProofError::BooleanConfig(BooleanTraceError::Boolean(
            BooleanPcsError::WitnessTooNarrow {
                needed: absorbed,
                actual: arity,
            },
        )))?;

    let pcs_config =
        BinaryPcsConfig::try_new_with_folding::<F, F>(committed, params, folding.min(committed))?;
    let mmcs = grouped_mmcs::<H, N>(&pcs_config, leaf_elements)?;
    let leaf_elements = leaf_elements_of(&pcs_config, &mmcs);
    let pcs = BooleanTracePcs::new(pcs_config, mmcs.clone(), mmcs, arity)
        .map_err(BinaryProofError::BooleanConfig)?;
    Ok(BooleanStarkConfig { pcs, leaf_elements })
}

/// A fresh transcript seeded for one commit, prove, or verify call.
fn binary_challenger<H: HarnessHash>() -> Challenger<H> {
    Challenger::from_hasher(b"p3-examples-binary-hash-air-v1".to_vec(), H::INSTANCE)
}

/// Tunable parameters for [`prove_binary_air`].
///
/// Defaults match `multi-stark/examples/prove_binary_field.rs`.
#[derive(Clone, Copy, Debug)]
pub struct BinaryProofOptions {
    /// Commitment scheme used for Boolean traces.
    pub pcs: BooleanPcsChoice,
    /// Log of the inverse code rate for the binary PCS.
    pub log_inv_rate: usize,
    /// Grinding bits the binary PCS demands once, before its query phase.
    pub pcs_pow_bits: usize,
    /// Composed security target of the whole proof, in bits.
    ///
    /// The binary PCS caps it at `125 - arity - log_inv_rate` once its queries are sampled.
    pub security_bits: usize,
    /// Sequential variable folds batched between binary-PCS commitments.
    pub folding: usize,
    /// Grinding bits demanded per sumcheck round.
    pub sumcheck_pow_bits: usize,
    /// Number of children each Merkle-tree node compresses: 2 or 4.
    ///
    /// 4 cuts the tree's compression count to a third, since a 4-ary node's 128 bytes of children still
    /// fit one Keccak-256 block, at the cost of larger authentication paths in the proof.
    pub merkle_arity: usize,
    /// Byte hash the Merkle trees and the Fiat-Shamir transcript share.
    pub hash: HashFamily,
    /// Field elements each Merkle leaf packs, or `None` to pack one fold batch's coset.
    ///
    /// A leaf wider than that coset shortens the tree and hashes longer messages, and pays for
    /// it in proof bytes: every query then authenticates symbols it did not ask for, and those
    /// symbols travel in the opening.
    pub leaf_elements: Option<usize>,
}

impl Default for BinaryProofOptions {
    fn default() -> Self {
        Self {
            pcs: BooleanPcsChoice::Folding,
            log_inv_rate: 2,
            pcs_pow_bits: 0,
            security_bits: 100,
            folding: 3,
            sumcheck_pow_bits: 0,
            merkle_arity: 2,
            hash: HashFamily::Keccak256,
            leaf_elements: None,
        }
    }
}

impl BinaryProofOptions {
    /// The binary-PCS parameters these options select.
    const fn pcs_params(&self) -> BinaryPcsParams {
        BinaryPcsParams {
            log_inv_rate: self.log_inv_rate,
            pow_bits: self.pcs_pow_bits,
            security_level: self.security_bits,
        }
    }
}

/// Measurements from one [`prove_binary_air`] run.
#[derive(Clone, Copy, Debug)]
pub struct BinaryProofReport {
    /// Trace row count.
    pub rows: usize,
    /// Trace column count.
    pub width: usize,
    /// Number of variables in the stacked polynomial the PCS commits to.
    pub stacked_variables: usize,
    /// Byte hash the Merkle trees and the transcript ran.
    pub hash: HashFamily,
    /// Field elements each Merkle leaf of the base codeword packed.
    pub leaf_elements: usize,
    /// Field elements the run asked a leaf to pack, or `None` for one fold batch's coset.
    ///
    /// A request above the base codeword's message length is capped, and only
    /// [`Self::leaf_elements`] describes what the commitment then packed.
    pub requested_leaf_elements: Option<usize>,
    /// Serialized proof size, in bytes.
    pub proof_bytes: usize,
    /// Wall-clock time to lay the trace out as a table and run `prove`.
    pub prove_seconds: f64,
    /// Wall-clock time spent in `verify`.
    pub verify_seconds: f64,
    /// Wall-clock time spent configuring, setting up, and assessing security.
    pub setup_seconds: f64,
    /// Composed security bits reported by `p3_multi_stark::security_report`.
    pub security_bits: f64,
    /// Commitment scheme selected for this run.
    pub pcs: PcsIdentity,
    /// WHIR schedule metadata, when the WHIR commitment was selected.
    pub whir: Option<WhirSummary>,
}

impl fmt::Display for BinaryProofReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Rows: {}", self.rows)?;
        writeln!(f, "Width: {}", self.width)?;
        writeln!(f, "Stacked variables: {}", self.stacked_variables)?;
        writeln!(f, "Hash: {}", self.hash)?;
        write!(
            f,
            "Merkle leaf: {} field elements ({} bytes",
            self.leaf_elements,
            self.leaf_elements.saturating_mul(F::NUM_BYTES)
        )?;
        if let Some(requested) = self
            .requested_leaf_elements
            .filter(|&requested| requested > self.leaf_elements)
        {
            write!(
                f,
                "; requested {requested}, capped at the base message length"
            )?;
        }
        writeln!(f, ")")?;
        writeln!(f, "Proof size: {} bytes", self.proof_bytes)?;
        writeln!(f, "Prove time: {:.3}s", self.prove_seconds)?;
        writeln!(f, "Verify time: {:.3}s", self.verify_seconds)?;
        writeln!(f, "Setup/security time: {:.3}s", self.setup_seconds)?;
        writeln!(f, "Composed security: {:.2} bits", self.security_bits)?;
        writeln!(f, "PCS: {:?}", self.pcs)?;
        if let Some(whir) = self.whir {
            writeln!(f, "WHIR regime: {:?}", whir.regime)?;
            writeln!(f, "WHIR term target: {} bits", whir.term_security_bits)?;
            writeln!(f, "WHIR packed variables: {}", whir.packed_variables)?;
            writeln!(f, "WHIR cap height: {}", whir.cap_height)?;
            writeln!(f, "WHIR opened positions: {}", whir.total_opened_positions)?;
            writeln!(
                f,
                "WHIR PCS payload estimate: {} bytes",
                whir.pcs_payload_bytes
            )?;
            writeln!(f, "WHIR max grinding: {} bits", whir.max_grinding_bits)?;
        }
        Ok(())
    }
}

/// Failure constructing the config, setting up keys, proving, or verifying a binary AIR.
///
/// The wrapped PCS errors project through `BinaryStarkConfig<2>` and `BooleanStarkConfig<2>`,
/// but neither commitment's error type depends on the Merkle arity or the hash, so the same
/// variant covers every supported combination.
#[derive(Debug, thiserror::Error)]
pub enum BinaryProofError {
    /// The requested PCS parameters do not describe a usable binary-PCS schedule.
    #[error("binary PCS configuration failed: {0}")]
    Config(BinaryPcsConfigError),
    /// Proving (including `setup`) rejected its configuration, budget, or security target.
    #[error("binary proof generation failed: {0}")]
    Prove(ProvingError<PcsProverError<BinaryStarkConfig<2>>>),
    /// The generated proof failed verification.
    #[error("binary proof verification failed: {0}")]
    Verify(VerificationError<PcsError<BinaryStarkConfig<2>>>),
    /// The Boolean commitment refused the schedule or the witness width it was built for.
    #[error("Boolean commitment configuration failed: {0}")]
    BooleanConfig(PcsProverError<BooleanStarkConfig<2>>),
    /// Proving through the Boolean commitment failed, including on a cell outside `{0, 1}`.
    #[error("Boolean proof generation failed: {0}")]
    BooleanProve(ProvingError<PcsProverError<BooleanStarkConfig<2>>>),
    /// The generated Boolean-committed proof failed verification.
    #[error("Boolean proof verification failed: {0}")]
    BooleanVerify(VerificationError<PcsError<BooleanStarkConfig<2>>>),
    /// A WHIR profile could not derive a schedule.
    #[error("WHIR profile configuration failed: {0}")]
    WhirProfile(ProfileError),
    /// The WHIR Boolean trace adapter refused its configuration.
    #[error("WHIR Boolean commitment configuration failed: {0}")]
    WhirConfig(BooleanTraceCommitmentError<BooleanWhirError>),
    /// A WHIR configuration exceeded its explicit schedule or payload budget.
    #[error("WHIR budget check failed: {0}")]
    WhirBudget(BinaryWhirBudgetError),
    /// WHIR was selected with unsupported harness options or AIR declarations.
    #[error("WHIR option is incompatible: {0}")]
    WhirIncompatible(WhirIncompatibility),
    /// WHIR proving failed.
    #[error("WHIR proof generation failed: {0}")]
    WhirProve(BooleanWhirProveError),
    /// WHIR verification failed.
    #[error("WHIR proof verification failed: {0}")]
    WhirVerify(BooleanWhirVerifyError),
    /// The statement's security assessment left a component unassessed or below target.
    #[error("binary proof security check failed: {0}")]
    Security(SecurityError),
    /// `options.merkle_arity` is not one of the arities the binary-field harness builds.
    #[error("unsupported Merkle arity {0}; expected 2 or 4")]
    UnsupportedMerkleArity(usize),
    /// `options.leaf_elements` is not a power of two, so no grouping matches it.
    #[error("unsupported leaf size {0}; expected a power of two")]
    UnsupportedLeafElements(usize),
}

/// The budget failure projected from the binary WHIR adapter.
pub type BinaryWhirBudgetError = p3_binary_pcs::whir::BudgetError;

/// The WHIR proving error projected through the two supported hash instantiations.
#[derive(Debug, thiserror::Error)]
pub enum BooleanWhirProveError {
    #[error("Keccak-256: {0}")]
    Keccak(ProvingError<PcsProverError<BooleanWhirStarkConfig<Keccak256Hash>>>),
    #[error("BLAKE3: {0}")]
    Blake3(ProvingError<PcsProverError<BooleanWhirStarkConfig<Blake3>>>),
}

/// The WHIR verification error projected through the two supported hash instantiations.
#[derive(Debug, thiserror::Error)]
pub enum BooleanWhirVerifyError {
    #[error("Keccak-256: {0}")]
    Keccak(VerificationError<PcsError<BooleanWhirStarkConfig<Keccak256Hash>>>),
    #[error("BLAKE3: {0}")]
    Blake3(VerificationError<PcsError<BooleanWhirStarkConfig<Blake3>>>),
}

impl From<BinaryPcsConfigError> for BinaryProofError {
    fn from(error: BinaryPcsConfigError) -> Self {
        Self::Config(error)
    }
}

impl From<ProvingError<PcsProverError<BinaryStarkConfig<2>>>> for BinaryProofError {
    fn from(error: ProvingError<PcsProverError<BinaryStarkConfig<2>>>) -> Self {
        Self::Prove(error)
    }
}

impl From<VerificationError<PcsError<BinaryStarkConfig<2>>>> for BinaryProofError {
    fn from(error: VerificationError<PcsError<BinaryStarkConfig<2>>>) -> Self {
        Self::Verify(error)
    }
}

/// What the harness needs of a configuration beyond [`MultiStarkConfig`]: the leaf geometry it
/// resolved, and how its proving and verification failures surface as a [`BinaryProofError`].
///
/// Coherence cannot tell the two configurations' error projections apart, so `From` impls for
/// both would overlap.
trait HarnessConfig: MultiStarkConfig {
    /// Field elements each Merkle leaf of the base codeword packs.
    fn leaf_elements(&self) -> usize;

    /// Wrap a failure from `setup` or proving.
    fn prove_error(error: ProvingError<PcsProverError<Self>>) -> BinaryProofError;

    /// Wrap a failure from verification.
    fn verify_error(error: VerificationError<PcsError<Self>>) -> BinaryProofError;

    /// Check a complete serialized proof against any configuration-specific byte budget.
    fn check_proof_bytes(&self, _bytes: usize) -> Result<(), BinaryProofError> {
        Ok(())
    }

    /// Return scalar WHIR schedule metadata when this configuration uses WHIR.
    fn whir_summary(&self) -> Option<WhirSummary> {
        None
    }
}

impl<const N: usize, Ntt, H> HarnessConfig for BinaryStarkConfig<N, Ntt, H>
where
    Ntt: AdditiveNtt<F> + Sync,
    H: HarnessHash,
{
    fn leaf_elements(&self) -> usize {
        self.leaf_elements
    }

    fn prove_error(error: ProvingError<PcsProverError<Self>>) -> BinaryProofError {
        BinaryProofError::Prove(error)
    }

    fn verify_error(error: VerificationError<PcsError<Self>>) -> BinaryProofError {
        BinaryProofError::Verify(error)
    }
}

impl<const N: usize, H: HarnessHash> HarnessConfig for BooleanStarkConfig<N, H> {
    fn leaf_elements(&self) -> usize {
        self.leaf_elements
    }

    fn prove_error(error: ProvingError<PcsProverError<Self>>) -> BinaryProofError {
        BinaryProofError::BooleanProve(error)
    }

    fn verify_error(error: VerificationError<PcsError<Self>>) -> BinaryProofError {
        BinaryProofError::BooleanVerify(error)
    }
}

/// AIR obligations the binary-field harness needs, stated once each.
///
/// `BinaryField128` is its own packing and its own extension packing (`F::Packing = F` and
/// `EF::ExtensionPacking = F`), so in [`p3_multi_stark::folder::ProverAir`]'s bound list the
/// scalar-base, packed-base, and verifier instantiations of each folder all become
/// `<'a, F, F, F>`. Naming that trait directly as `ProverAir<F, F>` leaves the compiler unable to
/// choose among the resulting duplicate supertrait obligations; this trait states each distinct
/// one exactly once, so its blanket impl below is what callers actually need to satisfy.
///
/// The subfield bound is the folder [`SubfieldBackend`] evaluates the first zerocheck round with,
/// inside `GF(4)`, and the two sliced bounds are the folders it and [`ReprBackend`] evaluate it
/// with sixty-four rows at a time. The last four are the folders [`ReprBackend`] evaluates the
/// later rounds with, in the polynomial basis, one row or one lane group of rows at a time.
///
/// The bus-symbolic bound lets setup discover an AIR's optional binary-bus declarations.
pub trait BinaryAir:
    BaseAir<F>
    + Air<InteractionSymbolicBuilder<F, F>>
    + Air<BusSymbolicBuilder<F, F>>
    + for<'a> Air<MultilinearFolder<'a, F, F, F>>
    + for<'a> Air<MultilinearFolder<'a, F, PackedExt<F, F>, PackedExt<F, F>>>
    + for<'a> Air<InteractionMultilinearFolder<'a, F, F, F>>
    + for<'a> Air<InteractionMultilinearFolder<'a, F, PackedExt<F, F>, PackedExt<F, F>>>
    + for<'a> Air<
        MultilinearFolder<'a, F, SubfieldVar<F, BinaryField2>, SubfieldAcc<F, BinaryField2>>,
    > + for<'a> Air<SlicedFolder<'a, F, BinaryField2, F>>
    + for<'a> Air<SlicedFolder<'a, F, BinaryField2, Ghash128>>
    + for<'a> Air<MultilinearFolder<'a, F, Ghash128, Ghash128>>
    + for<'a> Air<InteractionMultilinearFolder<'a, F, Ghash128, Ghash128>>
    + for<'a> Air<MultilinearFolder<'a, F, PackedRepr<F, Ghash128>, PackedRepr<F, Ghash128>>>
    + for<'a> Air<
        InteractionMultilinearFolder<'a, F, PackedRepr<F, Ghash128>, PackedRepr<F, Ghash128>>,
    >
{
}

impl<A> BinaryAir for A where
    A: BaseAir<F>
        + Air<InteractionSymbolicBuilder<F, F>>
        + Air<BusSymbolicBuilder<F, F>>
        + for<'a> Air<MultilinearFolder<'a, F, F, F>>
        + for<'a> Air<MultilinearFolder<'a, F, PackedExt<F, F>, PackedExt<F, F>>>
        + for<'a> Air<InteractionMultilinearFolder<'a, F, F, F>>
        + for<'a> Air<InteractionMultilinearFolder<'a, F, PackedExt<F, F>, PackedExt<F, F>>>
        + for<'a> Air<
            MultilinearFolder<'a, F, SubfieldVar<F, BinaryField2>, SubfieldAcc<F, BinaryField2>>,
        > + for<'a> Air<SlicedFolder<'a, F, BinaryField2, F>>
        + for<'a> Air<SlicedFolder<'a, F, BinaryField2, Ghash128>>
        + for<'a> Air<MultilinearFolder<'a, F, Ghash128, Ghash128>>
        + for<'a> Air<InteractionMultilinearFolder<'a, F, Ghash128, Ghash128>>
        + for<'a> Air<MultilinearFolder<'a, F, PackedRepr<F, Ghash128>, PackedRepr<F, Ghash128>>>
        + for<'a> Air<
            InteractionMultilinearFolder<'a, F, PackedRepr<F, Ghash128>, PackedRepr<F, Ghash128>>,
        >
{
}

/// A zerocheck backend the harness can prove with.
///
/// Every variant proves and verifies the same statement and emits a byte-identical proof; the
/// choice is a field-representation performance tradeoff, not a correctness one.
#[derive(Clone, Copy, Debug)]
pub enum Backend {
    /// [`SubfieldBackend`] over `GF(4)`.
    Subfield,
    /// [`ReprBackend`] over `GF(4)`, with later rounds in [`Ghash128`].
    PolyBasis,
    /// [`ReprBackend`] with one additional representation round evaluated directly on planes.
    /// This is a time/memory policy choice; proof bytes remain identical to [`Self::PolyBasis`].
    PolyBasisLate,
}

impl Backend {
    /// The backend this build proves with.
    ///
    /// With a hardware carryless multiply, later rounds run in the polynomial basis, where a
    /// product is that multiply alone and a tower product adds three changes of basis around it.
    /// Without one, later rounds stay in the tower basis.
    pub const fn preferred() -> Self {
        if poly_basis::HAS_HARDWARE_CLMUL {
            Self::PolyBasis
        } else {
            Self::Subfield
        }
    }

    /// Prove through this backend, whose proof and transcript are those of every other.
    fn prove<A: BinaryAir, C, H: HarnessHash>(
        self,
        config: &C,
        instances: ProverInstances<'_, C, A>,
        pow_bits: usize,
        challenger: &mut Challenger<H>,
    ) -> Result<MultiStarkProof<C>, ProvingError<PcsProverError<C>>>
    where
        C: MultiStarkConfig<Val = F, Challenge = F, Challenger = Challenger<H>>,
        C::Pcs: PrescribedPointPcs<F, Challenger<H>>,
        Challenger<H>: CanObserve<Commitment<C>>,
        Commitment<C>: Clone,
        ProverData<C>: Clone,
    {
        match self {
            Self::Subfield => prove_with_backend::<_, _, SubfieldBackend<BinaryField2>>(
                config, instances, pow_bits, challenger,
            ),
            Self::PolyBasis => prove_with_backend::<_, _, ReprBackend<BinaryField2, Ghash128>>(
                config, instances, pow_bits, challenger,
            ),
            Self::PolyBasisLate => {
                prove_with_backend::<_, _, ReprBackend<BinaryField2, Ghash128, true>>(
                    config, instances, pow_bits, challenger,
                )
            }
        }
    }
}

/// Proves and verifies `air` against `trace`, reporting size and timing measurements.
///
/// Encodes the binary-PCS codeword through the default additive NTT ([`PolyBasisNtt`]); see
/// [`prove_binary_air_with_ntt`] to choose a different one.
///
/// # Panics
///
/// - The trace height is not a power of two.
/// - `air` declares public values or preprocessed columns.
/// - `air` assumes a Boolean trace, which only [`prove_boolean_air`] commits.
pub fn prove_binary_air<A>(
    air: &A,
    trace: RowMajorMatrix<F>,
    options: BinaryProofOptions,
) -> Result<BinaryProofReport, BinaryProofError>
where
    A: BinaryAir,
{
    prove_binary_air_with_ntt(air, trace, options, PolyBasisNtt::default())
}

/// Proves and verifies `air` against `trace`, reporting size and timing measurements.
///
/// Dispatches on `options.merkle_arity` and `options.hash` to build a Merkle tree of that child
/// count over that hash, encodes the binary-PCS codeword through `ntt`, and runs its zerocheck
/// through [`Backend::preferred`]; see [`prove_binary_air_with_ntt_and_backend`] to choose a
/// different backend.
///
/// # Panics
///
/// - The trace height is not a power of two.
/// - `air` declares public values or preprocessed columns.
/// - `air` assumes a Boolean trace, which only [`prove_boolean_air`] commits.
pub fn prove_binary_air_with_ntt<A, Ntt>(
    air: &A,
    trace: RowMajorMatrix<F>,
    options: BinaryProofOptions,
    ntt: Ntt,
) -> Result<BinaryProofReport, BinaryProofError>
where
    A: BinaryAir,
    Ntt: AdditiveNtt<F> + Sync,
{
    prove_binary_air_with_ntt_and_backend(air, trace, options, ntt, Backend::preferred())
}

/// Proves and verifies `air` against `trace`, reporting size and timing measurements.
///
/// Dispatches on `options.merkle_arity` and `options.hash` to build a Merkle tree of that child
/// count over that hash, encodes the binary-PCS codeword through `ntt`, and runs its zerocheck
/// through `backend`: [`ReprBackend`] over `GF(4)` and [`Ghash128`] for [`Backend::PolyBasis`],
/// [`SubfieldBackend`] over `GF(4)` for [`Backend::Subfield`], or
/// [`ReprBackend<BinaryField2, Ghash128, true>`] with opt-in deferred materialization for
/// [`Backend::PolyBasisLate`]. Every backend emits a proof identical to the one
/// [`p3_multi_stark::prove`] does.
///
/// # Panics
///
/// - The trace height is not a power of two.
/// - `air` declares public values or preprocessed columns.
/// - `air` assumes a Boolean trace, which only [`prove_boolean_air`] commits.
pub fn prove_binary_air_with_ntt_and_backend<A, Ntt>(
    air: &A,
    trace: RowMajorMatrix<F>,
    options: BinaryProofOptions,
    ntt: Ntt,
    backend: Backend,
) -> Result<BinaryProofReport, BinaryProofError>
where
    A: BinaryAir,
    Ntt: AdditiveNtt<F> + Sync,
{
    match (options.merkle_arity, options.hash) {
        (2, HashFamily::Keccak256) => {
            prove_binary_air_with::<A, 2, Ntt, Keccak256Hash>(air, trace, options, ntt, backend)
        }
        (2, HashFamily::Blake3) => {
            prove_binary_air_with::<A, 2, Ntt, Blake3>(air, trace, options, ntt, backend)
        }
        (4, HashFamily::Keccak256) => {
            prove_binary_air_with::<A, 4, Ntt, Keccak256Hash>(air, trace, options, ntt, backend)
        }
        (4, HashFamily::Blake3) => {
            prove_binary_air_with::<A, 4, Ntt, Blake3>(air, trace, options, ntt, backend)
        }
        (other, _) => Err(BinaryProofError::UnsupportedMerkleArity(other)),
    }
}

/// Proves and verifies `air` against `trace` through an `N`-ary Merkle tree, reporting size and
/// timing measurements.
///
/// The commitment arity is the trace's log-height plus the ceiling of the log of its width:
/// one extra variable per doubling of the column count, since every column is stacked into a
/// single committed polynomial.
#[allow(clippy::needless_pass_by_value)]
fn prove_binary_air_with<A, const N: usize, Ntt, H>(
    air: &A,
    trace: RowMajorMatrix<F>,
    options: BinaryProofOptions,
    ntt: Ntt,
    backend: Backend,
) -> Result<BinaryProofReport, BinaryProofError>
where
    A: BinaryAir,
    Ntt: AdditiveNtt<F> + Sync,
    H: HarnessHash,
{
    if matches!(options.pcs, BooleanPcsChoice::Whir(_)) {
        return Err(BinaryProofError::WhirIncompatible(
            WhirIncompatibility::DenseField,
        ));
    }
    assert!(
        !BaseAir::<F>::assumes_boolean_trace(air),
        "the binary PCS commits field elements, so it cannot prove an AIR that assumes a Boolean trace"
    );
    let setup_start = Instant::now();
    let shape = TableShape::new(log2_strict_usize(trace.height()), trace.width());
    let (arity, _) = plan_stacked_layout(&[shape]);
    let config = binary_config::<N, Ntt, H>(
        arity,
        options.pcs_params(),
        options.folding,
        options.leaf_elements,
        ntt,
    )?;
    prove_and_verify(&config, air, shape, options, setup_start, backend, || {
        Table::new(trace.transpose())
    })
}

/// Proves and verifies a Boolean-valued `air` against `trace`, committing the trace as bits, and
/// reports size and timing measurements.
///
/// The commitment's alphabet is one bit per cell: bits pack into the committed elements by a
/// bijection, so a cell outside `{0, 1}` is not representable, and `air` need not constrain
/// booleanity.
///
/// Runs its zerocheck through [`Backend::preferred`]; see [`prove_boolean_air_with_backend`] to
/// choose a different backend.
///
/// # Errors
///
/// Besides the errors of [`prove_binary_air`], the Boolean commitment refuses a trace cell
/// outside `{0, 1}`, when the trace is committed.
///
/// # Panics
///
/// - The trace height is not a power of two.
/// - `air` declares public values or preprocessed columns.
pub fn prove_boolean_air<A>(
    air: &A,
    trace: Table<F>,
    options: BinaryProofOptions,
) -> Result<BinaryProofReport, BinaryProofError>
where
    A: BinaryAir,
{
    prove_boolean_air_with_backend(air, trace, options, Backend::preferred())
}

/// Assess the selected Boolean PCS and composed security for an AIR shape without a witness.
pub fn preflight_boolean_air<A: BinaryAir>(
    air: &A,
    shape: TableShape,
    options: BinaryProofOptions,
) -> Result<f64, BinaryProofError> {
    preflight_boolean_air_with_summary(air, shape, options).map(|(security_bits, _)| security_bits)
}

/// Assess a Boolean PCS and return its composed security and optional WHIR schedule summary.
pub fn preflight_boolean_air_with_summary<A: BinaryAir>(
    air: &A,
    shape: TableShape,
    options: BinaryProofOptions,
) -> Result<(f64, Option<WhirSummary>), BinaryProofError> {
    match (options.pcs, options.merkle_arity, options.hash) {
        (BooleanPcsChoice::Folding, 2, HashFamily::Keccak256) => {
            preflight_boolean_folding::<A, 2, Keccak256Hash>(air, shape, options)
        }
        (BooleanPcsChoice::Folding, 2, HashFamily::Blake3) => {
            preflight_boolean_folding::<A, 2, Blake3>(air, shape, options)
        }
        (BooleanPcsChoice::Folding, 4, HashFamily::Keccak256) => {
            preflight_boolean_folding::<A, 4, Keccak256Hash>(air, shape, options)
        }
        (BooleanPcsChoice::Folding, 4, HashFamily::Blake3) => {
            preflight_boolean_folding::<A, 4, Blake3>(air, shape, options)
        }
        (BooleanPcsChoice::Folding, arity, _) => {
            Err(BinaryProofError::UnsupportedMerkleArity(arity))
        }
        (BooleanPcsChoice::Whir(_), _, HashFamily::Keccak256) => {
            preflight_boolean_whir::<A, Keccak256Hash>(air, shape, options)
        }
        (BooleanPcsChoice::Whir(_), _, HashFamily::Blake3) => {
            preflight_boolean_whir::<A, Blake3>(air, shape, options)
        }
    }
}

fn preflight_boolean_folding<A, const N: usize, H>(
    air: &A,
    shape: TableShape,
    options: BinaryProofOptions,
) -> Result<(f64, Option<WhirSummary>), BinaryProofError>
where
    A: BinaryAir,
    H: HarnessHash,
{
    let setup_start = Instant::now();
    let config = boolean_config::<N, H>(
        shape,
        options.pcs_params(),
        options.folding,
        options.leaf_elements,
    )?;
    let (_, _, security_bits) = setup_and_assess(&config, air, shape, options)?;
    tracing::debug!(
        target: "p3_examples::binary",
        preflight_seconds = setup_start.elapsed().as_secs_f64(),
        "Boolean PCS preflight completed"
    );
    Ok((security_bits, None))
}

fn preflight_boolean_whir<A, H>(
    air: &A,
    shape: TableShape,
    options: BinaryProofOptions,
) -> Result<(f64, Option<WhirSummary>), BinaryProofError>
where
    A: BinaryAir,
    H: HarnessHash + WhirErrorProjection,
{
    let BooleanPcsChoice::Whir(whir) = options.pcs else {
        unreachable!("WHIR preflight requires a WHIR PCS choice")
    };
    let setup_start = Instant::now();
    let config = boolean_whir_config_with_schedule::<A, H>(air, shape, options, whir, true)?;
    let (_, _, security_bits) = setup_and_assess(&config, air, shape, options)?;
    tracing::debug!(
        target: "p3_examples::binary",
        preflight_seconds = setup_start.elapsed().as_secs_f64(),
        "WHIR Boolean PCS preflight completed"
    );
    Ok((security_bits, Some(config.summary)))
}

/// Proves and verifies a Boolean-valued `air` against `trace`, committing the trace as bits, and
/// reports size and timing measurements.
///
/// Dispatches on `options.merkle_arity` and `options.hash` to build a Merkle tree of that child
/// count over that hash, and runs its zerocheck through `backend`. The codeword is encoded
/// through [`PolyBasisNtt`], the Boolean commitment's own encoder.
///
/// # Errors
///
/// As [`prove_boolean_air`].
///
/// # Panics
///
/// - The trace height is not a power of two.
/// - `air` declares public values or preprocessed columns.
pub fn prove_boolean_air_with_backend<A>(
    air: &A,
    trace: Table<F>,
    options: BinaryProofOptions,
    backend: Backend,
) -> Result<BinaryProofReport, BinaryProofError>
where
    A: BinaryAir,
{
    if matches!(options.pcs, BooleanPcsChoice::Whir(_)) {
        return match options.hash {
            HashFamily::Keccak256 => {
                prove_boolean_whir_with::<A, Keccak256Hash>(air, trace, options, backend)
            }
            HashFamily::Blake3 => {
                prove_boolean_whir_with::<A, Blake3>(air, trace, options, backend)
            }
        };
    }
    match (options.merkle_arity, options.hash) {
        (2, HashFamily::Keccak256) => {
            prove_boolean_air_with::<A, 2, Keccak256Hash>(air, trace, options, backend)
        }
        (2, HashFamily::Blake3) => {
            prove_boolean_air_with::<A, 2, Blake3>(air, trace, options, backend)
        }
        (4, HashFamily::Keccak256) => {
            prove_boolean_air_with::<A, 4, Keccak256Hash>(air, trace, options, backend)
        }
        (4, HashFamily::Blake3) => {
            prove_boolean_air_with::<A, 4, Blake3>(air, trace, options, backend)
        }
        (other, _) => Err(BinaryProofError::UnsupportedMerkleArity(other)),
    }
}

/// Proves and verifies a Boolean-valued `air` against `trace` through an `N`-ary Merkle tree,
/// reporting size and timing measurements.
fn prove_boolean_air_with<A, const N: usize, H>(
    air: &A,
    trace: Table<F>,
    options: BinaryProofOptions,
    backend: Backend,
) -> Result<BinaryProofReport, BinaryProofError>
where
    A: BinaryAir,
    H: HarnessHash,
{
    let shape = trace.shape();
    let setup_start = Instant::now();
    let config = boolean_config::<N, H>(
        shape,
        options.pcs_params(),
        options.folding,
        options.leaf_elements,
    )?;
    prove_and_verify(&config, air, shape, options, setup_start, backend, || trace)
}

fn prove_boolean_whir_with<A, H>(
    air: &A,
    trace: Table<F>,
    options: BinaryProofOptions,
    backend: Backend,
) -> Result<BinaryProofReport, BinaryProofError>
where
    A: BinaryAir,
    H: HarnessHash + WhirErrorProjection,
{
    let shape = trace.shape();
    let setup_start = Instant::now();
    let BooleanPcsChoice::Whir(whir) = options.pcs else {
        unreachable!("WHIR proving requires a WHIR PCS choice")
    };
    let config = boolean_whir_config_with_schedule::<A, H>(air, shape, options, whir, false)?;
    prove_and_verify(&config, air, shape, options, setup_start, backend, || trace)
}

fn setup_and_assess<A, C, H>(
    config: &C,
    air: &A,
    shape: TableShape,
    options: BinaryProofOptions,
) -> Result<(ProvingKey<C>, VerifyingKey<C>, f64), BinaryProofError>
where
    A: BinaryAir,
    H: HarnessHash,
    C: HarnessConfig + MultiStarkConfig<Val = F, Challenge = F, Challenger = Challenger<H>>,
    C::Pcs: PrescribedPointPcs<F, Challenger<H>>,
    Challenger<H>: CanObserve<Commitment<C>>,
    Commitment<C>: Clone,
    ProverData<C>: Clone,
{
    assert_eq!(
        BaseAir::<F>::num_public_values(air),
        0,
        "the harness proves AIRs without public values"
    );
    assert_eq!(
        BaseAir::<F>::preprocessed_width(air),
        0,
        "the harness proves AIRs without preprocessed columns"
    );

    let (pk, vk) = setup(config, &[air], &mut binary_challenger()).map_err(C::prove_error)?;
    let public_values: [F; 0] = [];
    let verifier_instances = VerifierInstances::new(vec![VerifierInstance::new(
        air,
        &vk,
        shape.num_variables(),
        &public_values,
    )]);
    let report =
        security_report(config, &verifier_instances).map_err(BinaryProofError::Security)?;
    report
        .require_security(options.security_bits)
        .map_err(BinaryProofError::Security)?;
    let security_bits = report
        .security_bits()
        .expect("require_security succeeded, so every component is assessed");
    Ok((pk, vk, security_bits))
}

/// Proves and verifies `air` against `trace` under `config`, reporting size and timing
/// measurements.
///
/// The statement's security is assessed once against `options.security_bits` before proving,
/// so the timed phases are the plain prover and verifier. The Merkle grouping `config` resolved
/// is reported alongside the measurements.
fn prove_and_verify<A, C, H>(
    config: &C,
    air: &A,
    shape: TableShape,
    options: BinaryProofOptions,
    setup_start: Instant,
    backend: Backend,
    prepare_table: impl FnOnce() -> Table<F>,
) -> Result<BinaryProofReport, BinaryProofError>
where
    A: BinaryAir,
    H: HarnessHash,
    C: HarnessConfig + MultiStarkConfig<Val = F, Challenge = F, Challenger = Challenger<H>>,
    C::Pcs: PrescribedPointPcs<F, Challenger<H>>,
    Challenger<H>: CanObserve<Commitment<C>>,
    Commitment<C>: Clone,
    ProverData<C>: Clone,
    MultiStarkProof<C>: serde::Serialize + serde::de::DeserializeOwned,
{
    let rows = 1usize << shape.num_variables();
    let width = shape.width();
    let log_height = shape.num_variables();

    let (pk, vk, security_bits) = setup_and_assess(config, air, shape, options)?;
    let setup_seconds = setup_start.elapsed().as_secs_f64();

    let public_values: [F; 0] = [];
    let verifier_instances = || {
        VerifierInstances::new(vec![VerifierInstance::new(
            air,
            &vk,
            log_height,
            &public_values,
        )])
    };

    let prove_start = Instant::now();
    // Prepare the table inside the timed section. Dense callers transpose here; packed callers
    // move their bit-packed table through without materializing field cells.
    let table = prepare_table();
    let prover_instances =
        ProverInstances::new(vec![ProverInstance::new(air, table, &pk, &public_values)]);
    let proof = backend
        .prove(
            config,
            prover_instances,
            options.sumcheck_pow_bits,
            &mut binary_challenger(),
        )
        .map_err(C::prove_error)?;
    let prove_seconds = prove_start.elapsed().as_secs_f64();

    let bytes = postcard::to_allocvec(&proof).expect("postcard serialization must not fail");
    let proof_bytes = bytes.len();
    config.check_proof_bytes(proof_bytes)?;
    let proof: MultiStarkProof<C> =
        postcard::from_bytes(&bytes).expect("postcard round trip must not fail");

    let verify_start = Instant::now();
    verify(
        config,
        verifier_instances(),
        &proof,
        options.sumcheck_pow_bits,
        &mut binary_challenger(),
    )
    .map_err(C::verify_error)?;
    let verify_seconds = verify_start.elapsed().as_secs_f64();

    Ok(BinaryProofReport {
        rows,
        width,
        stacked_variables: config.pcs().num_vars(),
        hash: options.hash,
        leaf_elements: config.leaf_elements(),
        requested_leaf_elements: options.leaf_elements,
        proof_bytes,
        prove_seconds,
        verify_seconds,
        setup_seconds,
        security_bits,
        pcs: if config.whir_summary().is_some() {
            PcsIdentity::Whir
        } else {
            PcsIdentity::Folding
        },
        whir: config.whir_summary(),
    })
}

#[cfg(test)]
mod tests {
    use core::error::Error;

    use p3_air::symbolic::AirLayout;
    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
    use p3_binary_field::{Gf2, TowerLevel};
    use p3_blake3_air::{Blake3BinaryAir, NUM_BLAKE3_BINARY_COLS};
    use p3_bus::{BusActivation, BusDirection, BusInteractionBuilder, BusName, BusSymbolicBuilder};
    use p3_challenger::CanSample;
    use p3_field::{HasSubfield, PrimeCharacteristicRing};
    use p3_keccak_air::{KeccakBinaryAir, NUM_KECCAK_BINARY_COLS};
    use p3_lookup::{Count, IndexedLookupBuilder, InteractionBuilder, TraceWindow};
    use p3_multi_stark::prove;
    use p3_sha256_air::Sha256BinaryAir;
    use p3_util::log2_ceil_usize;

    use super::*;

    #[test]
    fn whir_l20_unique_schedule_configures_without_a_witness() {
        let options = BinaryProofOptions {
            pcs: BooleanPcsChoice::Whir(WhirOptions {
                regime: WhirRegime::UniqueDecoding,
                term_security_bits: 98,
                budget: BinaryWhirBudget {
                    max_stir_queries: 1024,
                    max_proof_bytes: 2 * 1024 * 1024,
                    max_grinding_bits: 0,
                },
            }),
            security_bits: 95,
            log_inv_rate: 1,
            folding: 4,
            merkle_arity: 2,
            ..BinaryProofOptions::default()
        };
        let BooleanPcsChoice::Whir(whir) = options.pcs else {
            unreachable!()
        };
        let config = boolean_whir_config::<_, Blake3>(
            &Blake3BinaryAir::default(),
            TableShape::new(20, 11_536),
            options,
            whir,
        )
        .expect("l20 unique schedule configures without a witness");
        assert_eq!(config.summary.packed_variables, 27);
        assert_eq!(config.summary.max_grinding_bits, 0);
    }

    #[test]
    fn whir_preflight_l18_and_l20_reaches_composed_target_under_both_hashes() {
        for hash in [HashFamily::Keccak256, HashFamily::Blake3] {
            for log_height in [18, 20] {
                let bits = preflight_boolean_air(
                    &Blake3BinaryAir::default(),
                    TableShape::new(log_height, NUM_BLAKE3_BINARY_COLS),
                    whir_preflight_options(hash, 98),
                )
                .expect("WHIR preflight reaches the composed target");
                assert!(
                    bits >= 95.0,
                    "hash={hash:?}, log_height={log_height}, bits={bits}"
                );
            }
        }
    }

    #[test]
    fn whir_preflight_rejects_inadequate_composed_security() {
        let error = preflight_boolean_air(
            &Blake3BinaryAir::default(),
            TableShape::new(20, NUM_BLAKE3_BINARY_COLS),
            whir_preflight_options(HashFamily::Blake3, 95),
        )
        .expect_err("per-term security 95 must not certify composed security 95");
        assert!(matches!(
            error,
            BinaryProofError::Security(SecurityError::InsufficientSecurity { requested: 95, .. })
        ));
    }

    #[test]
    fn whir_proves_small_packed_blake3_under_both_hashes() {
        let air = Blake3BinaryAir::default();
        let words = air.generate_random_trace_packed::<Gf2>(8);
        for hash in [HashFamily::Keccak256, HashFamily::Blake3] {
            let report = prove_boolean_air(
                &air,
                Table::from_packed_bits(words.clone(), 3),
                whir_small_options(hash),
            )
            .expect("small packed BLAKE3 WHIR proof must verify");
            assert_eq!(report.pcs, PcsIdentity::Whir);
            assert_eq!(report.hash, hash);
            assert_eq!(report.rows, 8);
            assert!(report.security_bits >= 95.0);
            assert!(report.whir.is_some());
        }
    }

    #[test]
    fn whir_proves_small_keccak_with_successors() {
        let air = KeccakBinaryAir::default();
        let words = air.generate_random_trace_packed::<Gf2>(1);
        let report = prove_boolean_air(
            &air,
            Table::from_packed_bits(words, 5),
            whir_small_options(HashFamily::Blake3),
        )
        .expect("small Keccak WHIR proof must verify");
        assert_eq!(report.pcs, PcsIdentity::Whir);
        assert_eq!(report.width, NUM_KECCAK_BINARY_COLS);
        assert!(report.security_bits >= 95.0);
    }

    #[test]
    fn whir_proves_small_blake3_sorted_proper_successor_subset() {
        let air = ShapeAir {
            width: 3,
            next: vec![0, 2],
            public_values: 0,
            preprocessed_width: 0,
        };
        let trace = Table::new(RowMajorMatrix::new(vec![F::ZERO; 256 * 3], 3).transpose());
        let report = prove_boolean_air(
            &air,
            trace,
            BinaryProofOptions {
                folding: 3,
                ..whir_small_options(HashFamily::Blake3)
            },
        )
        .expect("small sorted proper-successor WHIR proof must verify");
        assert_eq!(report.pcs, PcsIdentity::Whir);
        assert!(report.security_bits >= 95.0);
    }

    #[test]
    fn whir_schedule_budgets_refuse_query_and_grinding_limits_independently() {
        let air = Blake3BinaryAir::default();
        let shape = TableShape::new(4, NUM_BLAKE3_BINARY_COLS);
        let mut query_options = whir_small_options(HashFamily::Blake3);
        let BooleanPcsChoice::Whir(query_whir) = query_options.pcs else {
            unreachable!()
        };
        let generous = boolean_whir_config::<_, Blake3>(&air, shape, query_options, query_whir)
            .expect("small WHIR config derives");
        let actual_queries = generous.summary.total_opened_positions;
        assert!(actual_queries > 0);
        let query_whir = WhirOptions {
            budget: BinaryWhirBudget {
                max_stir_queries: actual_queries - 1,
                ..query_whir.budget
            },
            ..query_whir
        };
        query_options.pcs = BooleanPcsChoice::Whir(query_whir);
        let query_error = boolean_whir_config::<_, Blake3>(&air, shape, query_options, query_whir)
            .expect_err("one fewer query must be refused before setup");
        assert!(matches!(
            query_error,
            BinaryProofError::WhirBudget(BudgetError::Queries { actual, budget })
                if actual == actual_queries && budget + 1 == actual_queries
        ));

        let mut grind_options = whir_preflight_options(HashFamily::Blake3, 102);
        let BooleanPcsChoice::Whir(grind_whir) = grind_options.pcs else {
            unreachable!()
        };
        let grind_whir = WhirOptions {
            regime: WhirRegime::Johnson,
            budget: BinaryWhirBudget {
                max_stir_queries: usize::MAX,
                max_proof_bytes: usize::MAX,
                max_grinding_bits: usize::MAX,
            },
            ..grind_whir
        };
        grind_options.pcs = BooleanPcsChoice::Whir(grind_whir);
        let grind_shape = TableShape::new(20, NUM_BLAKE3_BINARY_COLS);
        let uncapped = boolean_whir_config::<_, Blake3>(
            &Blake3BinaryAir::default(),
            grind_shape,
            grind_options,
            grind_whir,
        )
        .expect("uncapped Johnson schedule derives");
        let actual_grinding = uncapped.summary.max_grinding_bits;
        assert!(actual_grinding > 0);
        let grind_whir = WhirOptions {
            budget: BinaryWhirBudget {
                max_grinding_bits: actual_grinding - 1,
                ..grind_whir.budget
            },
            ..grind_whir
        };
        grind_options.pcs = BooleanPcsChoice::Whir(grind_whir);
        let grind_error = boolean_whir_config::<_, Blake3>(
            &Blake3BinaryAir::default(),
            grind_shape,
            grind_options,
            grind_whir,
        )
        .expect_err("grinding below the derived requirement must be refused");
        assert!(matches!(
            grind_error,
            BinaryProofError::WhirBudget(BudgetError::Grinding {
                actual,
                budget
            })
                if actual == actual_grinding && budget + 1 == actual_grinding
        ));
    }

    #[test]
    fn whir_complete_serialized_byte_budget_checks_actual_proof_length() {
        let air = Blake3BinaryAir::default();
        let words = air.generate_random_trace_packed::<Gf2>(4);
        let trace = Table::from_packed_bits(words, 2);
        let baseline =
            prove_boolean_air(&air, trace.clone(), whir_small_options(HashFamily::Blake3))
                .expect("baseline WHIR proof must verify");
        let proof_len = baseline.proof_bytes;
        let mut under = whir_small_options(HashFamily::Blake3);
        let BooleanPcsChoice::Whir(under_whir) = under.pcs else {
            unreachable!()
        };
        let under_whir = WhirOptions {
            budget: BinaryWhirBudget {
                max_proof_bytes: proof_len - 1,
                ..under_whir.budget
            },
            ..under_whir
        };
        under.pcs = BooleanPcsChoice::Whir(under_whir);
        let error = prove_boolean_air(&air, trace.clone(), under).unwrap_err();
        assert!(matches!(
            error,
            BinaryProofError::WhirBudget(BudgetError::Bytes { actual, budget })
                if actual == proof_len && budget + 1 == proof_len
        ));

        let mut exact = whir_small_options(HashFamily::Blake3);
        let BooleanPcsChoice::Whir(exact_whir) = exact.pcs else {
            unreachable!()
        };
        let exact_whir = WhirOptions {
            budget: BinaryWhirBudget {
                max_proof_bytes: proof_len,
                ..exact_whir.budget
            },
            ..exact_whir
        };
        exact.pcs = BooleanPcsChoice::Whir(exact_whir);
        let accepted = prove_boolean_air(&air, trace, exact)
            .expect("a byte budget equal to the complete proof length accepts");
        assert_eq!(accepted.proof_bytes, proof_len);
    }

    #[test]
    fn whir_tampered_opening_values_and_final_polynomial_are_rejected() {
        let air = Blake3BinaryAir::default();
        let table = Table::from_packed_bits(air.generate_random_trace_packed::<Gf2>(4), 2);
        let shape = table.shape();
        let options = whir_small_options(HashFamily::Blake3);
        let BooleanPcsChoice::Whir(whir) = options.pcs else {
            unreachable!()
        };
        let config = boolean_whir_config::<_, Blake3>(&air, shape, options, whir)
            .expect("small WHIR config derives");
        let (pk, vk, _) = setup_and_assess(&config, &air, shape, options)
            .expect("setup and security assessment succeed");
        let public_values: [F; 0] = [];
        let prover_instances =
            ProverInstances::new(vec![ProverInstance::new(&air, table, &pk, &public_values)]);
        let proof = Backend::preferred()
            .prove(
                &config,
                prover_instances,
                options.sumcheck_pow_bits,
                &mut binary_challenger(),
            )
            .expect("honest WHIR proof succeeds");
        let bytes = postcard::to_allocvec(&proof).expect("proof serialization succeeds");
        let proof: MultiStarkProof<BooleanWhirStarkConfig<Blake3>> =
            postcard::from_bytes(&bytes).expect("proof deserialization succeeds");
        let instances = || {
            VerifierInstances::new(vec![VerifierInstance::new(
                &air,
                &vk,
                shape.num_variables(),
                &public_values,
            )])
        };

        verify(
            &config,
            instances(),
            &proof,
            options.sumcheck_pow_bits,
            &mut binary_challenger(),
        )
        .expect("serialized honest WHIR proof verifies");

        let mut values_tampered: MultiStarkProof<BooleanWhirStarkConfig<Blake3>> =
            postcard::from_bytes(&bytes).expect("proof replay deserialization succeeds");
        values_tampered.opening.values[0] += F::ONE;
        let values_error = verify(
            &config,
            instances(),
            &values_tampered,
            options.sumcheck_pow_bits,
            &mut binary_challenger(),
        )
        .expect_err("tampered opening value must be rejected");
        assert!(matches!(
            values_error,
            VerificationError::Opening(BooleanTraceCommitmentError::Boolean(
                BooleanWhirError::ReductionProof(
                    p3_sumcheck::ring_switch::bits::BitRingSwitchProofError::ClaimMismatch
                )
            ))
        ));

        let mut final_tampered: MultiStarkProof<BooleanWhirStarkConfig<Blake3>> =
            postcard::from_bytes(&bytes).expect("proof replay deserialization succeeds");
        let final_poly = final_tampered
            .opening
            .opening
            .opening
            .whir
            .final_poly
            .as_mut()
            .expect("small WHIR proof has a final polynomial");
        final_poly.as_mut_slice()[0] += F::ONE;
        let final_error = verify(
            &config,
            instances(),
            &final_tampered,
            options.sumcheck_pow_bits,
            &mut binary_challenger(),
        )
        .expect_err("tampered WHIR final polynomial must be rejected");
        assert!(matches!(
            final_error,
            VerificationError::Opening(BooleanTraceCommitmentError::Boolean(
                BooleanWhirError::Opening(p3_whir::VerifierError::StirChallengeFailed {
                    challenge_id: 0,
                    details,
                })
            )) if details == "STIR constraint verification failed on final polynomial"
        ));
    }

    #[test]
    fn folding_parent_transcript_has_stable_small_blake3_baseline() {
        let air = Blake3BinaryAir::default();
        let table = Table::from_packed_bits(air.generate_random_trace_packed::<Gf2>(4), 2);
        let (bytes, next_challenge) = boolean_proof_transcript(&air, table, Backend::preferred());
        let digest = Blake3.hash_iter(bytes.iter().copied());
        println!(
            "folding parent baseline: proof_bytes={}, digest={digest:02x?}, next={next_challenge:?}",
            bytes.len()
        );
        assert_eq!(bytes.len(), 232_202);
        assert_eq!(
            digest,
            [
                0x93, 0x96, 0x4d, 0x10, 0x64, 0xe8, 0x35, 0x9a, 0x83, 0x26, 0xa0, 0xc8, 0xad, 0x4a,
                0xcd, 0x55, 0xb4, 0x33, 0x72, 0x72, 0xb6, 0x05, 0x66, 0x2b, 0xf6, 0x35, 0x99, 0x4c,
                0x2e, 0x16, 0x0f, 0xe5,
            ]
        );
        assert_eq!(
            next_challenge,
            F::from_repr(0xa377_5f1b_cdc0_002a_c595_02ac_d23e_90bd)
        );
    }

    #[test]
    fn whir_johnson_security_report_exposes_opening_and_outer_terms() {
        let base = whir_small_options(HashFamily::Blake3);
        let BooleanPcsChoice::Whir(whir) = base.pcs else {
            unreachable!()
        };
        let options = BinaryProofOptions {
            pcs: BooleanPcsChoice::Whir(WhirOptions {
                regime: WhirRegime::Johnson,
                ..whir
            }),
            ..base
        };
        let air = Blake3BinaryAir::default();
        let shape = TableShape::new(4, NUM_BLAKE3_BINARY_COLS);
        let BooleanPcsChoice::Whir(whir) = options.pcs else {
            unreachable!()
        };
        let config = boolean_whir_config::<_, Blake3>(&air, shape, options, whir)
            .expect("Johnson small config must derive");
        let (_, vk) = setup(&config, &[&air], &mut binary_challenger()).expect("setup succeeds");
        let public_values: [F; 0] = [];
        let instances = VerifierInstances::new(vec![VerifierInstance::new(
            &air,
            &vk,
            shape.num_variables(),
            &public_values,
        )]);
        let report = security_report(&config, &instances).expect("security report succeeds");
        println!("Johnson security terms: {:?}", report.terms());
        assert!(
            report
                .terms()
                .iter()
                .any(|term| term.label == "whir-opening")
        );
        assert!(
            report
                .terms()
                .iter()
                .any(|term| term.label == "bit-ring-switch")
        );
        assert!(
            report
                .terms()
                .iter()
                .any(|term| term.label == "column-batching")
        );
        assert!(
            report
                .terms()
                .iter()
                .any(|term| term.label == "commitment-and-transcript-collision")
        );
        for label in ["constraint-batching", "zerocheck", "constraint-sumcheck"] {
            assert!(
                report.terms().iter().any(|term| term.label == label),
                "missing security term {label}"
            );
        }

        let unique_options = base;
        let BooleanPcsChoice::Whir(unique_whir) = unique_options.pcs else {
            unreachable!()
        };
        let unique_config =
            boolean_whir_config::<_, Blake3>(&air, shape, unique_options, unique_whir)
                .expect("unique small config must derive");
        let (_, unique_vk) =
            setup(&unique_config, &[&air], &mut binary_challenger()).expect("setup succeeds");
        let unique_instances = VerifierInstances::new(vec![VerifierInstance::new(
            &air,
            &unique_vk,
            shape.num_variables(),
            &public_values,
        )]);
        let unique_report =
            security_report(&unique_config, &unique_instances).expect("security report succeeds");
        let bits_for = |report: &p3_multi_stark::MultiStarkSecurityReport, label| {
            report
                .terms()
                .iter()
                .find(|term| term.label == label)
                .expect("security term is present")
                .bits
                .bits()
        };
        let unique_batching = bits_for(&unique_report, "constraint-batching");
        let johnson_batching = bits_for(&report, "constraint-batching");
        let expected_charge =
            p3_security::SecurityAssumption::JohnsonBound.list_size_bits(shape.num_variables(), 1);
        assert!((unique_batching - johnson_batching - expected_charge).abs() < 1e-9);
    }

    #[test]
    fn whir_johnson_grinding_budget_is_checked_before_witness_generation() {
        let base = BinaryProofOptions {
            pcs: BooleanPcsChoice::Whir(WhirOptions {
                regime: WhirRegime::Johnson,
                term_security_bits: 102,
                budget: BinaryWhirBudget {
                    max_stir_queries: usize::MAX,
                    max_proof_bytes: usize::MAX,
                    max_grinding_bits: usize::MAX,
                },
            }),
            security_bits: 95,
            log_inv_rate: 1,
            folding: 4,
            merkle_arity: 2,
            ..BinaryProofOptions::default()
        };
        let BooleanPcsChoice::Whir(whir) = base.pcs else {
            unreachable!()
        };
        let air = Blake3BinaryAir::default();
        let shape = TableShape::new(20, 11_536);
        let generous = boolean_whir_config::<_, Blake3>(&air, shape, base, whir)
            .expect("the uncapped Johnson schedule must derive");
        let actual = generous.summary.max_grinding_bits;
        assert!(actual > 0);
        let options = BinaryProofOptions {
            pcs: BooleanPcsChoice::Whir(WhirOptions {
                budget: BinaryWhirBudget {
                    max_grinding_bits: actual - 1,
                    ..whir.budget
                },
                ..whir
            }),
            ..base
        };
        let BooleanPcsChoice::Whir(whir) = options.pcs else {
            unreachable!()
        };
        let error = match boolean_whir_config::<_, Blake3>(&air, shape, options, whir) {
            Ok(_) => panic!("Johnson l20 must exceed the derived grinding cap g-1 (g={actual})"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            BinaryProofError::WhirBudget(BudgetError::Grinding {
                actual: reported,
                budget
            }) if reported == actual && budget == actual - 1
        ));
    }

    #[test]
    fn whir_rejects_each_incompatible_geometry_option_with_its_payload() {
        let cases = [
            (
                BinaryProofOptions {
                    merkle_arity: 4,
                    ..whir_test_options()
                },
                WhirIncompatibility::MerkleArity { actual: 4 },
            ),
            (
                BinaryProofOptions {
                    leaf_elements: Some(16),
                    ..whir_test_options()
                },
                WhirIncompatibility::LeafElements { actual: Some(16) },
            ),
            (
                BinaryProofOptions {
                    pcs_pow_bits: 1,
                    ..whir_test_options()
                },
                WhirIncompatibility::PcsPowBits { actual: 1 },
            ),
            (
                BinaryProofOptions {
                    log_inv_rate: 0,
                    ..whir_test_options()
                },
                WhirIncompatibility::NonRedundantRate { actual: 0 },
            ),
            (
                BinaryProofOptions {
                    folding: 0,
                    ..whir_test_options()
                },
                WhirIncompatibility::ZeroFolding,
            ),
            (
                BinaryProofOptions {
                    folding: 4,
                    ..whir_test_options()
                },
                WhirIncompatibility::FoldingExceeds {
                    requested: 4,
                    committed: 3,
                },
            ),
        ];
        for (options, expected) in cases {
            let actual = whir_error(
                &ShapeAir {
                    width: 3,
                    next: vec![],
                    public_values: 0,
                    preprocessed_width: 0,
                },
                small_whir_shape(),
                options,
            );
            assert!(matches!(
                (actual, expected),
                (
                    BinaryProofError::WhirIncompatible(actual),
                    expected
                ) if actual == expected
            ));
        }
    }

    #[test]
    fn whir_rejects_dense_trace_selection_before_binary_setup() {
        let error = prove_binary_air(
            &RecurrenceAir,
            recurrence_trace(4),
            BinaryProofOptions {
                pcs: BooleanPcsChoice::Whir(WhirOptions {
                    regime: WhirRegime::UniqueDecoding,
                    term_security_bits: 98,
                    budget: BinaryWhirBudget::PRODUCTION,
                }),
                ..BinaryProofOptions::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            BinaryProofError::WhirIncompatible(WhirIncompatibility::DenseField)
        ));
    }

    #[test]
    fn whir_rejects_public_preprocessed_and_overflowing_shapes_before_pricing() {
        let public = whir_error(
            &ShapeAir {
                width: 3,
                next: vec![],
                public_values: 1,
                preprocessed_width: 0,
            },
            small_whir_shape(),
            whir_test_options(),
        );
        assert!(matches!(
            public,
            BinaryProofError::WhirIncompatible(WhirIncompatibility::PublicValues { actual: 1 })
        ));

        let preprocessed = whir_error(
            &ShapeAir {
                width: 3,
                next: vec![],
                public_values: 0,
                preprocessed_width: 1,
            },
            small_whir_shape(),
            whir_test_options(),
        );
        assert!(matches!(
            preprocessed,
            BinaryProofError::WhirIncompatible(WhirIncompatibility::PreprocessedColumns {
                actual: 1
            })
        ));

        let overflowing = whir_error(
            &Blake3BinaryAir::default(),
            TableShape::new(usize::BITS as usize - 1, usize::MAX),
            whir_test_options(),
        );
        assert!(matches!(
            overflowing,
            BinaryProofError::WhirIncompatible(WhirIncompatibility::ShapeOverflow { .. })
        ));
    }

    #[test]
    fn whir_successor_claim_routes_cover_current_full_and_sorted_subset() {
        let current = boolean_whir_config::<_, Blake3>(
            &ShapeAir {
                width: 3,
                next: vec![],
                public_values: 0,
                preprocessed_width: 0,
            },
            small_whir_shape(),
            whir_test_options(),
            match whir_test_options().pcs {
                BooleanPcsChoice::Whir(options) => options,
                BooleanPcsChoice::Folding => unreachable!(),
            },
        )
        .expect("current-only successor declarations configure");
        let full_options = whir_test_options();
        let full = boolean_whir_config::<_, Blake3>(
            &ShapeAir {
                width: 3,
                next: vec![0, 1, 2],
                public_values: 0,
                preprocessed_width: 0,
            },
            small_whir_shape(),
            full_options,
            match full_options.pcs {
                BooleanPcsChoice::Whir(options) => options,
                BooleanPcsChoice::Folding => unreachable!(),
            },
        )
        .expect("ordered full successor declarations configure");
        let subset_options = whir_test_options();
        let subset = boolean_whir_config::<_, Blake3>(
            &ShapeAir {
                width: 3,
                next: vec![0, 2],
                public_values: 0,
                preprocessed_width: 0,
            },
            small_whir_shape(),
            subset_options,
            match subset_options.pcs {
                BooleanPcsChoice::Whir(options) => options,
                BooleanPcsChoice::Folding => unreachable!(),
            },
        )
        .expect("sorted proper successor declarations configure");
        assert_eq!(
            current.summary.total_opened_positions,
            full.summary.total_opened_positions
        );
        assert!(
            current.summary.pcs_payload_bytes < full.summary.pcs_payload_bytes,
            "current={:?}, full={:?}",
            current.summary,
            full.summary
        );
        assert!(
            full.summary.pcs_payload_bytes < subset.summary.pcs_payload_bytes,
            "full={:?}, subset={:?}",
            full.summary,
            subset.summary
        );
    }

    #[test]
    fn whir_successor_tensor_admission_uses_row_variables_not_width() {
        let current_options = whir_test_options();
        let current = boolean_whir_config::<_, Blake3>(
            &ShapeAir {
                width: 8,
                next: vec![],
                public_values: 0,
                preprocessed_width: 0,
            },
            TableShape::new(7, 8),
            current_options,
            match current_options.pcs {
                BooleanPcsChoice::Whir(options) => options,
                BooleanPcsChoice::Folding => unreachable!(),
            },
        )
        .expect("current-only declarations configure");
        let full_options = whir_test_options();
        let full = boolean_whir_config::<_, Blake3>(
            &ShapeAir {
                width: 8,
                next: (0..8).collect(),
                public_values: 0,
                preprocessed_width: 0,
            },
            TableShape::new(7, 8),
            full_options,
            match full_options.pcs {
                BooleanPcsChoice::Whir(options) => options,
                BooleanPcsChoice::Folding => unreachable!(),
            },
        )
        .expect("ordered full declarations configure");
        assert_eq!(
            current.summary.total_opened_positions,
            full.summary.total_opened_positions
        );
        assert_eq!(
            current.summary.pcs_payload_bytes,
            full.summary.pcs_payload_bytes
        );
    }

    #[test]
    fn whir_rejects_conflicting_embedded_and_explicit_options() {
        let options = whir_test_options();
        let BooleanPcsChoice::Whir(embedded) = options.pcs else {
            unreachable!()
        };
        let explicit = WhirOptions {
            term_security_bits: embedded.term_security_bits + 1,
            ..embedded
        };
        let error = boolean_whir_config::<_, Blake3>(
            &ShapeAir {
                width: 3,
                next: vec![],
                public_values: 0,
                preprocessed_width: 0,
            },
            small_whir_shape(),
            options,
            explicit,
        )
        .expect_err("conflicting WHIR option sources must be rejected");
        assert!(matches!(
            error,
            BinaryProofError::WhirIncompatible(
                WhirIncompatibility::PcsChoiceMismatch {
                    embedded: Some(actual),
                    explicit: expected,
                }
            ) if actual == embedded && expected == explicit
        ));
    }

    #[test]
    fn whir_rejects_permuted_duplicate_and_out_of_range_successors_before_budget_pricing() {
        let mut options = whir_test_options();
        let BooleanPcsChoice::Whir(mut whir) = options.pcs else {
            unreachable!()
        };
        whir.budget.max_stir_queries = 0;
        options.pcs = BooleanPcsChoice::Whir(whir);
        for (next, expected) in [
            (
                vec![1, 0, 2],
                WhirIncompatibility::SuccessorNotStrict {
                    previous: 1,
                    current: 0,
                },
            ),
            (
                vec![0, 0, 1],
                WhirIncompatibility::SuccessorNotStrict {
                    previous: 0,
                    current: 0,
                },
            ),
            (
                vec![0, 3],
                WhirIncompatibility::SuccessorOutOfRange {
                    column: 3,
                    width: 3,
                },
            ),
        ] {
            let error = whir_error(
                &ShapeAir {
                    width: 3,
                    next,
                    public_values: 0,
                    preprocessed_width: 0,
                },
                small_whir_shape(),
                options,
            );
            assert!(matches!(
                (error, expected),
                (BinaryProofError::WhirIncompatible(actual), expected) if actual == expected
            ));
        }
    }

    #[test]
    fn whir_rejects_each_symbolic_declaration_family_independently() {
        let cases: [(&dyn Fn() -> BinaryProofError, WhirInteractionFamily); 5] = [
            (
                &|| {
                    whir_error(
                        &LocalInteractionAir,
                        small_whir_shape(),
                        whir_test_options(),
                    )
                },
                WhirInteractionFamily::Local,
            ),
            (
                &|| {
                    whir_error(
                        &GlobalInteractionAir,
                        small_whir_shape(),
                        whir_test_options(),
                    )
                },
                WhirInteractionFamily::Global,
            ),
            (
                &|| {
                    whir_error(
                        &ExclusiveInteractionAir,
                        small_whir_shape(),
                        whir_test_options(),
                    )
                },
                WhirInteractionFamily::Exclusive,
            ),
            (
                &|| whir_error(&IndexedReadAir, small_whir_shape(), whir_test_options()),
                WhirInteractionFamily::IndexedRead,
            ),
            (
                &|| whir_error(&IndexedTableAir, small_whir_shape(), whir_test_options()),
                WhirInteractionFamily::IndexedTable,
            ),
        ];
        for (make_error, family) in cases {
            let error = make_error();
            assert!(matches!(
                error,
                BinaryProofError::WhirIncompatible(
                    WhirIncompatibility::UnsupportedInteractions(actual)
                ) if actual == family
            ));
        }
    }

    #[test]
    fn whir_rejects_binary_bus_before_budget_or_witness() {
        let profile = BinaryBusAir {
            direction: BusDirection::Push,
            conditional: false,
        };
        let layout = AirLayout::from_air::<F>(&profile);
        let legacy = InteractionSymbolicBuilder::<F, F>::from_air(&profile, layout);
        assert!(legacy.local_interactions().is_empty());
        assert!(legacy.global_interactions().is_empty());
        assert!(legacy.exclusive_interactions().is_empty());
        assert!(legacy.indexed_reads().is_empty());
        assert!(legacy.indexed_tables().is_empty());
        assert_eq!(
            BusSymbolicBuilder::<F, F>::from_air(&profile, layout)
                .interactions()
                .len(),
            1
        );
        for direction in [BusDirection::Push, BusDirection::Pull] {
            for conditional in [false, true] {
                let mut options = whir_test_options();
                let BooleanPcsChoice::Whir(mut whir) = options.pcs else {
                    unreachable!()
                };
                whir.budget.max_stir_queries = 0;
                options.pcs = BooleanPcsChoice::Whir(whir);
                let error = preflight_boolean_air(
                    &BinaryBusAir {
                        direction,
                        conditional,
                    },
                    small_whir_shape(),
                    options,
                )
                .expect_err("binary bus declarations are outside WHIR scope");
                assert!(matches!(
                    error,
                    BinaryProofError::WhirIncompatible(
                        WhirIncompatibility::UnsupportedInteractions(
                            WhirInteractionFamily::BinaryBus
                        )
                    )
                ));
            }
        }
    }

    #[test]
    fn whir_symbolic_refusals_precede_zero_query_pricing() {
        let mut options = whir_test_options();
        let BooleanPcsChoice::Whir(mut whir) = options.pcs else {
            unreachable!()
        };
        whir.budget.max_stir_queries = 0;
        options.pcs = BooleanPcsChoice::Whir(whir);

        let exclusive = whir_error(&ExclusiveInteractionAir, small_whir_shape(), options);
        assert!(matches!(
            exclusive,
            BinaryProofError::WhirIncompatible(WhirIncompatibility::UnsupportedInteractions(
                WhirInteractionFamily::Exclusive
            ))
        ));
        let indexed_table = whir_error(&IndexedTableAir, small_whir_shape(), options);
        assert!(matches!(
            indexed_table,
            BinaryProofError::WhirIncompatible(WhirIncompatibility::UnsupportedInteractions(
                WhirInteractionFamily::IndexedTable
            ))
        ));
    }

    #[test]
    fn folding_config_keeps_accepting_whir_only_successor_restrictions() {
        let air = ShapeAir {
            width: 3,
            next: vec![1, 0, 2],
            public_values: 0,
            preprocessed_width: 0,
        };
        let trace = Table::new(RowMajorMatrix::new(vec![F::ZERO; 64 * 3], 3).transpose());
        let report = prove_boolean_air(
            &air,
            trace,
            BinaryProofOptions {
                hash: HashFamily::Blake3,
                security_bits: 10,
                ..BinaryProofOptions::default()
            },
        )
        .expect("folding does not apply WHIR successor restrictions");
        assert_eq!(report.width, 3);
    }

    /// A nonlinear recurrence over `BinaryField128`: `(a, b) -> (b, a * b + a)`.
    ///
    /// Addition is XOR and multiplication is tower-field multiplication. Nonlinear
    /// constraints exercise interpolation beyond the two prime-subfield elements.
    struct RecurrenceAir;

    impl<F> BaseAir<F> for RecurrenceAir {
        fn width(&self) -> usize {
            2
        }
    }

    impl<AB: AirBuilder> Air<AB> for RecurrenceAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.current_slice();
            let next = main.next_slice();
            builder.when_transition().assert_eq(next[0], local[1]);
            builder
                .when_transition()
                .assert_eq(next[1], local[0] * local[1] + local[0]);
        }
    }

    fn recurrence_trace(log_height: usize) -> RowMajorMatrix<F> {
        let mut a = F::from_repr(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210);
        let mut b = F::from_repr(0xfedc_ba98_7654_3210_0123_4567_89ab_cdef);
        let mut values = Vec::with_capacity(2 << log_height);
        for _ in 0..1 << log_height {
            values.extend([a, b]);
            (a, b) = (b, a * b + a);
        }
        RowMajorMatrix::new(values, 2)
    }

    #[derive(Clone, Debug)]
    struct ShapeAir {
        width: usize,
        next: Vec<usize>,
        public_values: usize,
        preprocessed_width: usize,
    }

    impl<X> BaseAir<X> for ShapeAir {
        fn width(&self) -> usize {
            self.width
        }

        fn num_public_values(&self) -> usize {
            self.public_values
        }

        fn preprocessed_width(&self) -> usize {
            self.preprocessed_width
        }

        fn main_next_row_columns(&self) -> Vec<usize> {
            self.next.clone()
        }
    }

    impl<AB: AirBuilder> Air<AB> for ShapeAir {
        fn eval(&self, builder: &mut AB) {
            builder.assert_zero(builder.main().current_slice()[0]);
        }
    }

    struct GlobalInteractionAir;

    impl<X> BaseAir<X> for GlobalInteractionAir {
        fn width(&self) -> usize {
            3
        }
    }

    impl<AB> Air<AB> for GlobalInteractionAir
    where
        AB: AirBuilder + InteractionBuilder,
    {
        fn eval(&self, builder: &mut AB) {
            let value = builder.main().current_slice()[0];
            builder.push_interaction("global", [value], Count::bounded(AB::Expr::ONE, 1));
        }
    }

    struct LocalInteractionAir;

    impl<X> BaseAir<X> for LocalInteractionAir {
        fn width(&self) -> usize {
            3
        }
    }

    impl<AB> Air<AB> for LocalInteractionAir
    where
        AB: AirBuilder + InteractionBuilder,
    {
        fn eval(&self, builder: &mut AB) {
            let value = builder.main().current_slice()[0];
            builder.push_local_interaction([(vec![value.into()], Count::provided(AB::Expr::ONE))]);
        }
    }

    struct ExclusiveInteractionAir;

    impl<X> BaseAir<X> for ExclusiveInteractionAir {
        fn width(&self) -> usize {
            3
        }
    }

    impl<AB> Air<AB> for ExclusiveInteractionAir
    where
        AB: AirBuilder + InteractionBuilder,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let flag = main.current_slice()[0];
            let value = main.current_slice()[1];
            builder.push_exclusive_interaction(
                "exclusive",
                [(
                    flag.into(),
                    Count::bounded(AB::Expr::ONE, 1),
                    vec![value.into()],
                )],
            );
        }
    }

    struct IndexedReadAir;

    impl<X> BaseAir<X> for IndexedReadAir {
        fn width(&self) -> usize {
            3
        }
    }

    impl<AB> Air<AB> for IndexedReadAir
    where
        AB: AirBuilder + IndexedLookupBuilder,
    {
        fn eval(&self, builder: &mut AB) {
            builder.push_indexed_read("table", 0, [1]);
        }
    }

    struct IndexedTableAir;

    impl<X> BaseAir<X> for IndexedTableAir {
        fn width(&self) -> usize {
            3
        }
    }

    struct BinaryBusAir {
        direction: BusDirection,
        conditional: bool,
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "vpclmulqdq",
        any(target_feature = "avx2", target_feature = "avx512f")
    ))]
    type BinaryBusPacked = PackedExt<F, p3_binary_field::PackedGhash128>;
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "vpclmulqdq",
        any(target_feature = "avx2", target_feature = "avx512f")
    )))]
    type BinaryBusPacked = PackedExt<F, Ghash128>;

    impl<X> BaseAir<X> for BinaryBusAir {
        fn width(&self) -> usize {
            3
        }
    }

    fn eval_bus<AB>(air: &BinaryBusAir, builder: &mut AB)
    where
        AB: AirBuilder + BusInteractionBuilder,
    {
        let main = builder.main();
        let activation = if air.conditional {
            BusActivation::Boolean(main.current_slice()[1].into())
        } else {
            BusActivation::Always
        };
        builder.push_bus_interaction(
            BusName::new("whir-test-bus"),
            air.direction,
            [main.current_slice()[0]],
            activation,
        );
    }

    impl Air<BusSymbolicBuilder<F, F>> for BinaryBusAir {
        fn eval(&self, builder: &mut BusSymbolicBuilder<F, F>) {
            eval_bus(self, builder);
        }
    }

    impl Air<InteractionSymbolicBuilder<F, F>> for BinaryBusAir {
        fn eval(&self, builder: &mut InteractionSymbolicBuilder<F, F>) {
            eval_bus(self, builder);
        }
    }

    impl<'a> Air<MultilinearFolder<'a, F, F, F>> for BinaryBusAir {
        fn eval(&self, builder: &mut MultilinearFolder<'a, F, F, F>) {
            eval_bus(self, builder);
        }
    }

    impl<'a> Air<MultilinearFolder<'a, F, PackedExt<F, F>, PackedExt<F, F>>> for BinaryBusAir {
        fn eval(&self, builder: &mut MultilinearFolder<'a, F, PackedExt<F, F>, PackedExt<F, F>>) {
            eval_bus(self, builder);
        }
    }

    impl<'a> Air<InteractionMultilinearFolder<'a, F, F, F>> for BinaryBusAir {
        fn eval(&self, builder: &mut InteractionMultilinearFolder<'a, F, F, F>) {
            eval_bus(self, builder);
        }
    }

    impl<'a> Air<InteractionMultilinearFolder<'a, F, PackedExt<F, F>, PackedExt<F, F>>>
        for BinaryBusAir
    {
        fn eval(
            &self,
            builder: &mut InteractionMultilinearFolder<'a, F, PackedExt<F, F>, PackedExt<F, F>>,
        ) {
            eval_bus(self, builder);
        }
    }

    impl<'a>
        Air<MultilinearFolder<'a, F, SubfieldVar<F, BinaryField2>, SubfieldAcc<F, BinaryField2>>>
        for BinaryBusAir
    {
        fn eval(
            &self,
            _builder: &mut MultilinearFolder<
                'a,
                F,
                SubfieldVar<F, BinaryField2>,
                SubfieldAcc<F, BinaryField2>,
            >,
        ) {
        }
    }

    impl<'a> Air<SlicedFolder<'a, F, BinaryField2, F>> for BinaryBusAir {
        fn eval(&self, _builder: &mut SlicedFolder<'a, F, BinaryField2, F>) {}
    }

    impl<'a> Air<SlicedFolder<'a, F, BinaryField2, Ghash128>> for BinaryBusAir {
        fn eval(&self, _builder: &mut SlicedFolder<'a, F, BinaryField2, Ghash128>) {}
    }

    impl<'a> Air<MultilinearFolder<'a, F, Ghash128, Ghash128>> for BinaryBusAir {
        fn eval(&self, builder: &mut MultilinearFolder<'a, F, Ghash128, Ghash128>) {
            eval_bus(self, builder);
        }
    }

    impl<'a> Air<InteractionMultilinearFolder<'a, F, Ghash128, Ghash128>> for BinaryBusAir {
        fn eval(&self, builder: &mut InteractionMultilinearFolder<'a, F, Ghash128, Ghash128>) {
            eval_bus(self, builder);
        }
    }

    impl<'a> Air<MultilinearFolder<'a, F, BinaryBusPacked, BinaryBusPacked>> for BinaryBusAir {
        fn eval(&self, builder: &mut MultilinearFolder<'a, F, BinaryBusPacked, BinaryBusPacked>) {
            eval_bus(self, builder);
        }
    }

    impl<'a> Air<InteractionMultilinearFolder<'a, F, BinaryBusPacked, BinaryBusPacked>>
        for BinaryBusAir
    {
        fn eval(
            &self,
            builder: &mut InteractionMultilinearFolder<'a, F, BinaryBusPacked, BinaryBusPacked>,
        ) {
            eval_bus(self, builder);
        }
    }

    impl<AB> Air<AB> for IndexedTableAir
    where
        AB: AirBuilder + IndexedLookupBuilder,
    {
        fn eval(&self, builder: &mut AB) {
            builder.push_indexed_table("table", TraceWindow::Main, [0]);
        }
    }

    fn whir_test_options() -> BinaryProofOptions {
        BinaryProofOptions {
            pcs: BooleanPcsChoice::Whir(WhirOptions {
                regime: WhirRegime::UniqueDecoding,
                term_security_bits: 98,
                budget: BinaryWhirBudget {
                    max_stir_queries: usize::MAX,
                    max_proof_bytes: usize::MAX,
                    max_grinding_bits: usize::MAX,
                },
            }),
            security_bits: 95,
            log_inv_rate: 1,
            folding: 1,
            merkle_arity: 2,
            ..BinaryProofOptions::default()
        }
    }

    fn whir_preflight_options(hash: HashFamily, term_security_bits: usize) -> BinaryProofOptions {
        BinaryProofOptions {
            pcs: BooleanPcsChoice::Whir(WhirOptions {
                regime: WhirRegime::UniqueDecoding,
                term_security_bits,
                budget: BinaryWhirBudget {
                    max_stir_queries: usize::MAX,
                    max_proof_bytes: usize::MAX,
                    max_grinding_bits: 0,
                },
            }),
            security_bits: 95,
            log_inv_rate: 1,
            folding: 4,
            merkle_arity: 2,
            hash,
            ..BinaryProofOptions::default()
        }
    }

    fn whir_small_options(hash: HashFamily) -> BinaryProofOptions {
        BinaryProofOptions {
            pcs: BooleanPcsChoice::Whir(WhirOptions {
                regime: WhirRegime::UniqueDecoding,
                term_security_bits: 102,
                budget: BinaryWhirBudget {
                    max_stir_queries: usize::MAX,
                    max_proof_bytes: usize::MAX,
                    max_grinding_bits: usize::MAX,
                },
            }),
            security_bits: 95,
            log_inv_rate: 1,
            folding: 4,
            merkle_arity: 2,
            hash,
            ..BinaryProofOptions::default()
        }
    }

    fn small_whir_shape() -> TableShape {
        TableShape::new(8, 3)
    }

    fn whir_error<A: BinaryAir>(
        air: &A,
        shape: TableShape,
        options: BinaryProofOptions,
    ) -> BinaryProofError {
        let BooleanPcsChoice::Whir(whir) = options.pcs else {
            unreachable!()
        };
        boolean_whir_config::<_, Blake3>(air, shape, options, whir).unwrap_err()
    }

    #[test]
    fn proves_and_verifies_a_tiny_binary_air() {
        let log_height = 4;
        let trace = recurrence_trace(log_height);
        let report = prove_binary_air(&RecurrenceAir, trace, BinaryProofOptions::default())
            .expect("a tiny binary AIR proof must verify");
        assert_eq!(report.rows, 1 << log_height);
        assert_eq!(report.width, 2);
        assert_eq!(report.stacked_variables, log_height + 1);
        assert!(report.security_bits >= 100.0);
    }

    /// The serialized proof of `air` on `trace`, then the next challenge its transcript draws.
    ///
    /// With a backend the zerocheck runs through it, otherwise through [`prove`].
    fn proof_transcript<A: BinaryAir>(
        air: &A,
        trace: &RowMajorMatrix<F>,
        backend: Option<Backend>,
    ) -> (Vec<u8>, F) {
        let arity = log2_strict_usize(trace.height()) + log2_ceil_usize(trace.width());
        let params = BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 0,
            security_level: 100,
        };
        let config = binary_config::<2, PolyBasisNtt, Keccak256Hash>(
            arity,
            params,
            3,
            None,
            PolyBasisNtt::default(),
        )
        .expect("the test shape configures the PCS");
        let (pk, _) = setup(&config, &[air], &mut binary_challenger()).expect("setup succeeds");

        let public_values: [F; 0] = [];
        let instances = ProverInstances::new(vec![ProverInstance::new(
            air,
            Table::new(trace.clone().transpose()),
            &pk,
            &public_values,
        )]);
        let mut challenger = binary_challenger();
        let proof = match backend {
            Some(backend) => backend.prove(&config, instances, 0, &mut challenger),
            None => prove(&config, instances, 0, &mut challenger),
        }
        .expect("an honest trace proves");
        let bytes = postcard::to_allocvec(&proof).expect("postcard serialization must not fail");
        (bytes, CanSample::<F>::sample(&mut challenger))
    }

    /// Rows of the shortest stage [`Backend::PolyBasisLate`] defers: `p3-multi-stark` names the
    /// exponent `MIN_LATE_BOUNDARY_VARS`.
    ///
    /// The harness traces sit below it, so their [`Backend::PolyBasisLate`] cases pin the
    /// fallback that backend takes there, not the deferral.
    const LATE_BOUNDARY_FLOOR: usize = 1 << 11;

    /// The Boolean-committed proof and next transcript challenge for a table representation.
    fn boolean_proof_transcript<A: BinaryAir>(
        air: &A,
        table: Table<F>,
        backend: Backend,
    ) -> (Vec<u8>, F) {
        assert!(
            1 << table.num_variables() < LATE_BOUNDARY_FLOOR,
            "a harness table this tall would exercise the deferral, not its fallback"
        );
        let shape = table.shape();
        let config = boolean_config::<2, Keccak256Hash>(
            shape,
            BinaryPcsParams {
                log_inv_rate: 1,
                pow_bits: 0,
                security_level: 100,
            },
            4,
            None,
        )
        .expect("the test shape configures the Boolean PCS");
        let (pk, _) = setup(&config, &[air], &mut binary_challenger()).expect("setup succeeds");
        let public_values: [F; 0] = [];
        let instances =
            ProverInstances::new(vec![ProverInstance::new(air, table, &pk, &public_values)]);
        let mut challenger = binary_challenger();
        let proof = backend
            .prove(&config, instances, 0, &mut challenger)
            .expect("an honest Boolean trace proves");
        let bytes = postcard::to_allocvec(&proof).expect("postcard serialization must not fail");
        (bytes, CanSample::<F>::sample(&mut challenger))
    }

    /// Whether every cell of `trace` lies in `GF(4)`, the cell condition for the subfield kernels.
    fn cells_fit_gf4(trace: &RowMajorMatrix<F>) -> bool {
        <F as HasSubfield<BinaryField2>>::all_in_subfield(&trace.values)
    }

    /// Require every harness backend to emit the proof and transcript of [`prove`].
    ///
    /// The trace sits below [`LATE_BOUNDARY_FLOOR`], so [`Backend::PolyBasisLate`] runs its
    /// fallback here.
    fn assert_backends_prove_byte_for_byte<A: BinaryAir>(air: &A, trace: &RowMajorMatrix<F>) {
        assert!(
            trace.height() < LATE_BOUNDARY_FLOOR,
            "a harness trace this tall would exercise the deferral, not its fallback"
        );
        let generic = proof_transcript(air, trace, None);
        for backend in [
            Backend::Subfield,
            Backend::PolyBasis,
            Backend::PolyBasisLate,
        ] {
            assert_eq!(
                proof_transcript(air, trace, Some(backend)),
                generic,
                "{backend:?}"
            );
        }
    }

    #[test]
    fn backends_prove_the_keccak_air_byte_for_byte() {
        // One permutation pads to 32 bit-valued rows of a degree-three AIR with successor columns.
        let air = KeccakBinaryAir::default();
        let trace = air.generate_random_trace_rows::<F>(1, 0);
        assert_eq!(trace.height(), 32);
        assert!(cells_fit_gf4(&trace));
        assert_backends_prove_byte_for_byte(&air, &trace);
    }

    #[test]
    fn backends_prove_the_blake3_air_byte_for_byte() {
        // Four compressions, one bit-valued row each, of a degree-two AIR.
        let air = Blake3BinaryAir::default();
        let trace = air.generate_random_trace_rows::<F>(4, 0);
        assert!(cells_fit_gf4(&trace));
        assert_backends_prove_byte_for_byte(&air, &trace);
    }

    #[test]
    fn backends_prove_the_sha256_air_byte_for_byte() {
        // Four compressions, one bit-valued row each, of a degree-two AIR twice Blake3's width.
        let air = Sha256BinaryAir::default();
        let trace = air.generate_random_trace_rows::<F>(4, 0);
        assert!(cells_fit_gf4(&trace));
        assert_backends_prove_byte_for_byte(&air, &trace);
    }

    #[test]
    fn dense_and_packed_sha256_tables_have_identical_boolean_proofs() {
        let air = Sha256BinaryAir::assuming_boolean_trace();
        let dense = Table::new(air.generate_random_trace_rows::<F>(4, 0).transpose());
        let packed = Table::from_packed_bits(air.generate_random_trace_packed::<Gf2>(4), 2);
        for backend in [
            Backend::Subfield,
            Backend::PolyBasis,
            Backend::PolyBasisLate,
        ] {
            assert_eq!(
                boolean_proof_transcript(&air, dense.clone(), backend),
                boolean_proof_transcript(&air, packed.clone(), backend),
                "{backend:?}"
            );
        }
    }

    #[test]
    fn dense_and_packed_blake3_tables_have_identical_boolean_proofs() {
        let air = Blake3BinaryAir::assuming_boolean_trace();
        let dense = Table::new(air.generate_random_trace_rows::<F>(4, 0).transpose());
        let packed = Table::from_packed_bits(air.generate_random_trace_packed::<Gf2>(4), 2);
        for backend in [
            Backend::Subfield,
            Backend::PolyBasis,
            Backend::PolyBasisLate,
        ] {
            assert_eq!(
                boolean_proof_transcript(&air, dense.clone(), backend),
                boolean_proof_transcript(&air, packed.clone(), backend),
                "{backend:?}"
            );
        }
    }

    #[test]
    fn dense_and_packed_keccak_tables_have_identical_boolean_proofs() {
        // Three permutations fill 75 of 128 rows: the packed blocks hold permutations that
        // straddle block boundaries and padding rows.
        let air = KeccakBinaryAir::assuming_boolean_trace();
        let dense = Table::new(air.generate_random_trace_rows::<F>(3, 0).transpose());
        let packed = Table::from_packed_bits(air.generate_random_trace_packed::<Gf2>(3), 7);
        for backend in [
            Backend::Subfield,
            Backend::PolyBasis,
            Backend::PolyBasisLate,
        ] {
            assert_eq!(
                boolean_proof_transcript(&air, dense.clone(), backend),
                boolean_proof_transcript(&air, packed.clone(), backend),
                "{backend:?}"
            );
        }
    }

    #[test]
    fn backends_prove_a_full_width_trace_byte_for_byte() {
        // The recurrence starts from full-width cells, so its stage cannot fit `GF(4)`: its first
        // round runs the generic kernel, and its later rounds run in each backend's field.
        let trace = recurrence_trace(4);
        assert!(!cells_fit_gf4(&trace));
        assert_backends_prove_byte_for_byte(&RecurrenceAir, &trace);
    }

    #[test]
    fn proof_size_differs_between_merkle_arities() {
        let log_height = 4;
        let report2 = prove_binary_air(
            &RecurrenceAir,
            recurrence_trace(log_height),
            BinaryProofOptions::default(),
        )
        .expect("a tiny binary AIR proof must verify at arity 2");
        let report4 = prove_binary_air(
            &RecurrenceAir,
            recurrence_trace(log_height),
            BinaryProofOptions {
                merkle_arity: 4,
                ..BinaryProofOptions::default()
            },
        )
        .expect("a tiny binary AIR proof must verify at arity 4");
        assert_ne!(report2.proof_bytes, report4.proof_bytes);
    }

    #[test]
    fn proof_size_differs_between_leaf_geometries() {
        let log_height = 6;
        let report = |leaf_elements| {
            prove_binary_air(
                &RecurrenceAir,
                recurrence_trace(log_height),
                BinaryProofOptions {
                    folding: 2,
                    leaf_elements,
                    ..BinaryProofOptions::default()
                },
            )
            .expect("a tiny binary AIR proof must verify at every leaf size")
        };
        // The fold coset is four symbols, so `None` and an explicit four agree exactly.
        let coset = report(None);
        assert_eq!(coset.leaf_elements, 4);
        assert_eq!(report(Some(4)).proof_bytes, coset.proof_bytes);

        // A wider leaf reshapes both the tree and the opening, so the proof encoding moves.
        // It does not move the statement's security.
        let wide = report(Some(16));
        assert_eq!(wide.leaf_elements, 16);
        assert_ne!(wide.proof_bytes, coset.proof_bytes);
        assert_eq!(wide.security_bits, coset.security_bits);
    }

    #[test]
    fn a_blake3_commitment_proves_the_same_statement_with_different_bytes() {
        let log_height = 4;
        let report = |hash| {
            prove_binary_air(
                &RecurrenceAir,
                recurrence_trace(log_height),
                BinaryProofOptions {
                    hash,
                    ..BinaryProofOptions::default()
                },
            )
            .expect("a tiny binary AIR proof must verify under either hash")
        };
        let keccak = report(HashFamily::Keccak256);
        let blake3 = report(HashFamily::Blake3);

        assert_eq!(keccak.hash, HashFamily::Keccak256);
        assert_eq!(blake3.hash, HashFamily::Blake3);
        // Both digests are 32 bytes, so the composed security is the same bound.
        assert_eq!(blake3.security_bits, keccak.security_bits);
        // A different transcript draws different challenges, hence a different proof.
        assert_ne!(blake3.proof_bytes, keccak.proof_bytes);
    }

    #[test]
    fn a_boolean_trace_proves_under_either_hash() {
        // Three columns over this many rows stack into fourteen variables, seven of which one
        // committed element absorbs, so the base message is 128 symbols: wide enough for the
        // commitment to pack the requested leaf whole.
        let log_height = 12;
        for hash in [HashFamily::Keccak256, HashFamily::Blake3] {
            let report = prove_boolean_air(
                &XorAir,
                Table::new(xor_trace(log_height).transpose()),
                BinaryProofOptions {
                    hash,
                    leaf_elements: Some(64),
                    ..BinaryProofOptions::default()
                },
            )
            .expect("a Boolean trace must prove and verify under either hash");
            assert_eq!(report.hash, hash);
            assert_eq!(report.leaf_elements, 64);
            assert!(report.security_bits >= 100.0);
        }
    }

    #[test]
    fn a_leaf_wider_than_the_base_message_reports_the_cap() {
        // Three columns over 256 rows stack into ten variables, seven of which one committed
        // element absorbs, leaving a base message of eight symbols.
        let log_height = 8;
        let report = |leaf_elements| {
            prove_boolean_air(
                &XorAir,
                Table::new(xor_trace(log_height).transpose()),
                BinaryProofOptions {
                    leaf_elements,
                    ..BinaryProofOptions::default()
                },
            )
            .expect("a Boolean trace must prove and verify at every leaf size")
        };

        // The commitment packs the whole message into one leaf and cannot pack more, so a
        // wider request and the message itself describe the same tree.
        let capped = report(Some(1 << 20));
        let exact = report(Some(8));
        assert_eq!(capped.leaf_elements, 8);
        assert_eq!(capped.requested_leaf_elements, Some(1 << 20));
        assert_eq!(capped.leaf_elements, exact.leaf_elements);
        assert_eq!(capped.proof_bytes, exact.proof_bytes);
        assert!(
            capped
                .to_string()
                .contains("Merkle leaf: 8 field elements (128 bytes; requested 1048576, capped")
        );
    }

    #[test]
    fn rejects_a_leaf_size_that_is_not_a_power_of_two() {
        let result = prove_binary_air(
            &RecurrenceAir,
            recurrence_trace(4),
            BinaryProofOptions {
                leaf_elements: Some(48),
                ..BinaryProofOptions::default()
            },
        );
        let error = result.expect_err("a leaf size off a power of two has no grouping");
        assert_eq!(
            error.to_string(),
            "unsupported leaf size 48; expected a power of two"
        );
        assert!(error.source().is_none());
    }

    #[test]
    fn rejects_an_unsupported_merkle_arity() {
        let log_height = 4;
        let result = prove_binary_air(
            &RecurrenceAir,
            recurrence_trace(log_height),
            BinaryProofOptions {
                merkle_arity: 3,
                ..BinaryProofOptions::default()
            },
        );
        assert!(matches!(
            &result,
            Err(BinaryProofError::UnsupportedMerkleArity(3))
        ));
        let error = result.unwrap_err();
        assert_eq!(
            error.to_string(),
            "unsupported Merkle arity 3; expected 2 or 4"
        );
        assert!(error.source().is_none());
    }

    /// Three bit columns, the third the XOR of the first two.
    ///
    /// Every constraint reads the current row only.
    struct XorAir;

    impl<F> BaseAir<F> for XorAir {
        fn width(&self) -> usize {
            3
        }

        fn main_next_row_columns(&self) -> Vec<usize> {
            Vec::new()
        }
    }

    impl<AB: AirBuilder> Air<AB> for XorAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.current_slice();
            builder.assert_bools([local[0], local[1]]);
            // Addition is XOR in characteristic 2.
            builder.assert_eq(local[2], local[0] + local[1]);
        }
    }

    /// Every row's inputs are the two low bits of its index.
    fn xor_trace(log_height: usize) -> RowMajorMatrix<F> {
        let values = (0..1usize << log_height)
            .flat_map(|row| {
                let (a, b) = (row & 1 == 1, (row >> 1) & 1 == 1);
                [F::from_bool(a), F::from_bool(b), F::from_bool(a ^ b)]
            })
            .collect();
        RowMajorMatrix::new(values, 3)
    }

    #[test]
    fn proves_and_verifies_a_trace_committed_as_bits() {
        // 256 rows of three columns stack to ten bit variables, three past one element's seven.
        let log_height = 8;
        let report = prove_boolean_air(
            &XorAir,
            Table::new(xor_trace(log_height).transpose()),
            BinaryProofOptions::default(),
        )
        .expect("a Boolean trace must prove and verify");
        assert_eq!(report.rows, 1 << log_height);
        assert_eq!(report.width, 3);
        assert_eq!(report.stacked_variables, log_height + 2);
        assert!(report.security_bits >= 100.0);
    }

    #[test]
    fn proves_and_verifies_keccak_committed_as_bits() {
        // One permutation pads to 32 rows, and every constraint links a row to the next.
        //
        // The Boolean commitment opens both views of all 1625 columns in one reduction.
        let air = KeccakBinaryAir::assuming_boolean_trace();
        let trace = air.generate_random_trace_rows::<F>(1, 0);
        let width = trace.width();
        let report = prove_boolean_air(
            &air,
            Table::new(trace.transpose()),
            BinaryProofOptions::default(),
        )
        .expect("a Keccak-f trace committed as bits must prove and verify");
        assert_eq!(report.rows, 32);
        assert_eq!(report.stacked_variables, 5 + log2_ceil_usize(width));
        assert!(report.security_bits >= 100.0);
    }

    #[test]
    fn proves_and_verifies_a_packed_keccak_trace() {
        let air = KeccakBinaryAir::assuming_boolean_trace();
        let words = air.generate_random_trace_packed::<Gf2>(1);
        let table = Table::<F>::from_packed_bits(words, 5);
        let report = prove_boolean_air(&air, table, BinaryProofOptions::default())
            .expect("a packed Keccak-f trace must prove and verify");
        assert_eq!(report.rows, 32);
        assert_eq!(report.width, NUM_KECCAK_BINARY_COLS);
    }

    #[test]
    fn proves_and_verifies_a_packed_blake3_trace() {
        let air = Blake3BinaryAir::assuming_boolean_trace();
        let words = air.generate_random_trace_packed::<Gf2>(4);
        let table = Table::<F>::from_packed_bits(words, 2);
        let report = prove_boolean_air(&air, table, BinaryProofOptions::default())
            .expect("a packed Blake3 trace must prove and verify");
        assert_eq!(report.rows, 4);
        assert_eq!(report.width, NUM_BLAKE3_BINARY_COLS);
    }

    struct NonBooleanAir;

    impl<F> BaseAir<F> for NonBooleanAir {
        fn width(&self) -> usize {
            1
        }

        fn main_next_row_columns(&self) -> Vec<usize> {
            Vec::new()
        }
    }

    impl<AB: AirBuilder> Air<AB> for NonBooleanAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.current_slice();
            builder.assert_eq(local[0], local[0]);
        }
    }

    #[test]
    #[should_panic(expected = "cannot prove an AIR that assumes a Boolean trace")]
    fn field_commitment_refuses_an_air_that_assumes_a_boolean_trace() {
        let air = KeccakBinaryAir::assuming_boolean_trace();
        let trace = air.generate_random_trace_rows::<F>(1, 0);
        let _ = prove_binary_air(&air, trace, BinaryProofOptions::default());
    }

    #[test]
    fn boolean_commitment_refuses_nonboolean_dense_cells() {
        let trace = Table::new(RowMajorMatrix::new(vec![F::from_repr(2); 256], 256));
        let result = prove_boolean_air(&NonBooleanAir, trace, BinaryProofOptions::default());
        assert!(matches!(result, Err(BinaryProofError::BooleanProve(_))));
    }

    #[test]
    fn configuration_error_reports_context_and_inner_message_once() {
        let error = BinaryProofError::from(BinaryPcsConfigError::InvalidFoldingFactor {
            requested: 0,
            num_variables: 8,
        });
        assert_eq!(
            error.to_string(),
            "binary PCS configuration failed: folding factor log 0 must be in 1..=8"
        );
        assert!(error.source().is_none());
    }
}
