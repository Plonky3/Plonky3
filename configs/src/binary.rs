use alloc::vec::Vec;

use p3_binary_field::BinaryChallenger;
pub use p3_binary_field::BinaryField128 as Val;
use p3_binary_pcs::{BinaryPcs, BinaryPcsProverData};
pub use p3_binary_pcs::{BinaryPcsConfig, BinaryPcsConfigError, BinaryPcsParams};
use p3_challenger::HashChallenger;
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multi_stark::config::MultiStarkConfig;
use p3_sumcheck::layout::{Layout, SuffixProver, Table, Witness};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};

/// Binary-field challenges live in the base field GF(2^128).
pub type Challenge = Val;
/// Keccak-256 Merkle tree, with a single root (cap height zero).
pub type Mmcs = MerkleTreeMmcs<
    Val,
    u8,
    SerializingHasher<Keccak256Hash>,
    CompressionFunctionFromHasher<Keccak256Hash, 2, 32>,
    2,
    32,
>;
/// Binary-field transcript over Keccak-256.
pub type Challenger = BinaryChallenger<Val, HashChallenger<u8, Keccak256Hash, 32>>;
/// Non-hiding additive-domain multilinear commitment scheme.
pub type Pcs = BinaryPcs<Mmcs>;

/// Binary PCS adapter accepted by [`crate::multi_stark`].
///
/// Owns witness construction and committed-table access. Polynomial arities and
/// PCS security parameters remain explicit in [`BinaryPcsConfig`]. The trace
/// tables must have at least one variable (two rows). The PCS arity is the
/// arity of the *stacked* polynomial, including all columns and tables, not
/// just the trace height. No automatic arity inference or padding is added.
pub struct Config {
    pcs: Pcs,
    preprocessed_pcs: Option<Pcs>,
}

impl Config {
    /// Use an already validated binary PCS configuration for the main trace.
    ///
    /// This selects Keccak-256 commitments. The PCS target excludes STARK-level
    /// errors; targets above 128 bits also exceed the digest's collision bound.
    #[must_use]
    pub fn new(params: BinaryPcsConfig) -> Self {
        Self {
            pcs: pcs(params),
            preprocessed_pcs: None,
        }
    }

    /// Supply the PCS configuration for the stacked preprocessed columns.
    ///
    /// Its arity may differ from the main trace. Call this before setup when
    /// the AIR has preprocessed columns; setup without it panics.
    #[must_use]
    pub fn with_preprocessed(mut self, params: BinaryPcsConfig) -> Self {
        self.preprocessed_pcs = Some(pcs(params));
        self
    }
}

fn pcs(params: BinaryPcsConfig) -> Pcs {
    let mmcs = Mmcs::new(
        SerializingHasher::new(Keccak256Hash),
        CompressionFunctionFromHasher::new(Keccak256Hash),
        0,
    );
    BinaryPcs::new(params, mmcs)
}

/// Start a fresh transcript with the application's protocol/version domain.
///
/// Use identical domain bytes for setup, proving and verification, and a new
/// challenger for each invocation. Never reuse the prover's mutated state for
/// verification. Additional application context must be observed identically
/// by both parties.
#[must_use]
pub fn challenger(domain: &[u8]) -> Challenger {
    Challenger::from_hasher(domain.to_vec(), Keccak256Hash)
}

impl MultiStarkConfig for Config {
    type Val = Val;
    type Challenge = Challenge;
    type Challenger = Challenger;
    type Pcs = Pcs;

    fn pcs(&self) -> &Pcs {
        &self.pcs
    }

    fn preprocessed_pcs(&self) -> &Pcs {
        self.preprocessed_pcs.as_ref().expect(
            "binary Config requires with_preprocessed before setup with preprocessed columns",
        )
    }

    fn min_num_variables(&self) -> usize {
        1
    }

    fn build_witness(&self, tables: Vec<Table<Val>>) -> Witness<Val> {
        SuffixProver::<Val, Val>::new_witness(tables, 0)
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a BinaryPcsProverData<Mmcs>,
        table_index: usize,
    ) -> &'a Table<Val> {
        prover_data.table(table_index)
    }
}
