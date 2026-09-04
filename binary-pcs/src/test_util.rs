//! Shared fixtures for `p3-binary-pcs` tests: a Merkle tree MMCS over `BinaryField128` with a
//! Keccak byte hasher, the matching binary Fiat-Shamir challenger, and a full commit/open
//! lifecycle for tests that check `verify` against a (possibly mutated) genuine proof.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{BinaryChallenger, BinaryField128};
use p3_challenger::HashChallenger;
use p3_commit::{Mmcs, MultilinearPcs};
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sumcheck::layout::{Layout, SuffixProver, Table};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::params::{BinaryPcsConfig, BinaryPcsParams};
use crate::pcs::BinaryPcs;
use crate::proof::BinaryPcsProof;

type F = BinaryField128;
type MyHash = SerializingHasher<Keccak256Hash>;
type MyCompress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
pub(crate) type MyMmcs = MerkleTreeMmcs<F, u8, MyHash, MyCompress, 2, 32>;
pub(crate) type MyChallenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

pub(crate) const fn mmcs() -> MyMmcs {
    MyMmcs::new(
        MyHash::new(Keccak256Hash),
        MyCompress::new(Keccak256Hash),
        0,
    )
}

pub(crate) const fn challenger() -> MyChallenger {
    MyChallenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// Fixed parameters `run_lifecycle` derives its config from.
const fn params() -> BinaryPcsParams {
    BinaryPcsParams {
        log_inv_rate: 2,
        pow_bits: 4,
        security_level: 40,
    }
}

/// Commits a random single-column table, opens column 0 at a transcript-sampled point, and
/// returns everything a caller needs to replay or mutate the proof: the PCS instance (reusable
/// for a fresh verify call), the commitment, the genuine proof, and the opening protocol that
/// produced it.
///
/// `num_variables` and `folding` are forwarded to [`BinaryPcsConfig::try_new`] verbatim; only
/// `folding = 0` derives a config this crate's commit phase accepts.
#[allow(clippy::type_complexity)]
pub(crate) fn run_lifecycle(
    num_variables: usize,
    folding: usize,
) -> (
    BinaryPcs<MyMmcs, SuffixProver<F, F>>,
    <MyMmcs as Mmcs<F>>::Commitment,
    BinaryPcsProof<MyMmcs>,
    OpeningProtocol,
) {
    let mut rng = SmallRng::seed_from_u64(0xB1DA_u64);
    let table = Table::rand(&mut rng, 1, num_variables);
    let witness = SuffixProver::<F, F>::new_witness(vec![table], folding);

    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(num_variables, 1),
        vec![OpeningBatch::new(vec![0], Vec::new())],
    )]);

    let config = BinaryPcsConfig::try_new(num_variables, folding, params()).unwrap();
    let pcs = BinaryPcs::new(config, mmcs());

    let mut prover_challenger = challenger();
    let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger);
    let proof = pcs.open(prover_data, protocol.clone(), &mut prover_challenger);

    (pcs, commitment, proof, protocol)
}
