//! Shared fixtures for this crate's tests.
//!
//! A Merkle tree scheme at each of the two tower levels the tests commit over.
//! The matching binary Fiat-Shamir challenger for each.
//!
//! A full commit and open lifecycle, for tests that mutate a genuine proof.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{BinaryChallenger, BinaryField64, BinaryField128};
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
pub(crate) type NarrowMmcs = MerkleTreeMmcs<BinaryField64, u8, MyHash, MyCompress, 2, 32>;
pub(crate) type MyChallenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;
pub(crate) type NarrowChallenger =
    BinaryChallenger<BinaryField64, HashChallenger<u8, Keccak256Hash, 32>>;

pub(crate) const fn mmcs() -> MyMmcs {
    MyMmcs::new(
        MyHash::new(Keccak256Hash),
        MyCompress::new(Keccak256Hash),
        0,
    )
}

/// The same scheme over the narrower tower level a small-alphabet commitment uses.
pub(crate) const fn narrow_mmcs() -> NarrowMmcs {
    NarrowMmcs::new(
        MyHash::new(Keccak256Hash),
        MyCompress::new(Keccak256Hash),
        0,
    )
}

pub(crate) const fn challenger() -> MyChallenger {
    MyChallenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// The sponge a narrow-alphabet run speaks, whose grinding witness is a narrow element.
pub(crate) const fn narrow_challenger() -> NarrowChallenger {
    NarrowChallenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// Fixed parameters the lifecycle fixture derives its config from.
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
#[allow(clippy::type_complexity)]
pub(crate) fn run_lifecycle(
    num_variables: usize,
) -> (
    BinaryPcs<F, F, MyMmcs, MyMmcs>,
    <MyMmcs as Mmcs<F>>::Commitment,
    BinaryPcsProof<F, F, MyMmcs, MyMmcs>,
    OpeningProtocol,
) {
    let mut rng = SmallRng::seed_from_u64(0xB1DA_u64);
    let table = Table::rand(&mut rng, 1, num_variables);
    let witness = SuffixProver::<F, F>::new_witness(vec![table], 0);

    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(num_variables, 1),
        vec![OpeningBatch::new(vec![0], Vec::new())],
    )]);

    let config = BinaryPcsConfig::try_new::<F, F>(num_variables, params()).unwrap();
    let pcs = BinaryPcs::new(config, mmcs(), mmcs());

    let mut prover_challenger = challenger();
    let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger).unwrap();
    let proof = pcs
        .open(prover_data, protocol.clone(), &mut prover_challenger)
        .unwrap();

    (pcs, commitment, proof, protocol)
}

#[cfg(test)]
mod conformance {
    use alloc::vec;

    use p3_challenger::CanSample;
    use p3_commit::MultilinearPcs;
    use p3_sumcheck::layout::{Layout, SuffixProver, Table};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::{F, challenger, mmcs, params};
    use crate::{BinaryPcs, BinaryPcsConfig};

    #[test]
    fn the_commit_phase_binds_exactly_what_the_binding_method_binds() {
        // Invariant: a verifier never commits, so it replays the prover's binding
        // by calling the scheme's binding method.
        //
        // The two are interchangeable only while they leave the sponge in one state.
        //
        //     prover  : commit(witness, a)          -> a
        //     verifier: observe_commitment(root, b) -> b
        //     a and b must sample alike
        //
        // This scheme binds through the layout's typed commitment phase, like the others.
        //
        // The property is therefore checked the same way.
        const NUM_VARIABLES: usize = 6;

        let mut rng = SmallRng::seed_from_u64(0xB1DA);
        let table = Table::rand(&mut rng, 1, NUM_VARIABLES);
        let witness = SuffixProver::<F, F>::new_witness(vec![table], 0);

        let config = BinaryPcsConfig::try_new::<F, F>(NUM_VARIABLES, params()).unwrap();
        let pcs = BinaryPcs::new(config, mmcs(), mmcs());

        let mut committed = challenger();
        let (commitment, _) = pcs.commit(witness, &mut committed).unwrap();

        let mut replayed = challenger();
        pcs.observe_commitment(&commitment, &mut replayed);

        assert_eq!(
            CanSample::<F>::sample(&mut committed),
            CanSample::<F>::sample(&mut replayed),
        );
    }
}
