//! Keys, and the Merkle tree a secret key keeps over its epochs.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::PrimeCharacteristicRing;

use crate::hash::{DIGEST_LEN, Hasher, PARAMETER_LEN};
use crate::{Digest, F, Params};

/// A public key: the Merkle root over every epoch, and the hash parameter all its hashes take.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PublicKey {
    /// The root of the tree whose leaves are the epochs' chain ends.
    pub root: Digest,
    /// The parameter that separates this key's hashes from every other key's.
    pub parameter: [F; PARAMETER_LEN],
}

/// A secret key: the seed its chains derive from, and the tree over its active epochs.
#[derive(Clone, Debug)]
pub struct SecretKey {
    /// The parameters the key was generated for.
    pub(crate) params: Params,
    /// The public key's hash parameter.
    pub(crate) parameter: [F; PARAMETER_LEN],
    /// The seed every chain start and randomness derives from.
    pub(crate) seed: Digest,
    /// The Merkle tree over the epochs.
    pub(crate) tree: Tree,
}

impl SecretKey {
    /// The parameters the key was generated for.
    pub const fn params(&self) -> &Params {
        &self.params
    }

    /// The number of leading epochs the key can sign.
    pub fn active_epochs(&self) -> usize {
        self.tree.levels[0].len()
    }
}

/// A Merkle tree whose leaves past the active ones are the zero digest.
///
/// Only the nodes over active leaves are stored, so a long lifetime costs its active part only.
#[derive(Clone, Debug)]
pub(crate) struct Tree {
    /// The stored nodes of each level, leaves first.
    levels: Vec<Vec<Digest>>,
    /// The root of an inactive subtree at each level.
    empty: Vec<Digest>,
}

impl Tree {
    /// The tree of `height` over `leaves`, padded with inactive leaves.
    pub(crate) fn new(hasher: &Hasher, leaves: Vec<Digest>, height: usize) -> Self {
        assert!(
            !leaves.is_empty() && leaves.len() <= 1 << height,
            "between one leaf and a full tree"
        );
        let mut empty = vec![[F::ZERO; DIGEST_LEN]];
        let mut levels = vec![leaves];
        for level in 0..height {
            let below = &levels[level];
            let parents = (0..below.len().div_ceil(2))
                .map(|i| {
                    let child = |j: usize| below.get(j).copied().unwrap_or(empty[level]);
                    hasher.node(child(2 * i), child(2 * i + 1))
                })
                .collect();
            levels.push(parents);
            empty.push(hasher.node(empty[level], empty[level]));
        }
        Self { levels, empty }
    }

    /// The root.
    pub(crate) fn root(&self) -> Digest {
        self.levels[self.levels.len() - 1][0]
    }

    /// The siblings from leaf `index` up to the root.
    pub(crate) fn path(&self, index: usize) -> Vec<Digest> {
        let height = self.levels.len() - 1;
        (0..height)
            .map(|level| {
                let sibling = (index >> level) ^ 1;
                self.levels[level]
                    .get(sibling)
                    .copied()
                    .unwrap_or(self.empty[level])
            })
            .collect()
    }
}
