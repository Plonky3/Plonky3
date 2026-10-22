//! Keys, and the Merkle tree a secret key keeps over its epochs.

use alloc::vec;
use alloc::vec::Vec;

use crate::hash::{PARAMETER_LEN, TweakableHash};
use crate::{Digest, F, Params};

/// A public key: the Merkle root over every epoch, and the hash parameter all its hashes take.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PublicKey<D = Digest, P = [F; PARAMETER_LEN]> {
    /// The root of the tree whose leaves are the epochs' chain ends.
    pub root: D,
    /// The parameter that separates this key's hashes from every other key's.
    pub parameter: P,
}

/// A secret key: the seed its chains derive from, and the tree over its active epochs.
#[derive(Clone, Debug)]
pub struct SecretKey<D = Digest, P = [F; PARAMETER_LEN]> {
    /// The parameters the key was generated for.
    pub(crate) params: Params,
    /// The public key's hash parameter.
    pub(crate) parameter: P,
    /// The seed every chain start and randomness derives from.
    pub(crate) seed: D,
    /// The Merkle tree over the epochs.
    pub(crate) tree: Tree<D>,
}

impl<D, P> SecretKey<D, P> {
    /// The parameters the key was generated for.
    pub const fn params(&self) -> &Params {
        &self.params
    }

    /// The number of leading epochs the key can sign.
    pub fn active_epochs(&self) -> usize {
        self.tree.levels[0].len()
    }
}

/// A Merkle tree whose leaves past the active ones are inactive.
///
/// Only the nodes over active leaves are stored, so a long lifetime costs its active part only.
#[derive(Clone, Debug)]
pub(crate) struct Tree<D> {
    /// The stored nodes of each level, leaves first.
    levels: Vec<Vec<D>>,
    /// The root of an inactive subtree at each level.
    empty: Vec<D>,
}

impl<D: Copy> Tree<D> {
    /// The tree of `height` over `leaves`, padded with inactive leaves.
    pub(crate) fn new<H: TweakableHash<Digest = D>>(
        hasher: &H,
        parameter: &H::Parameter,
        leaves: Vec<D>,
        height: usize,
    ) -> Self {
        assert!(
            !leaves.is_empty() && leaves.len() <= 1 << height,
            "between one leaf and a full tree"
        );
        let empty: Vec<D> = (0..=height)
            .map(|level| hasher.inactive_root(parameter, level))
            .collect();
        let mut levels = vec![leaves];
        for level in 0..height {
            let below = &levels[level];
            let parents = (0..below.len().div_ceil(2))
                .map(|i| {
                    let child = |j: usize| below.get(j).copied().unwrap_or(empty[level]);
                    hasher.node(parameter, level + 1, i, child(2 * i), child(2 * i + 1))
                })
                .collect();
            levels.push(parents);
        }
        Self { levels, empty }
    }

    /// The root.
    pub(crate) fn root(&self) -> D {
        self.levels[self.levels.len() - 1][0]
    }

    /// The siblings from leaf `index` up to the root.
    pub(crate) fn path(&self, index: usize) -> Vec<D> {
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
