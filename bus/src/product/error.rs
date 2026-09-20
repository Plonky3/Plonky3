//! Errors returned by product-tree statement and proof validation.

use thiserror::Error;

/// Invalid dimensions rejected before a transcript is created.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum ProductGkrShapeError {
    /// A batch without trees has no statement.
    #[error("product GKR received {num_trees} trees, minimum is {minimum}")]
    NoTrees {
        /// The rejected tree count.
        num_trees: usize,
        /// The smallest nonempty batch.
        minimum: usize,
    },
    /// A shared-root shape needs two roots to share.
    #[error("shared-root product GKR received {num_trees} trees, minimum is {minimum}")]
    SharedRootNeedsTwoTrees {
        /// The rejected tree count.
        num_trees: usize,
        /// The smallest batch containing two roots.
        minimum: usize,
    },
    /// Per-layer child messages would overflow their machine-word length.
    #[error("product GKR received {num_trees} trees, maximum is {maximum}")]
    TreeCountOverflow {
        /// The rejected tree count.
        num_trees: usize,
        /// The largest count whose four-child message fits in memory.
        maximum: usize,
    },
    /// The logical tree size cannot be represented by the target architecture.
    #[error("product GKR log height {log_height} exceeds maximum {maximum}")]
    HeightOverflow {
        /// The rejected base-two tree height.
        log_height: usize,
        /// The largest height whose capacity fits in one machine word.
        maximum: usize,
    },
}

/// Invalid inputs or proofs rejected by the product reduction.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ProductGkrError {
    /// Degree-five interpolation needs six distinct challenge-field elements.
    #[error("product GKR requires six distinct challenge-field interpolation nodes")]
    ChallengeFieldTooSmall,
    /// The input batch disagrees with the verifier-derived tree count.
    #[error("product GKR received {actual} trees, expected {expected}")]
    TreeCountMismatch {
        /// Tree count fixed by the statement.
        expected: usize,
        /// Tree count supplied by the caller.
        actual: usize,
    },
    /// An explicit prefix is longer than its logical tree.
    #[error("product GKR tree {tree} has {actual} leaves, logical capacity is {capacity}")]
    InputTooLong {
        /// Position of the malformed tree.
        tree: usize,
        /// Logical leaf capacity.
        capacity: usize,
        /// Supplied prefix length.
        actual: usize,
    },
    /// Two roots required to be structurally shared are unequal.
    #[error("the first two product roots are not equal")]
    SharedRootMismatch,
    /// A proof carries the wrong number of encoded roots.
    #[error("product GKR proof has {actual} root messages, expected {expected}")]
    RootCountMismatch {
        /// Root count fixed by the statement.
        expected: usize,
        /// Root count carried by the proof.
        actual: usize,
    },
    /// A proof carries the wrong number of reduction layers.
    #[error("product GKR proof has {actual} layers, expected {expected}")]
    LayerCountMismatch {
        /// Layer count fixed by the statement.
        expected: usize,
        /// Layer count carried by the proof.
        actual: usize,
    },
    /// One layer has the wrong arity or number of rounds.
    #[error("product GKR layer {layer} has a malformed shape")]
    MalformedLayer {
        /// Position of the malformed layer.
        layer: usize,
    },
    /// One layer's sumcheck does not close on its child claims.
    #[error("product GKR layer {layer} failed its consistency check")]
    LayerConsistency {
        /// Position of the inconsistent layer.
        layer: usize,
    },
}
