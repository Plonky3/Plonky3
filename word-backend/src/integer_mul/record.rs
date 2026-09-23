//! The transcript record of one multiplication reduction.

use alloc::vec::Vec;

use p3_sumcheck::generic_degree::GenericDegreeProof;
use serde::{Deserialize, Serialize};

/// Transcript record of one full-width unsigned multiplication reduction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntegerMulProof<F, EF> {
    /// Shared value of both exponent lifts at the sampled row point.
    pub(crate) root: EF,
    /// Reduction of the factor tree, `g^(a * b)`, to its two factor columns.
    pub(crate) factor: TreeProof<F, EF>,
    /// Reduction of the result tree, `g^(lo + 2^w * hi)`, to its two limb columns.
    pub(crate) result: TreeProof<F, EF>,
}

/// Transcript record of one product tree, root layer first.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreeProof<F, EF> {
    /// One reduction per layer, from the root towards the leaves.
    pub(crate) layers: Vec<LayerProof<F, EF>>,
    /// Sumcheck reducing the leaf claim to the committed operand columns.
    pub(crate) leaf: GenericDegreeProof<F, EF>,
    /// Operand evaluations the leaf sumcheck ends on.
    pub(crate) values: [EF; 2],
}

/// Transcript record of one product layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerProof<F, EF> {
    /// Sumcheck reducing one layer claim to a claim on the two halves below it.
    pub(crate) sumcheck: GenericDegreeProof<F, EF>,
    /// Evaluations of the lower layer with its new last coordinate at zero, then at one.
    pub(crate) halves: [EF; 2],
}
