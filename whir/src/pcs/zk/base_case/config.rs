//! Protocol shape shared by the base-case prover and verifier.

use alloc::vec::Vec;

use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

use crate::pcs::zk::committer::FoldedRsCode;
use crate::pcs::zk::mask::MaskGroupShape;

/// Shared base-case protocol shape.
#[derive(Debug, Clone)]
pub struct BaseCaseZkConfig<F> {
    /// Folded source code: the code of the virtual oracle being checked.
    pub code: FoldedRsCode<F>,
    /// Carried mask oracle groups, in chronological commit order.
    pub mask_groups: Vec<MaskGroupShape>,
    /// Spot checks against the source oracle.
    pub num_queries: usize,
    /// Spot checks per carried mask group.
    pub mask_queries: usize,
    /// PoW difficulty before the spot checks.
    pub pow_bits: usize,
}

/// Prover-side view of one carried mask group.
pub struct MaskGroupWitness<'a, F, EF, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Member mask messages, one per column.
    pub messages: &'a [Vec<EF>],
    /// Encoding randomness used when each member was committed.
    pub randomness: &'a [Vec<EF>],
    /// Current dense covectors, one per member.
    pub covectors: &'a [Vec<EF>],
    /// Merkle prover data behind the group commitment.
    pub data: &'a MaskProverData<F, EF, MT>,
}

/// Prover data type shared by every mask group in the pipeline.
pub type MaskProverData<F, EF, MT> =
    <ExtensionMmcs<F, EF, MT> as Mmcs<EF>>::ProverData<RowMajorMatrix<EF>>;
