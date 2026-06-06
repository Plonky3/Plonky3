//! Prover-side oracle state carried between phases.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_commit::Mmcs;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_matrix::extension::FlatMatrixView;
use p3_multilinear_util::poly::Poly;

/// Prover-side handoff between the commit and open phases.
pub struct HidingWhirProverData<F, EF, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Committed multilinear evaluations.
    pub message: Poly<F>,
    /// Limb-major ZK encoding randomness of the initial oracle.
    pub randomness: Vec<F>,
    /// Merkle prover data behind the initial commitment.
    pub merkle: MT::ProverData<DenseMatrix<F>>,
    /// Marker tying the data to its extension field.
    pub(crate) _marker: PhantomData<EF>,
}

/// Merkle prover data of the active committed oracle.
pub(super) enum ZkRoundData<F, EF, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Base-field initial oracle.
    Base(MT::ProverData<DenseMatrix<F>>),
    /// Extension-field folded oracle.
    Ext(<MT as Mmcs<F>>::ProverData<FlatMatrixView<F, EF, DenseMatrix<EF>>>),
}
