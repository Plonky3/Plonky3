//! WHIR-facing opening layer for a WARP accumulator.
//!
//! This module is the WHIR-facing boundary for proving the final WARP
//! accumulator checks with WHIR-compatible openings. It provides a
//! precommitted opening proof for the RS/MLE accumulator opening
//!
//! ```text
//!     f_hat(alpha) = mu
//! ```
//!
//! and a sumcheck proof for the Boolean PESAT decider claim
//! `Pb(beta, C^{-1}(f)) = eta`, using Plonky3's
//! [`MultilinearPcs`](p3_commit::MultilinearPcs) abstraction. The important
//! soundness condition is enforced explicitly throughout: PCS openings are
//! checked against the accumulator's existing commitment `rt`. A WHIR wrapper
//! that opens a fresh unrelated commitment would be unsound as a WARP
//! finalizer.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{MultilinearOpenedValues, MultilinearPcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_whir::pcs::proof::WhirProof;
use p3_whir::pcs::{WhirDeferredProverData, WhirPcs};
use serde::{Deserialize, Serialize};

use crate::accumulator::{AccumulatorInstance, AccumulatorWitness};
use crate::code::ReedSolomonCode;
use crate::error::{DeciderError, FinalizerError};
use crate::protocol::{
    AccumulatorBatchOpeningBackend, AccumulatorCommitmentBackend,
    ExternalCodewordBatchOpeningProver, ExternalCodewordBatchOpeningVerifier,
    ExternalCodewordOpeningProver, ExternalCodewordOpeningVerifier, ExternalCommitmentObserver,
    ExternalCommittedCodeword,
};
use crate::relation::{BooleanPesat, BundledPesat};
use crate::sumcheck::{SumcheckProof, observe_and_sample, verify_sumcheck};

mod proof;
pub use proof::{WhirAccumulatorOpeningProof, WhirPesatProof, WhirWarpFinalizerProof};

mod precommitted;
pub use precommitted::{
    AccumulatorPointOpeningBackend, PrecommittedAccumulatorPcs,
    PrecommittedAccumulatorPcsProverData,
};

mod domain;

mod extension_limb;
pub use extension_limb::{
    ExtensionLimbPcs, ExtensionLimbPcsError, ExtensionLimbPcsProof, ExtensionLimbPcsProverData,
    WhirLimbAccumulatorBackend, WhirLimbAccumulatorOpeningProof, WhirLimbAccumulatorProverData,
};

mod codeword_backend;
pub use codeword_backend::{WhirCodewordBackend, WhirCommittedCodeword};

mod accumulator_opening;
pub use accumulator_opening::WhirAccumulatorOpeningProtocol;

mod finalizer;
pub use finalizer::{
    WhirBooleanPesatProtocol, WhirBooleanWarpFinalizerProtocol,
    WhirPrecommittedBooleanWarpFinalizerProtocol,
};

fn boolean_index_point<EF: Field>(index: usize, log_len: usize) -> Vec<EF> {
    (0..log_len)
        .map(|i| {
            if (index >> (log_len - 1 - i)) & 1 == 1 {
                EF::ONE
            } else {
                EF::ZERO
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
