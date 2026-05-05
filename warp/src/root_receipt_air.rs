//! Stark-backed WARP root receipt scaffold.
//!
//! This module wires WARP root receipts into `stark-backend`/SWIRL by exposing
//! the canonical root-claim digest as public values of a tiny AIR. It is not
//! the full WARP verifier arithmetization; it is the outer proof plumbing and
//! public-input contract that the later OpenVM-recursion verifier AIRs should
//! target.

use std::sync::Arc;
use std::vec;
use std::vec::Vec;

use openvm_stark_backend::{
    AirRef, PartitionedBaseAir, StarkProtocolConfig,
    prover::{AirProvingContext, ColMajorMatrix, CpuColMajorBackend, ProvingContext},
};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

/// A minimal AIR binding a native WARP root receipt digest into public values.
#[derive(Clone, Debug)]
pub struct WarpRootReceiptAir {
    digest_len: usize,
}

impl WarpRootReceiptAir {
    /// Create an AIR for a digest with `digest_len` base-field limbs.
    pub fn new(digest_len: usize) -> Self {
        assert!(digest_len > 0, "root receipt digest must be non-empty");
        Self { digest_len }
    }

    /// Number of base-field public values exposed by this AIR.
    pub fn digest_len(&self) -> usize {
        self.digest_len
    }
}

impl<F> PartitionedBaseAir<F> for WarpRootReceiptAir {}

impl<F> BaseAir<F> for WarpRootReceiptAir {
    fn width(&self) -> usize {
        self.digest_len + 1
    }

    fn num_public_values(&self) -> usize {
        self.digest_len
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }
}

impl<AB> Air<AB> for WarpRootReceiptAir
where
    AB: AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice().to_vec();
        let public_values = builder.public_values().to_vec();

        builder.when_first_row().assert_one(row[0]);
        for i in 0..self.digest_len {
            builder
                .when_first_row()
                .assert_eq(row[i + 1], public_values[i]);
        }
    }
}

/// Build the AIR object used for a root receipt digest of `digest_len` limbs.
pub fn root_receipt_air<SC>(digest_len: usize) -> AirRef<SC>
where
    SC: StarkProtocolConfig,
{
    Arc::new(WarpRootReceiptAir::new(digest_len)) as AirRef<SC>
}

/// Build a minimal proving context for a root receipt digest.
///
/// The digest is constrained on the first row. We keep a second unconstrained
/// row so the current `stark-backend` CPU prover sees a non-degenerate trace.
pub fn root_receipt_air_context<SC>(
    claim_digest: &[SC::F],
) -> AirProvingContext<CpuColMajorBackend<SC>>
where
    SC: StarkProtocolConfig,
    SC::F: Field,
{
    assert!(
        !claim_digest.is_empty(),
        "root receipt digest must be non-empty"
    );
    let width = claim_digest.len() + 1;
    let mut row = Vec::with_capacity(2 * width);
    row.push(SC::F::ONE);
    row.extend_from_slice(claim_digest);
    row.resize(2 * width, SC::F::ZERO);

    let trace = RowMajorMatrix::new(row, width);
    AirProvingContext::simple(
        ColMajorMatrix::from_row_major(&trace),
        claim_digest.to_vec(),
    )
}

/// Build a `stark-backend` proving context for a single root receipt AIR.
pub fn root_receipt_proving_context<SC>(
    air_id: usize,
    claim_digest: &[SC::F],
) -> ProvingContext<CpuColMajorBackend<SC>>
where
    SC: StarkProtocolConfig,
    SC::F: Field,
{
    ProvingContext::new(vec![(air_id, root_receipt_air_context::<SC>(claim_digest))])
}
