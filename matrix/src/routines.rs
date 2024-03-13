use alloc::vec;
use alloc::vec::Vec;
use itertools::izip;
use p3_field::{ExtensionField, Field};
use tracing::instrument;

use crate::MatrixRows;

/// Tranposed matrix-vector product: Mᵀv
/// Can handle a vector of extensions of the matrix field, the other way around
/// would require a different method.
/// TODO: make faster (currently ~100ms of m31_keccak)
#[instrument(skip_all, fields(dims = %m.dimensions()))]
pub fn columnwise_dot_product<F, EF, M>(m: M, v: impl Iterator<Item = EF>) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
    M: MatrixRows<F>,
{
    let mut accs = vec![EF::zero(); m.width()];
    for (row, vx) in izip!(m.rows(), v) {
        for (acc, mx) in izip!(&mut accs, row) {
            *acc += vx * mx;
        }
    }
    accs
}
