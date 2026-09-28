//! An AIR for the Keccak-f permutation over a field of characteristic 2.
//!
//! Every state bit is a column. Field addition is XOR and the product of two bits is their AND,
//! so each round is a low-degree polynomial map on the bits of one row.

mod air;
mod columns;
mod generation;

pub use air::*;
pub use columns::*;
pub use generation::*;

use crate::constants::R;

/// Locate the lane of `A'` that the rho and pi steps rotate into `B[x, y]`.
///
/// ```text
///     B[y, 2x + 3y] = ROT(A'[x, y], R[x][y])
///     B[x, y]       = ROT(A'[x + 3y, x], R[x + 3y][x])        (lane indices mod 5)
/// ```
///
/// # Returns
///
/// `(y', x', rot)` with `x' = (x + 3y) mod 5` and `y' = x`:
/// bit `z` of `B[x, y]` is bit `(z - rot) mod 64` of `A'[x', y']`.
/// The first two entries are in the y-major order of the state columns.
const fn rho_pi_source(x: usize, y: usize) -> (usize, usize, usize) {
    let source_x = (x + 3 * y) % 5;
    let source_y = x;
    (source_y, source_x, R[source_x][source_y] as usize)
}
