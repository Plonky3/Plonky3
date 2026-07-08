//! Shared circulant-Karatsuba dispatch for the dense Poseidon1 full-round MDS matrix.
//!
//! Used by both constraint evaluation ([`crate::air`]) and trace generation
//! ([`crate::generation`]), so the two share the same width-16/24 fast path.

use p3_field::{Algebra, PrimeCharacteristicRing};
use p3_mds::karatsuba_convolution::{mds_circulant_karatsuba_16, mds_circulant_karatsuba_24};
use p3_poseidon1::external::mds_multiply;

/// Multiply `state` by the dense MDS matrix.
///
/// Dispatches to a circulant Karatsuba convolution for supported widths (16, 24),
/// falling back to the dense `O(WIDTH^2)` multiply otherwise.
#[inline]
pub(crate) fn mds_dispatch<F: PrimeCharacteristicRing, A: Algebra<F>, const WIDTH: usize>(
    state: &mut [A; WIDTH],
    circ_col: &[F; WIDTH],
    dense_mds: &[[F; WIDTH]; WIDTH],
) {
    match WIDTH {
        16 => {
            // WIDTH == 16 in this arm, so both fixed-size views always succeed.
            let state_16: &mut [A; 16] = state.as_mut_slice().try_into().expect("WIDTH == 16");
            let col_16: &[F; 16] = circ_col.as_slice().try_into().expect("WIDTH == 16");
            mds_circulant_karatsuba_16(state_16, col_16);
        }
        24 => {
            // WIDTH == 24 in this arm, so both fixed-size views always succeed.
            let state_24: &mut [A; 24] = state.as_mut_slice().try_into().expect("WIDTH == 24");
            let col_24: &[F; 24] = circ_col.as_slice().try_into().expect("WIDTH == 24");
            mds_circulant_karatsuba_24(state_24, col_24);
        }
        // No circulant fast path for this width: fall back to the dense O(WIDTH^2) multiply.
        _ => mds_multiply(state, dense_mds),
    }
}
