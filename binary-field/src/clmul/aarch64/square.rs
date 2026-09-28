//! Repeated squaring in `GF(2^64)` as one bit-matrix product, with masks instead of lookups.
//!
//! Squaring is `F_2`-linear, so squaring `K` times is a fixed `64 x 64` bit matrix `M`.
//!
//! Its column `c` is the image of `x^c`, so the power is a sum of columns:
//!
//! ```text
//!     x^(2^K)  =  sum_c  bit_c(x) * M_c
//! ```
//!
//! One register holds two columns side by side.
//!
//! `CMTST` turns two bits of `x` into two all-ones or all-zeros masks, one per quadword.
//!
//! So each pair of columns costs one mask, one `AND` and one exclusive or:
//!
//! ```text
//!     mask_j  =  [ bit_2j(x) ? ~0 : 0 ,  bit_(2j+1)(x) ? ~0 : 0 ]
//!     acc    ^=  mask_j & [ M_2j , M_(2j+1) ]
//! ```
//!
//! The cost is the same for every `K`.
//!
//! No address or branch depends on `x`, so the product stays constant time.

use core::arch::aarch64::{
    uint64x2_t, vandq_u64, vdupq_n_u64, veorq_u64, vgetq_lane_u64, vld1q_u64, vtstq_u64,
};

use crate::clmul::gf64::repeated_square;
use crate::clmul::wide::Lanes64;

/// Pairs of columns in the matrix.
const PAIRS: usize = 32;

/// Independent partial sums, so the exclusive ors do not form one dependency chain.
const CHAINS: usize = 4;

/// The single-bit masks each pair of columns tests: `[x^(2j), x^(2j + 1)]`.
const BITS: [[u64; 2]; PAIRS] = {
    let mut bits = [[0; 2]; PAIRS];
    let mut j = 0;
    while j < PAIRS {
        bits[j] = [1 << (2 * j), 1 << (2 * j + 1)];
        j += 1;
    }
    bits
};

/// The columns of `x -> x^(2^K)`, two per register.
struct Columns<const K: usize>([[u64; 2]; PAIRS]);

impl<const K: usize> Columns<K> {
    /// The columns, evaluated at compile time.
    const NEW: Self = {
        let mut columns = [[0; 2]; PAIRS];
        let mut j = 0;
        while j < PAIRS {
            // Column c is where the map sends the basis vector x^c.
            columns[j] = [
                repeated_square(1 << (2 * j), K),
                repeated_square(1 << (2 * j + 1), K),
            ];
            j += 1;
        }
        Self(columns)
    };
}

/// `x^(2^K)` in one bit-matrix product.
#[inline]
pub(crate) fn square_times<const K: usize>(x: u64) -> u64 {
    // A reference in a constant is promoted to a static, so the table is never copied.
    let columns: &'static [[u64; 2]; PAIRS] = const { &Columns::<K>::NEW.0 };

    // SAFETY: this module compiles only with `aes`, which implies `neon`.
    //
    // Every load reads one whole row of a constant table.
    unsafe {
        // The operand in both quadwords, so each quadword tests its own bit.
        let x = vdupq_n_u64(x);

        // One masked pair of columns: all ones in a quadword exactly when its bit of x is set.
        let term = |bits: &[u64; 2], column: &[u64; 2]| {
            let mask = vtstq_u64(x, vld1q_u64(bits.as_ptr()));
            vandq_u64(mask, vld1q_u64(column.as_ptr()))
        };

        // Four partial sums in named registers.
        //
        // An array indexed by the loop counter would live in memory instead.
        let zero = uint64x2_t::zero();
        let (mut a, mut b, mut c, mut d) = (zero, zero, zero, zero);
        for (bits, column) in BITS
            .as_chunks::<CHAINS>()
            .0
            .iter()
            .zip(columns.as_chunks::<CHAINS>().0)
        {
            a = veorq_u64(a, term(&bits[0], &column[0]));
            b = veorq_u64(b, term(&bits[1], &column[1]));
            c = veorq_u64(c, term(&bits[2], &column[2]));
            d = veorq_u64(d, term(&bits[3], &column[3]));
        }

        // The even columns summed in one quadword, the odd columns in the other.
        let sum = a.xor3(b, c).xor(d);
        vgetq_lane_u64::<0>(sum) ^ vgetq_lane_u64::<1>(sum)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::square_times;
    use crate::clmul::gf64::repeated_square;
    use crate::clmul::poly_square_64;

    /// The operands whose images are extreme: the identities, all ones, the top bits.
    const CORNERS: [u64; 5] = [0, 1, u64::MAX, 1 << 63, 0xf << 60];

    #[test]
    fn the_matrix_product_is_exact_on_the_basis() {
        // Invariant: the map is linear, so agreeing on all 64 basis vectors settles every input.
        //
        // Fixture state: every run length the inversion chain takes.
        for c in 0..64 {
            let x = 1u64 << c;
            assert_eq!(square_times::<3>(x), repeated_square(x, 3), "x^{c}");
            assert_eq!(square_times::<6>(x), repeated_square(x, 6), "x^{c}");
            assert_eq!(square_times::<12>(x), repeated_square(x, 12), "x^{c}");
            assert_eq!(square_times::<24>(x), repeated_square(x, 24), "x^{c}");
        }
        for x in CORNERS {
            assert_eq!(square_times::<24>(x), repeated_square(x, 24), "{x:#x}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        #[test]
        fn the_matrix_product_matches_repeated_squaring(x: u64) {
            // The powers the inversion chain takes, against the field's own squaring.
            let repeated = |k: usize| (0..k).fold(x, |y, _| poly_square_64(y));
            prop_assert_eq!(square_times::<3>(x), repeated(3));
            prop_assert_eq!(square_times::<6>(x), repeated(6));
            prop_assert_eq!(square_times::<12>(x), repeated(12));
            prop_assert_eq!(square_times::<24>(x), repeated(24));
        }
    }
}
