//! Repeated squaring in `GF(2^64)` as one bit-matrix product, on NEON.
//!
//! Squaring is `F_2`-linear, so squaring `K` times is a fixed `64 x 64` bit matrix `M`.
//!
//! Its product with `x` is the sum of the columns that the set bits of `x` select:
//!
//! ```text
//!     M x  =  sum_c  x_c M_c              M_c  =  (x^c)^(2^K)
//! ```
//!
//! A register holds two columns, and one compare turns the two matching bits of `x` into masks.
//!
//! So a pair of columns costs one compare, one AND and one exclusive or, and no branch.
//!
//! The cost is the same for every `K`, and every column is read whatever `x` holds.

use core::arch::aarch64::{
    vandq_u64, vdupq_n_u64, veorq_u64, vgetq_lane_u64, vld1q_u64, vshlq_n_u64, vtstq_u64,
};

use super::gf64::square_times_slow;

/// The columns of `x -> x^(2^K)`, column `c` being the image of `x^c`.
struct Columns<const K: usize>([u64; 64]);

impl<const K: usize> Columns<K> {
    /// The columns, evaluated at compile time.
    const NEW: Self = Self::new();

    /// Squares every basis vector `K` times.
    const fn new() -> Self {
        let mut columns = [0u64; 64];
        let mut c = 0;
        while c < 64 {
            columns[c] = square_times_slow(1 << c, K);
            c += 1;
        }
        Self(columns)
    }
}

/// The bits the first four column pairs test, one pair per register.
///
/// Pair `p + 4` tests the bits of pair `p` moved up by eight.
const FIRST_BITS: [u64; 8] = [1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1 << 7];

/// `x^(2^K)` in one bit-matrix product.
#[inline]
pub(crate) fn square_times<const K: usize>(x: u64) -> u64 {
    // A reference in a constant is promoted to a static, so the table is never copied.
    let columns: &'static [u64; 64] = const { &Columns::<K>::NEW.0 };

    // SAFETY: this module compiles only with `neon`, which every intrinsic below requires.
    //
    // Every load reads two quadwords inside a table of 64 or 8.
    unsafe {
        let x = vdupq_n_u64(x);

        // Four chains of bit pairs, each stepping eight bits a turn.
        //
        // Four short chains keep the shifts off the critical path.
        let mut bits = [
            vld1q_u64(FIRST_BITS.as_ptr()),
            vld1q_u64(FIRST_BITS.as_ptr().add(2)),
            vld1q_u64(FIRST_BITS.as_ptr().add(4)),
            vld1q_u64(FIRST_BITS.as_ptr().add(6)),
        ];

        // Eight partial sums, so no exclusive-or chain is longer than four.
        let mut sums = [vdupq_n_u64(0); 8];
        for turn in 0..8 {
            for (chain, bit) in bits.iter_mut().enumerate() {
                let pair = 4 * turn + chain;
                let column = vld1q_u64(columns.as_ptr().add(2 * pair));

                // All ones in a lane whose bit of `x` is set, zero otherwise.
                let mask = vtstq_u64(x, *bit);
                sums[pair % 8] = veorq_u64(sums[pair % 8], vandq_u64(column, mask));
                *bit = vshlq_n_u64::<8>(*bit);
            }
        }

        // A three-level tree, then the two lanes together.
        let halves = [
            veorq_u64(sums[0], sums[1]),
            veorq_u64(sums[2], sums[3]),
            veorq_u64(sums[4], sums[5]),
            veorq_u64(sums[6], sums[7]),
        ];
        let sum = veorq_u64(
            veorq_u64(halves[0], halves[1]),
            veorq_u64(halves[2], halves[3]),
        );
        vgetq_lane_u64::<0>(sum) ^ vgetq_lane_u64::<1>(sum)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::square_times;
    use crate::clmul::gf64::square_times_slow;
    use crate::clmul::poly_square_64;

    #[test]
    fn the_matrix_product_is_exact_on_the_basis() {
        // Invariant: the map is linear, so agreeing on all 64 basis vectors settles every input.
        //
        // Fixture state: every run length the inversion chain takes.
        for c in 0..64 {
            let x = 1u64 << c;
            assert_eq!(square_times::<3>(x), square_times_slow(x, 3), "x^{c}");
            assert_eq!(square_times::<6>(x), square_times_slow(x, 6), "x^{c}");
            assert_eq!(square_times::<12>(x), square_times_slow(x, 12), "x^{c}");
            assert_eq!(square_times::<24>(x), square_times_slow(x, 24), "x^{c}");
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
