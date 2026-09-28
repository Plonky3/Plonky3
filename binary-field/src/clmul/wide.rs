//! `GF(2^64)` and its cubic extension, written once over an abstract register of 64-bit lanes.
//!
//! One backend is a 128-bit register, serving the scalar kernels.
//!
//! Another is the 256-bit register behind the packings.
//!
//! A scalar model compiled under `cfg(test)` checks the algebra on every target.
//!
//! # The shape of a product
//!
//! A carryless multiply takes one quadword from each 128-bit lane of both operands.
//!
//! Taking the even quadwords, then the odd ones, covers every lane pair with two instructions:
//!
//! ```text
//!     a        = [ a_0 a_1 | a_2 a_3 | ... ]       one element per quadword
//!     even     = [ a_0 b_0 | a_2 b_2 | ... ]       128 bits per product
//!     odd      = [ a_1 b_1 | a_3 b_3 | ... ]
//! ```
//!
//! Interleaving the halves of those two registers puts every product back in element order:
//!
//! ```text
//!     low      = [ lo(a_0 b_0) lo(a_1 b_1) | ... ]
//!     high     = [ hi(a_0 b_0) hi(a_1 b_1) | ... ]
//! ```
//!
//! Reduction then acts on 64-bit lanes, with shifts and a byte shuffle and no multiply at all.

// The cubic kernels and the byte table serve only the packings and the byte-shuffle backends.
//
// A scalar build without them still compiles this module for the reduction it shares.
#![cfg_attr(
    not(all(
        target_arch = "x86_64",
        target_feature = "vpclmulqdq",
        any(target_feature = "avx2", target_feature = "avx512f")
    )),
    allow(dead_code)
)]

/// Selects the low quadword of both operands.
pub(crate) const LOW_BY_LOW: i32 = 0x00;

/// Selects the high quadword of both operands.
pub(crate) const HIGH_BY_HIGH: i32 = 0x11;

/// Selects the high quadword of the first operand and the low quadword of the second.
pub(crate) const HIGH_BY_LOW: i32 = 0x01;

/// The fold of the top nibble of a high half, as a 16-entry byte table.
///
/// Entry `s` is `g(spill(s))`, with both maps as in the reduction below:
///
/// ```text
///     spill(s) = s ^ (s >> 1) ^ (s >> 3)          the bits that cross x^64 again
///     g(v)     = v ^ (v << 1) ^ (v << 3) ^ (v << 4)
/// ```
///
/// A spill has degree at most 3, so every entry fits in a byte.
///
/// A byte shuffle is then the whole lookup.
pub(crate) const TOP_NIBBLE_FOLD: [u8; 16] = {
    let mut table = [0u8; 16];
    let mut s = 0;
    while s < 16 {
        // Nibble s holds bits 60 .. 63 of a high half, so bit 3 of s is bit 63.
        let spill = s ^ (s >> 1) ^ (s >> 3);

        // Fold the spill by the tail once more.
        //
        // Degree 3 + 4 = 7 stays inside the byte.
        table[s] = (spill ^ (spill << 1) ^ (spill << 3) ^ (spill << 4)) as u8;
        s += 1;
    }
    table
};

/// The lane operations the algebra below is written against.
///
/// Every method acts on 64-bit or 128-bit lanes independently, never across a 128-bit lane.
pub(crate) trait Lanes64: Copy {
    /// All lanes zero.
    fn zero() -> Self;

    /// Bitwise exclusive or.
    fn xor(self, other: Self) -> Self;

    /// Three-way exclusive or.
    ///
    /// One instruction where the target has a ternary logic op, two otherwise.
    #[inline(always)]
    fn xor3(self, b: Self, c: Self) -> Self {
        self.xor(b).xor(c)
    }

    /// The carryless product of one quadword of each operand, in every 128-bit lane.
    ///
    /// Bit 0 of `IMM` picks the half of the first argument, bit 4 the half of the second.
    fn clmul<const IMM: i32>(self, other: Self) -> Self;

    /// The low quadword of each operand, paired within each 128-bit lane.
    fn unpack_low(self, other: Self) -> Self;

    /// The high quadword of each operand, paired within each 128-bit lane.
    fn unpack_high(self, other: Self) -> Self;

    /// Each 64-bit lane shifted left.
    fn shl<const N: i32>(self) -> Self;

    /// Each 64-bit lane shifted right.
    fn shr<const N: i32>(self) -> Self;

    /// The part of the fold that the top nibble of each 64-bit lane contributes.
    ///
    /// Written with shifts here.
    ///
    /// A backend with a byte shuffle replaces them with one lookup in the nibble table.
    #[inline(always)]
    fn fold_top_nibble(self) -> Self {
        // The bits of the high half that the first fold pushes past x^63.
        let spill = self.shr::<63>().xor3(self.shr::<61>(), self.shr::<60>());

        // Their own fold by the tail, x^4 + x^3 + x + 1.
        spill
            .xor3(spill.shl::<1>(), spill.shl::<3>())
            .xor(spill.shl::<4>())
    }
}

/// Reduces `low + high x^64` modulo `x^64 + x^4 + x^3 + x + 1`, lane by lane.
///
/// # Algorithm
///
/// The modulus rewrites `x^64` as the tail `T = x^4 + x^3 + x + 1`:
///
/// ```text
///     high x^64  =  high T  =  g(high)  +  spill(high) x^64
///     g(v)       =  v ^ (v << 1) ^ (v << 3) ^ (v << 4)         truncated to 64 bits
///     spill(v)   =  (v >> 63) ^ (v >> 61) ^ (v >> 60)          the bits g drops
/// ```
///
/// The spill has degree at most 3, so its own fold `g(spill)` has degree at most 7.
///
/// That second fold is exact: nothing crosses `x^64` a third time.
///
/// The spill depends only on the top nibble, so its fold is one table lookup.
#[inline(always)]
pub(crate) fn fold<L: Lanes64>(low: L, high: L) -> L {
    // The shifted terms of g(high), truncated to 64 bits by the lane width.
    let shifted = high.shl::<1>().xor3(high.shl::<3>(), high.shl::<4>());

    // low + g(high) + g(spill(high)), the two folds summed by linearity.
    low.xor3(high, high.fold_top_nibble()).xor(shifted)
}

/// The unreduced products of two registers of elements, split by quadword parity.
///
/// Addition of products is exclusive or, so a sum of them stays in this form.
///
/// Reduction is `F_2`-linear, so a whole sum is reduced once at the end.
#[derive(Clone, Copy)]
pub(crate) struct Wide<L> {
    /// The 128-bit products of the elements in even quadwords.
    pub(crate) even: L,
    /// The 128-bit products of the elements in odd quadwords.
    pub(crate) odd: L,
}

impl<L: Lanes64> Wide<L> {
    /// The empty sum.
    #[inline(always)]
    pub(crate) fn zero() -> Self {
        Self {
            even: L::zero(),
            odd: L::zero(),
        }
    }

    /// Every lane of `a` times the same lane of `b`, unreduced.
    #[inline(always)]
    pub(crate) fn mul(a: L, b: L) -> Self {
        // Two instructions cover every lane: even quadwords, then odd ones.
        Self {
            even: a.clmul::<LOW_BY_LOW>(b),
            odd: a.clmul::<HIGH_BY_HIGH>(b),
        }
    }

    /// Lane-wise sum.
    #[inline(always)]
    pub(crate) fn xor(self, other: Self) -> Self {
        Self {
            even: self.even.xor(other.even),
            odd: self.odd.xor(other.odd),
        }
    }

    /// Lane-wise sum of three.
    #[inline(always)]
    pub(crate) fn xor3(self, b: Self, c: Self) -> Self {
        Self {
            even: self.even.xor3(b.even, c.even),
            odd: self.odd.xor3(b.odd, c.odd),
        }
    }

    /// The reduced elements, back in their original lane order.
    #[inline(always)]
    pub(crate) fn reduce(self) -> L {
        // Interleave the halves so quadword k holds the low, then the high, half of product k.
        //
        //     low   = [ lo(p_0) lo(p_1) | lo(p_2) lo(p_3) | ... ]
        //     high  = [ hi(p_0) hi(p_1) | hi(p_2) hi(p_3) | ... ]
        fold(
            self.even.unpack_low(self.odd),
            self.even.unpack_high(self.odd),
        )
    }
}

/// The unreduced product in `GF(2^64)[y] / (y^3 + y + 1)`, one coordinate register each.
///
/// # Algorithm
///
/// Karatsuba over three limbs: six products instead of the schoolbook nine.
///
/// ```text
///     c_i   = a_i b_i
///     d_ij  = (a_i + a_j) (b_i + b_j)  =  c_i + c_j + (a_i b_j + a_j b_i)
/// ```
///
/// The product spans `y^0 .. y^4`, and `y^3 = y + 1`, `y^4 = y^2 + y` fold the top two:
///
/// ```text
///     r_0  =  c_0 + y^3 term                 =  c_0 + c_1 + c_2 + d_12
///     r_1  =  y^1 term + y^3 + y^4 terms     =  c_0 + d_01 + d_12
///     r_2  =  y^2 term + y^4 term            =  c_0 + c_1 + d_02
/// ```
///
/// Folding before reducing leaves three coordinates to reduce, not five.
#[inline(always)]
pub(crate) fn cubic_mul<L: Lanes64>(a: [L; 3], b: [L; 3]) -> [Wide<L>; 3] {
    let [a0, a1, a2] = a;
    let [b0, b1, b2] = b;

    // The three diagonal products.
    let c0 = Wide::mul(a0, b0);
    let c1 = Wide::mul(a1, b1);
    let c2 = Wide::mul(a2, b2);

    // One product per pair of limbs, carrying both of that pair's cross terms.
    let d01 = Wide::mul(a0.xor(a1), b0.xor(b1));
    let d02 = Wide::mul(a0.xor(a2), b0.xor(b2));
    let d12 = Wide::mul(a1.xor(a2), b1.xor(b2));

    // The one sum two coordinates share.
    let shared = c0.xor(d12);

    // The folded coordinates r_0, r_1, r_2, still unreduced in the base field.
    [shared.xor3(c1, c2), shared.xor(d01), c0.xor3(c1, d02)]
}

/// The unreduced square in the cubic extension.
///
/// Squaring is `F_2`-linear, so the cross terms vanish and only the fold survives:
///
/// ```text
///     (a_0 + a_1 y + a_2 y^2)^2  =  a_0^2 + a_1^2 y^2 + a_2^2 y^4
///                                =  a_0^2 + a_2^2 y + (a_1^2 + a_2^2) y^2
/// ```
#[inline(always)]
pub(crate) fn cubic_square<L: Lanes64>(a: [L; 3]) -> [Wide<L>; 3] {
    // The three coordinate squares, unreduced.
    let [s0, s1, s2] = a.map(|x| Wide::mul(x, x));

    // y^4 = y^2 + y moves the top square onto the two coordinates above the constant.
    [s0, s2, s1.xor(s2)]
}

/// The unreduced product of an extension element by a coefficient-field element.
///
/// The scalar stays inside each coordinate: three products, no fold.
#[inline(always)]
pub(crate) fn cubic_mul_base<L: Lanes64>(a: [L; 3], k: L) -> [Wide<L>; 3] {
    // One product per coordinate.
    //
    // The degree in y never grows, so nothing folds.
    a.map(|x| Wide::mul(x, k))
}

/// A scalar stand-in for a register, so the algebra above is checked on every target.
#[cfg(test)]
pub(crate) mod model {
    use super::Lanes64;

    /// Two 64-bit lanes per 128-bit lane, held in plain integers.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct Model<const LANES: usize>(pub(crate) [[u64; 2]; LANES]);

    impl<const LANES: usize> Model<LANES> {
        /// Applies `f` to every 64-bit lane.
        fn map(self, f: impl Fn(u64) -> u64) -> Self {
            Self(self.0.map(|lane| lane.map(&f)))
        }

        /// Applies `f` to every pair of 128-bit lanes.
        fn zip(self, other: Self, f: impl Fn([u64; 2], [u64; 2]) -> [u64; 2]) -> Self {
            Self(core::array::from_fn(|i| f(self.0[i], other.0[i])))
        }
    }

    impl<const LANES: usize> Lanes64 for Model<LANES> {
        fn zero() -> Self {
            Self([[0; 2]; LANES])
        }

        fn xor(self, other: Self) -> Self {
            self.zip(other, |x, y| [x[0] ^ y[0], x[1] ^ y[1]])
        }

        fn clmul<const IMM: i32>(self, other: Self) -> Self {
            self.zip(other, |x, y| {
                // Bit 0 of the immediate picks the quadword of x, bit 4 the quadword of y.
                let product = super::super::scalar_clmul_64x64(
                    x[(IMM & 0x01) as usize],
                    y[((IMM >> 4) & 0x01) as usize],
                );
                // The 128-bit product fills the whole lane, low quadword first.
                [product as u64, (product >> 64) as u64]
            })
        }

        fn unpack_low(self, other: Self) -> Self {
            self.zip(other, |x, y| [x[0], y[0]])
        }

        fn unpack_high(self, other: Self) -> Self {
            self.zip(other, |x, y| [x[1], y[1]])
        }

        fn shl<const N: i32>(self) -> Self {
            self.map(|v| v << N)
        }

        fn shr<const N: i32>(self) -> Self {
            self.map(|v| v >> N)
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::model::Model;
    use super::{Lanes64, TOP_NIBBLE_FOLD, Wide, cubic_mul, cubic_mul_base, cubic_square, fold};
    use crate::clmul::scalar_clmul_64x64;

    /// Lanes in the model: two 128-bit lanes, so a lane-crossing slip shows as a mismatch.
    const LANES: usize = 2;

    /// Multiplication straight from the modulus, one bit of `b` at a time.
    fn schoolbook(a: u64, b: u64) -> u64 {
        // Horner in x over the bits of b, from the top one down.
        let mut acc = 0u64;
        for i in (0..64).rev() {
            // Multiply the accumulator by x, rewriting x^64 as the tail 0b1_1011.
            let overflowed = acc >> 63 == 1;
            acc <<= 1;
            if overflowed {
                acc ^= 0b1_1011;
            }

            // Add a when this bit of b is set.
            if (b >> i) & 1 == 1 {
                acc ^= a;
            }
        }
        acc
    }

    /// Cubic multiplication by the schoolbook nine products and a term-by-term fold.
    fn cubic_schoolbook(a: [u64; 3], b: [u64; 3]) -> [u64; 3] {
        // Nine reduced products, placed by total degree in y.
        let mut raw = [0u64; 5];
        for i in 0..3 {
            for j in 0..3 {
                raw[i + j] ^= schoolbook(a[i], b[j]);
            }
        }
        // y^4 = y^2 + y, then y^3 = y + 1.
        [raw[0] ^ raw[3], raw[1] ^ raw[3] ^ raw[4], raw[2] ^ raw[4]]
    }

    /// Four model elements, lane `2i + j` at quadword `j` of 128-bit lane `i`.
    fn model(values: [u64; 2 * LANES]) -> Model<LANES> {
        // Consecutive elements pair up inside one 128-bit lane, as in a real register.
        Model(core::array::from_fn(|i| [values[2 * i], values[2 * i + 1]]))
    }

    /// The model's lanes read back in element order.
    fn values(m: Model<LANES>) -> [u64; 2 * LANES] {
        // Element k sits in 128-bit lane k / 2, quadword k % 2.
        core::array::from_fn(|k| m.0[k / 2][k % 2])
    }

    /// The operands whose products and folds are extreme.
    ///
    /// ```text
    ///     0, 1              the identities
    ///     all ones          every coefficient of the product is live
    ///     x^63              the highest degree, so the spill is largest
    ///     0x1b              the modulus tail itself
    ///     0xf << 60         every bit of the top nibble, which the table folds
    /// ```
    const CORNERS: [u64; 6] = [0, 1, u64::MAX, 1 << 63, 0x1b, 0xf << 60];

    #[test]
    fn the_nibble_table_matches_the_shifted_fold() {
        // Invariant: the byte table and the shift form fold every nibble the same way.
        //
        // Fixture state: all 16 nibbles, each placed at bits 60 .. 63 of one lane.
        for s in 0..16u64 {
            let lanes = Model::<1>([[s << 60, 0]]);

            // The model keeps the trait's shift form, so it is the independent side.
            let default = <Model<1> as Lanes64>::fold_top_nibble(lanes);
            assert_eq!(
                u64::from(TOP_NIBBLE_FOLD[s as usize]),
                default.0[0][0],
                "{s}"
            );
        }
    }

    #[test]
    fn the_fold_is_exact_on_the_extremes() {
        // Invariant: `low + high x^64` reduces to the schoolbook product of its factors.
        //
        // Fixture state: 6 corners squared = 36 pairs, including the largest spill.
        for a in CORNERS {
            for b in CORNERS {
                // The bit-serial 128-bit product, split into its two halves for the fold.
                let product = scalar_clmul_64x64(a, b);
                let got = fold(
                    Model::<1>([[product as u64, 0]]),
                    Model::<1>([[(product >> 64) as u64, 0]]),
                );
                assert_eq!(got.0[0][0], schoolbook(a, b), "{a:#x} * {b:#x}");
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn the_lane_product_matches_the_schoolbook_one(
            a in any::<[u64; 2 * LANES]>(),
            b in any::<[u64; 2 * LANES]>(),
        ) {
            // Invariant: every lane holds its own product, in its own position.
            //
            // Fixture state: 4 elements across 2 lanes, so a swapped half shows up.
            let got = values(Wide::mul(model(a), model(b)).reduce());
            let want: [u64; 2 * LANES] = core::array::from_fn(|k| schoolbook(a[k], b[k]));
            prop_assert_eq!(got, want);
        }

        #[test]
        fn a_sum_of_wide_products_reduces_to_the_sum_of_products(
            a in any::<[u64; 2 * LANES]>(),
            b in any::<[u64; 2 * LANES]>(),
            c in any::<[u64; 2 * LANES]>(),
        ) {
            // Invariant: reduction is linear, so a sum reduces once.
            //
            //     reduce(ab + bc + ca)  =  reduce(ab) + reduce(bc) + reduce(ca)
            let wide = Wide::mul(model(a), model(b))
                .xor3(Wide::mul(model(b), model(c)), Wide::mul(model(c), model(a)));
            let want: [u64; 2 * LANES] = core::array::from_fn(|k| {
                schoolbook(a[k], b[k]) ^ schoolbook(b[k], c[k]) ^ schoolbook(c[k], a[k])
            });
            prop_assert_eq!(values(wide.reduce()), want);
        }

        #[test]
        fn the_cubic_kernels_match_the_schoolbook_ones(
            a in any::<[[u64; 2 * LANES]; 3]>(),
            b in any::<[[u64; 2 * LANES]; 3]>(),
            k in any::<[u64; 2 * LANES]>(),
        ) {
            // Fixture state: coordinate i of lane j of an operand is a[i][j].
            let (x, y) = (a.map(model), b.map(model));

            // All three kernels, reduced coordinate by coordinate.
            let product = cubic_mul(x, y).map(|w| values(w.reduce()));
            let square = cubic_square(x).map(|w| values(w.reduce()));
            let scaled = cubic_mul_base(x, model(k)).map(|w| values(w.reduce()));

            // Each lane against the schoolbook product of its own element.
            for j in 0..2 * LANES {
                let (aj, bj) = ([a[0][j], a[1][j], a[2][j]], [b[0][j], b[1][j], b[2][j]]);
                let at = |r: [[u64; 2 * LANES]; 3]| [r[0][j], r[1][j], r[2][j]];
                prop_assert_eq!(at(product), cubic_schoolbook(aj, bj));
                prop_assert_eq!(at(square), cubic_schoolbook(aj, aj));
                prop_assert_eq!(at(scaled), cubic_schoolbook(aj, [k[j], 0, 0]));
            }
        }
    }
}
