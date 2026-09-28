//! `GF(2^64)` and its cubic extension on one 128-bit register, over the shared lane algebra.
//!
//! A `GF(2^64)` element sits in the low quadword of a register.
//!
//! A `GF(2^192)` element takes two coordinates per register.
//!
//! A carryless multiply picks one quadword of each operand, so one register pair feeds two products:
//!
//! ```text
//!     pair   = [ a_0        , a_1       ]      the first two coordinates
//!     tail   = [ a_2        , a_0 + a_1 ]      the third, beside one Karatsuba sum
//!     sums   = [ a_0 + a_2  , a_1 + a_2 ]      the other two Karatsuba sums
//! ```
//!
//! The low and the high products of those three pairs are the six Karatsuba terms.

use super::wide::{HIGH_BY_HIGH, HIGH_BY_LOW, LOW_BY_LOW, Lanes64, Wide};

/// A 128-bit register, with the moves the scalar kernels need on top of the lane operations.
pub(crate) trait Register128: Lanes64 {
    /// Whether a carryless multiply costs no more than an exclusive or.
    ///
    /// Then the nine schoolbook products beat Karatsuba's six, whose operand sums need shuffles.
    const CHEAP_MULTIPLY: bool;

    /// The value in the low quadword, and zero above it.
    fn lift(value: u64) -> Self;

    /// The low quadword.
    fn lower(self) -> u64;

    /// The two quadwords exchanged.
    fn swap(self) -> Self;

    /// The first two coordinates as one register, and the third in the low quadword of another.
    ///
    /// Each register is one load, in the shape a stored result takes.
    ///
    /// So a result forwards straight into the next product.
    fn load(a: &[u64; 3]) -> (Self, Self);

    /// The three coordinates, from a register holding the first two and one holding the third.
    fn store(pair: Self, last: Self) -> [u64; 3];

    /// The value in the low quadword, where the mixed carryless product reads it.
    fn load_scalar(k: &u64) -> Self;
}

/// Reduces the 128-bit product in a register to the element in its low quadword.
#[inline(always)]
fn reduce<R: Register128>(product: R) -> u64 {
    product.reduce_lane().lower()
}

/// Multiplication in `GF(2^64)`, taking and returning the polynomial representation.
///
/// One carryless product, then a fold that never leaves the vector register.
#[inline]
pub(crate) fn poly_mul_64<R: Register128>(a: u64, b: u64) -> u64 {
    // The 128-bit product fills the register, then folds back into its low quadword.
    reduce(R::lift(a).clmul::<LOW_BY_LOW>(R::lift(b)))
}

/// Squaring in `GF(2^64)`, taking and returning the polynomial representation.
#[inline]
pub(crate) fn poly_square_64<R: Register128>(a: u64) -> u64 {
    let x = R::lift(a);

    // The carryless square spreads the bits of x to even positions, then folds.
    reduce(x.clmul::<LOW_BY_LOW>(x))
}

/// A `GF(2^64)` dot product, summing unreduced products before one reduction.
#[inline]
pub(crate) fn poly_dot_64<R: Register128>(pairs: impl Iterator<Item = (u64, u64)>) -> u64 {
    // Reduction is linear, so the sum stays unreduced until the fold below.
    let sum = pairs.fold(R::zero(), |sum, (a, b)| {
        sum.xor(R::lift(a).clmul::<LOW_BY_LOW>(R::lift(b)))
    });

    // One fold for the whole sum.
    reduce(sum)
}

/// One cubic-extension element as the three registers the products read.
#[derive(Clone, Copy)]
struct Operand<R> {
    /// `[a_0, a_1]`.
    pair: R,
    /// `[a_2, a_0 + a_1]`.
    tail: R,
    /// `[a_0 + a_2, a_1 + a_2]`.
    sums: R,
}

impl<R: Register128> Operand<R> {
    /// Lays out the coordinates and the three Karatsuba sums.
    #[inline(always)]
    fn new(a: &[u64; 3]) -> Self {
        let (pair, last) = R::load(a);

        // `a_2` in both quadwords.
        let third = last.unpack_low(last);

        // `a_0 + a_1` in both quadwords, from the pair and its own swap.
        let crossed = pair.xor(pair.swap());

        // The three product-ready registers of the module header.
        Self {
            pair,
            tail: third.unpack_low(crossed),
            sums: pair.xor(third),
        }
    }
}

/// The three unreduced output coordinates, each a 128-bit product.
#[derive(Clone, Copy)]
struct Unreduced<R>([R; 3]);

impl<R: Register128> Unreduced<R> {
    /// The empty sum.
    #[inline(always)]
    fn zero() -> Self {
        Self([R::zero(); 3])
    }

    /// Coordinate-wise sum.
    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].xor(other.0[i])))
    }

    /// The reduced coordinates.
    #[inline(always)]
    fn reduce(self) -> [u64; 3] {
        let [r0, r1, r2] = self.0;

        // The first two coordinates reduce together, one per quadword.
        let pair = Wide { even: r0, odd: r1 }.reduce();

        // The third reduces alone, into its own low quadword.
        let last = r2.reduce_lane();

        // Back to memory in the same 16 + 8 byte shape the loads read.
        R::store(pair, last)
    }
}

/// The unreduced cubic product by Karatsuba, with the top two powers of `y` already folded.
///
/// ```text
///     r_0  =  c_0 + c_1 + c_2 + d_12
///     r_1  =  c_0 + d_01 + d_12
///     r_2  =  c_0 + c_1 + d_02
/// ```
#[inline(always)]
fn karatsuba<R: Register128>(a: &[u64; 3], b: &[u64; 3]) -> Unreduced<R> {
    let (a, b) = (Operand::<R>::new(a), Operand::<R>::new(b));

    // The six Karatsuba terms, two per register pair.
    let c0 = a.pair.clmul::<LOW_BY_LOW>(b.pair);
    let c1 = a.pair.clmul::<HIGH_BY_HIGH>(b.pair);
    let c2 = a.tail.clmul::<LOW_BY_LOW>(b.tail);
    let d01 = a.tail.clmul::<HIGH_BY_HIGH>(b.tail);
    let d02 = a.sums.clmul::<LOW_BY_LOW>(b.sums);
    let d12 = a.sums.clmul::<HIGH_BY_HIGH>(b.sums);

    // The one sum two coordinates share.
    let shared = c0.xor(d12);

    // The folded coordinates r_0, r_1, r_2, still unreduced in the base field.
    Unreduced([shared.xor3(c1, c2), shared.xor(d01), c0.xor3(c1, d02)])
}

/// The unreduced cubic product by the schoolbook, with the top two powers of `y` folded.
///
/// Nine products `c_ij = a_i b_j`, straight from the loaded registers:
///
/// ```text
///     t    =  c_12 + c_21                        the y^3 term, folded twice
///     r_0  =  c_00 + t
///     r_1  =  c_01 + c_10 + c_22 + t
///     r_2  =  c_02 + c_11 + c_20 + c_22
/// ```
#[inline(always)]
fn schoolbook<R: Register128>(a: &[u64; 3], b: &[u64; 3]) -> Unreduced<R> {
    let ((pair_a, last_a), (pair_b, last_b)) = (R::load(a), R::load(b));

    // The third coordinates in both quadwords, and the first pair of b swapped.
    //
    //     third_a  = [ a_2 , a_2 ]
    //     swapped  = [ b_1 , b_0 ]
    let (third_a, third_b) = (last_a.unpack_low(last_a), last_b.unpack_low(last_b));
    let swapped = pair_b.swap();

    // Each product takes one quadword of each register.
    let c00 = pair_a.clmul::<LOW_BY_LOW>(pair_b);
    let c11 = pair_a.clmul::<HIGH_BY_HIGH>(pair_b);
    let c01 = pair_a.clmul::<LOW_BY_LOW>(swapped);
    let c10 = pair_a.clmul::<HIGH_BY_HIGH>(swapped);
    let c02 = pair_a.clmul::<LOW_BY_LOW>(third_b);
    let c12 = pair_a.clmul::<HIGH_BY_HIGH>(third_b);
    let c20 = third_a.clmul::<LOW_BY_LOW>(pair_b);
    let c21 = third_a.clmul::<HIGH_BY_HIGH>(pair_b);
    let c22 = third_a.clmul::<LOW_BY_LOW>(third_b);

    // y^3 = y + 1 and y^4 = y^2 + y, applied to the five degrees in y.
    let t = c12.xor(c21);
    Unreduced([
        c00.xor(t),
        c01.xor3(c10, c22).xor(t),
        c02.xor3(c11, c20).xor(c22),
    ])
}

/// The unreduced cubic product, on the schedule the register favors.
#[inline(always)]
fn mul_unreduced<R: Register128>(a: &[u64; 3], b: &[u64; 3]) -> Unreduced<R> {
    if R::CHEAP_MULTIPLY {
        schoolbook(a, b)
    } else {
        karatsuba(a, b)
    }
}

/// Cubic multiplication: six or nine carryless products, then two reductions.
#[inline]
pub(crate) fn poly_mul_192<R: Register128>(a: &[u64; 3], b: &[u64; 3]) -> [u64; 3] {
    // One register fold for the first two coordinates, one for the third.
    mul_unreduced::<R>(a, b).reduce()
}

/// Cubic squaring: three carryless products.
///
/// ```text
///     (a_0 + a_1 y + a_2 y^2)^2  =  a_0^2 + a_2^2 y + (a_1^2 + a_2^2) y^2
/// ```
#[inline]
pub(crate) fn poly_square_192<R: Register128>(a: &[u64; 3]) -> [u64; 3] {
    let (pair, third) = R::load(a);

    // The three coordinate squares, unreduced.
    let s0 = pair.clmul::<LOW_BY_LOW>(pair);
    let s1 = pair.clmul::<HIGH_BY_HIGH>(pair);
    let s2 = third.clmul::<LOW_BY_LOW>(third);

    // y^4 = y^2 + y moves the top square onto the two coordinates above the constant.
    Unreduced([s0, s2, s1.xor(s2)]).reduce()
}

/// The unreduced multiple by a coefficient-field element: three products, no fold in `y`.
#[inline(always)]
fn mul_by_64_unreduced<R: Register128>(a: &[u64; 3], k: &u64) -> Unreduced<R> {
    let (pair, third) = R::load(a);
    let scalar = R::load_scalar(k);

    // a_0 k and a_1 k from the pair, a_2 k from the third register.
    Unreduced([
        pair.clmul::<LOW_BY_LOW>(scalar),
        pair.clmul::<HIGH_BY_LOW>(scalar),
        third.clmul::<LOW_BY_LOW>(scalar),
    ])
}

/// Multiplication by a coefficient-field element.
#[inline]
pub(crate) fn poly_mul_192_by_64<R: Register128>(a: &[u64; 3], k: &u64) -> [u64; 3] {
    // Three products, then the same two reductions as a full product.
    mul_by_64_unreduced::<R>(a, k).reduce()
}

/// A cubic dot product, summing unreduced products before one reduction.
#[inline]
pub(crate) fn poly_dot_192<'a, R: Register128>(
    pairs: impl Iterator<Item = (&'a [u64; 3], &'a [u64; 3])>,
) -> [u64; 3] {
    // Reduction is linear, so the whole sum reduces once.
    pairs
        .fold(Unreduced::<R>::zero(), |sum, (a, b)| {
            sum.xor(mul_unreduced(a, b))
        })
        .reduce()
}

/// A sum of coefficient-field multiples, before one reduction.
#[inline]
pub(crate) fn poly_dot_192_by_64<'a, R: Register128>(
    pairs: impl Iterator<Item = (&'a [u64; 3], &'a u64)>,
) -> [u64; 3] {
    // Reduction is linear, so the whole sum reduces once.
    pairs
        .fold(Unreduced::<R>::zero(), |sum, (a, k)| {
            sum.xor(mul_by_64_unreduced(a, k))
        })
        .reduce()
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{
        Register128, karatsuba, poly_dot_64, poly_dot_192, poly_dot_192_by_64, poly_mul_64,
        poly_mul_192, poly_mul_192_by_64, poly_square_64, poly_square_192, schoolbook,
    };
    use crate::clmul::wide::model::Model;
    use crate::clmul::{poly_mul_64 as reference_mul_64, poly_mul_192 as reference_mul_192};

    // The model register: moves written with plain array indexing.
    impl Register128 for Model<1> {
        const CHEAP_MULTIPLY: bool = false;

        fn lift(value: u64) -> Self {
            Self([[value, 0]])
        }

        fn lower(self) -> u64 {
            self.0[0][0]
        }

        fn swap(self) -> Self {
            Self([[self.0[0][1], self.0[0][0]]])
        }

        fn load(a: &[u64; 3]) -> (Self, Self) {
            // The upper quadword of the third register is junk the kernels must ignore.
            (Self([[a[0], a[1]]]), Self([[a[2], u64::MAX]]))
        }

        fn store(pair: Self, last: Self) -> [u64; 3] {
            [pair.0[0][0], pair.0[0][1], last.0[0][0]]
        }

        fn load_scalar(k: &u64) -> Self {
            // Junk in the upper quadword again, which no product may read.
            Self([[*k, u64::MAX]])
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn the_register_kernels_match_the_selected_backend(
            a: [u64; 3], b: [u64; 3], c: [u64; 3], k: u64, l: u64,
        ) {
            // Invariant: the register choreography computes the field operations exactly.
            //
            // Fixture state: the model register, against the backend this target dispatches to.
            type M = Model<1>;
            prop_assert_eq!(poly_mul_64::<M>(a[0], b[0]), reference_mul_64(a[0], b[0]));
            prop_assert_eq!(poly_square_64::<M>(a[0]), reference_mul_64(a[0], a[0]));
            prop_assert_eq!(
                poly_dot_64::<M>([(a[0], b[0]), (a[1], b[1])].into_iter()),
                reference_mul_64(a[0], b[0]) ^ reference_mul_64(a[1], b[1])
            );

            // Both product schedules, whichever the model register would pick.
            prop_assert_eq!(karatsuba::<M>(&a, &b).reduce(), reference_mul_192(&a, &b));
            prop_assert_eq!(schoolbook::<M>(&a, &b).reduce(), reference_mul_192(&a, &b));

            // The cubic kernels against the dispatched cubic product.
            prop_assert_eq!(poly_mul_192::<M>(&a, &b), reference_mul_192(&a, &b));
            prop_assert_eq!(poly_square_192::<M>(&a), reference_mul_192(&a, &a));
            prop_assert_eq!(poly_mul_192_by_64::<M>(&a, &k), reference_mul_192(&a, &[k, 0, 0]));

            // Both sums, each against its terms reduced one by one.
            let sum = |x: [u64; 3], y: [u64; 3]| core::array::from_fn(|i| x[i] ^ y[i]);
            prop_assert_eq!(
                poly_dot_192::<M>([(&a, &b), (&b, &c)].into_iter()),
                sum(reference_mul_192(&a, &b), reference_mul_192(&b, &c))
            );
            prop_assert_eq!(
                poly_dot_192_by_64::<M>([(&a, &k), (&c, &l)].into_iter()),
                sum(reference_mul_192(&a, &[k, 0, 0]), reference_mul_192(&c, &[l, 0, 0]))
            );
        }
    }
}
