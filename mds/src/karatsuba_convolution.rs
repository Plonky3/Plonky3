//! Calculate the convolution of two vectors using a Karatsuba-style
//! decomposition and the CRT.
//!
//! This is not a new idea, but we did have the pleasure of
//! reinventing it independently. Some references:
//! - `<https://cr.yp.to/lineartime/multapps-20080515.pdf>`
//! - `<https://2π.com/23/convolution/>`
//!
//! Given a vector v \in F^N, let v(x) \in F[x] denote the polynomial
//! v_0 + v_1 x + ... + v_{N - 1} x^{N - 1}.  Then w is equal to the
//! convolution v * u if and only if w(x) = v(x)u(x) mod x^N - 1.
//! Additionally, define the negacyclic convolution by w(x) = v(x)u(x)
//! mod x^N + 1.  Using the Chinese remainder theorem we can compute
//! w(x) as
//!     w(x) = 1/2 (w_0(x) + w_1(x)) + x^{N/2}/2 (w_0(x) - w_1(x))
//! where
//!     w_0 = v(x)u(x) mod x^{N/2} - 1
//!     w_1 = v(x)u(x) mod x^{N/2} + 1
//!
//! To compute w_0 and w_1 we first compute
//!                  v_0(x) = v(x) mod x^{N/2} - 1
//!                  v_1(x) = v(x) mod x^{N/2} + 1
//!                  u_0(x) = u(x) mod x^{N/2} - 1
//!                  u_1(x) = u(x) mod x^{N/2} + 1
//!
//! Now w_0 is the convolution of v_0 and u_0 which we can compute
//! recursively.  For w_1 we compute the negacyclic convolution
//! v_1(x)u_1(x) mod x^{N/2} + 1 using Karatsuba.
//!
//! There are 2 possible approaches to applying Karatsuba which mirror
//! the DIT vs DIF approaches to FFT's, the left/right decomposition
//! or the even/odd decomposition. The latter seems to have fewer
//! operations and so it is the one implemented below, though it does
//! require a bit more data manipulation. It works as follows:
//!
//! Define the even v_e and odd v_o parts so that v(x) = (v_e(x^2) + x v_o(x^2)).
//! Then v(x)u(x)
//!    = (v_e(x^2)u_e(x^2) + x^2 v_o(x^2)u_o(x^2))
//!      + x ((v_e(x^2) + v_o(x^2))(u_e(x^2) + u_o(x^2))
//!            - (v_e(x^2)u_e(x^2) + v_o(x^2)u_o(x^2)))
//! This reduces the problem to 3 negacyclic convolutions of size N/2 which
//! can be computed recursively.
//!
//! Of course, for small sizes we just explicitly write out the O(n^2)
//! approach.

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use p3_field::{Algebra, Field};

/// Bound alias for the wide operand type (used for both lhs and output).
///
/// Must support addition, subtraction, negation, and in-place variants.
pub trait ConvolutionElt:
    Add<Output = Self> + AddAssign + Copy + Neg<Output = Self> + Sub<Output = Self> + SubAssign
{
}

impl<T> ConvolutionElt for T where
    T: Add<Output = T> + AddAssign + Copy + Neg<Output = T> + Sub<Output = T> + SubAssign
{
}

/// Bound alias for the narrow operand type (rhs only).
///
/// Requires addition, subtraction, negation, and copy.
pub trait ConvolutionRhs:
    Add<Output = Self> + Copy + Neg<Output = Self> + Sub<Output = Self>
{
}

impl<T> ConvolutionRhs for T where T: Add<Output = T> + Copy + Neg<Output = T> + Sub<Output = T> {}

/// Trait for computing cyclic and negacyclic convolutions.
///
/// Implementors choose how to lift field elements into a wider type,
/// compute dot products, and reduce back.
/// This allows integer-lifted arithmetic (e.g. i64) to avoid modular
/// reductions inside the inner loops.
///
/// # Overflow contract
///
/// For a convolution of size N, it must be possible to add N elements
/// of type T without overflow, and similarly for U.
/// The product of one T and one U element must not overflow T after
/// about N further additions.
///
/// # Performance notes
///
/// In practice one operand is a compile-time constant (the MDS matrix).
/// The compiler folds the constant arithmetic at compile time.
/// For large matrices (N >= 24), the compile-time-generated constants
/// are about N times bigger than strictly necessary.
pub trait Convolve<F, T: ConvolutionElt, U: ConvolutionRhs> {
    /// Additive identity for the wide operand type `T`.
    ///
    /// Used to initialize output and scratch arrays before the convolution
    /// fills them with computed values.
    const T_ZERO: T;

    /// Additive identity for the narrow operand type `U`.
    ///
    /// Used to initialize temporary arrays for the RHS decomposition
    /// in the recursive CRT / Karatsuba steps.
    const U_ZERO: U;

    /// Divide an element of `T` by 2.
    ///
    /// - For integers (`i64`, `i128`): arithmetic right shift by 1.
    /// - For field elements: multiplication by the multiplicative inverse of 2.
    fn halve(val: T) -> T;

    /// Given an input element, retrieve the corresponding internal
    /// element that will be used in calculations.
    fn read(input: F) -> T;

    /// Given input vectors `lhs` and `rhs`, calculate their dot
    /// product. The result can be reduced with respect to the modulus
    /// (of `F`), but it must have the same lower 10 bits as the dot
    /// product if all inputs are considered integers. See
    /// `monty-31/src/mds.rs::barrett_red_monty31()` for an example
    /// of how this can be implemented in practice.
    fn parity_dot<const N: usize>(lhs: [T; N], rhs: [U; N]) -> T;

    /// Convert an internal element of type `T` back into an external
    /// element.
    fn reduce(z: T) -> F;

    /// Convolve `lhs` and `rhs`.
    ///
    /// The parameter `conv` should be the function in this trait that
    /// corresponds to length `N`.
    #[inline(always)]
    fn apply<const N: usize, C: Fn([T; N], [U; N], &mut [T])>(
        lhs: [F; N],
        rhs: [U; N],
        conv: C,
    ) -> [F; N] {
        let lhs = lhs.map(Self::read);
        let mut output = [Self::T_ZERO; N];
        conv(lhs, rhs, &mut output);
        output.map(Self::reduce)
    }

    #[inline(always)]
    fn conv3(lhs: [T; 3], rhs: [U; 3], output: &mut [T]) {
        output[0] = Self::parity_dot(lhs, [rhs[0], rhs[2], rhs[1]]);
        output[1] = Self::parity_dot(lhs, [rhs[1], rhs[0], rhs[2]]);
        output[2] = Self::parity_dot(lhs, [rhs[2], rhs[1], rhs[0]]);
    }

    #[inline(always)]
    fn negacyclic_conv3(lhs: [T; 3], rhs: [U; 3], output: &mut [T]) {
        output[0] = Self::parity_dot(lhs, [rhs[0], -rhs[2], -rhs[1]]);
        output[1] = Self::parity_dot(lhs, [rhs[1], rhs[0], -rhs[2]]);
        output[2] = Self::parity_dot(lhs, [rhs[2], rhs[1], rhs[0]]);
    }

    #[inline(always)]
    fn conv4(lhs: [T; 4], rhs: [U; 4], output: &mut [T]) {
        // NB: This is just explicitly implementing
        // conv_n_recursive::<4, 2, _, _>(lhs, rhs, output, Self::conv2, Self::negacyclic_conv2)
        let u_p = [lhs[0] + lhs[2], lhs[1] + lhs[3]];
        let u_m = [lhs[0] - lhs[2], lhs[1] - lhs[3]];
        let v_p = [rhs[0] + rhs[2], rhs[1] + rhs[3]];
        let v_m = [rhs[0] - rhs[2], rhs[1] - rhs[3]];

        output[0] = Self::parity_dot(u_m, [v_m[0], -v_m[1]]);
        output[1] = Self::parity_dot(u_m, [v_m[1], v_m[0]]);
        output[2] = Self::parity_dot(u_p, v_p);
        output[3] = Self::parity_dot(u_p, [v_p[1], v_p[0]]);

        output[0] += output[2];
        output[1] += output[3];

        output[0] = Self::halve(output[0]);
        output[1] = Self::halve(output[1]);

        output[2] -= output[0];
        output[3] -= output[1];
    }

    #[inline(always)]
    fn negacyclic_conv4(lhs: [T; 4], rhs: [U; 4], output: &mut [T]) {
        output[0] = Self::parity_dot(lhs, [rhs[0], -rhs[3], -rhs[2], -rhs[1]]);
        output[1] = Self::parity_dot(lhs, [rhs[1], rhs[0], -rhs[3], -rhs[2]]);
        output[2] = Self::parity_dot(lhs, [rhs[2], rhs[1], rhs[0], -rhs[3]]);
        output[3] = Self::parity_dot(lhs, [rhs[3], rhs[2], rhs[1], rhs[0]]);
    }

    /// Compute output(x) = lhs(x)rhs(x) mod x^N - 1 recursively using
    /// a convolution and negacyclic convolution of size HALF_N = N/2.
    #[inline(always)]
    fn conv_n_recursive<const N: usize, const HALF_N: usize, C, NC>(
        lhs: [T; N],
        rhs: [U; N],
        output: &mut [T],
        inner_conv: C,
        inner_negacyclic_conv: NC,
    ) where
        C: Fn([T; HALF_N], [U; HALF_N], &mut [T]),
        NC: Fn([T; HALF_N], [U; HALF_N], &mut [T]),
    {
        debug_assert_eq!(2 * HALF_N, N);
        let mut lhs_pos = [Self::T_ZERO; HALF_N]; // lhs_pos = lhs(x) mod x^{N/2} - 1
        let mut lhs_neg = [Self::T_ZERO; HALF_N]; // lhs_neg = lhs(x) mod x^{N/2} + 1
        let mut rhs_pos = [Self::U_ZERO; HALF_N]; // rhs_pos = rhs(x) mod x^{N/2} - 1
        let mut rhs_neg = [Self::U_ZERO; HALF_N]; // rhs_neg = rhs(x) mod x^{N/2} + 1

        for i in 0..HALF_N {
            let s = lhs[i];
            let t = lhs[i + HALF_N];

            lhs_pos[i] = s + t;
            lhs_neg[i] = s - t;

            let s = rhs[i];
            let t = rhs[i + HALF_N];

            rhs_pos[i] = s + t;
            rhs_neg[i] = s - t;
        }

        let (left, right) = output.split_at_mut(HALF_N);

        // left = w1 = lhs(x)rhs(x) mod x^{N/2} + 1
        inner_negacyclic_conv(lhs_neg, rhs_neg, left);

        // right = w0 = lhs(x)rhs(x) mod x^{N/2} - 1
        inner_conv(lhs_pos, rhs_pos, right);

        for i in 0..HALF_N {
            left[i] += right[i]; // w_0 + w_1
            left[i] = Self::halve(left[i]); // (w_0 + w_1)/2
            right[i] -= left[i]; // (w_0 - w_1)/2
        }
    }

    /// Compute output(x) = lhs(x)rhs(x) mod x^N + 1 recursively using
    /// three negacyclic convolutions of size HALF_N = N/2.
    #[inline(always)]
    fn negacyclic_conv_n_recursive<const N: usize, const HALF_N: usize, NC>(
        lhs: [T; N],
        rhs: [U; N],
        output: &mut [T],
        inner_negacyclic_conv: NC,
    ) where
        NC: Fn([T; HALF_N], [U; HALF_N], &mut [T]),
    {
        debug_assert_eq!(2 * HALF_N, N);
        let mut lhs_even = [Self::T_ZERO; HALF_N];
        let mut lhs_odd = [Self::T_ZERO; HALF_N];
        let mut lhs_sum = [Self::T_ZERO; HALF_N];
        let mut rhs_even = [Self::U_ZERO; HALF_N];
        let mut rhs_odd = [Self::U_ZERO; HALF_N];
        let mut rhs_sum = [Self::U_ZERO; HALF_N];

        for i in 0..HALF_N {
            let s = lhs[2 * i];
            let t = lhs[2 * i + 1];
            lhs_even[i] = s;
            lhs_odd[i] = t;
            lhs_sum[i] = s + t;

            let s = rhs[2 * i];
            let t = rhs[2 * i + 1];
            rhs_even[i] = s;
            rhs_odd[i] = t;
            rhs_sum[i] = s + t;
        }

        let mut even_s_conv = [Self::T_ZERO; HALF_N];
        let (left, right) = output.split_at_mut(HALF_N);

        // Recursively compute the size N/2 negacyclic convolutions of
        // the even parts, odd parts, and sums.
        inner_negacyclic_conv(lhs_even, rhs_even, &mut even_s_conv);
        inner_negacyclic_conv(lhs_odd, rhs_odd, left);
        inner_negacyclic_conv(lhs_sum, rhs_sum, right);

        // Adjust so that the correct values are in right and
        // even_s_conv respectively:
        right[0] -= even_s_conv[0] + left[0];
        even_s_conv[0] -= left[HALF_N - 1];

        for i in 1..HALF_N {
            right[i] -= even_s_conv[i] + left[i];
            even_s_conv[i] += left[i - 1];
        }

        // Interleave even_s_conv and right in the output:
        for i in 0..HALF_N {
            output[2 * i] = even_s_conv[i];
            output[2 * i + 1] = output[i + HALF_N];
        }
    }

    #[inline(always)]
    fn conv6(lhs: [T; 6], rhs: [U; 6], output: &mut [T]) {
        Self::conv_n_recursive(lhs, rhs, output, Self::conv3, Self::negacyclic_conv3);
    }

    #[inline(always)]
    fn negacyclic_conv6(lhs: [T; 6], rhs: [U; 6], output: &mut [T]) {
        Self::negacyclic_conv_n_recursive(lhs, rhs, output, Self::negacyclic_conv3);
    }

    #[inline(always)]
    fn conv8(lhs: [T; 8], rhs: [U; 8], output: &mut [T]) {
        Self::conv_n_recursive(lhs, rhs, output, Self::conv4, Self::negacyclic_conv4);
    }

    #[inline(always)]
    fn negacyclic_conv8(lhs: [T; 8], rhs: [U; 8], output: &mut [T]) {
        Self::negacyclic_conv_n_recursive(lhs, rhs, output, Self::negacyclic_conv4);
    }

    #[inline(always)]
    fn conv12(lhs: [T; 12], rhs: [U; 12], output: &mut [T]) {
        Self::conv_n_recursive(lhs, rhs, output, Self::conv6, Self::negacyclic_conv6);
    }

    #[inline(always)]
    fn negacyclic_conv12(lhs: [T; 12], rhs: [U; 12], output: &mut [T]) {
        Self::negacyclic_conv_n_recursive(lhs, rhs, output, Self::negacyclic_conv6);
    }

    #[inline(always)]
    fn conv16(lhs: [T; 16], rhs: [U; 16], output: &mut [T]) {
        Self::conv_n_recursive(lhs, rhs, output, Self::conv8, Self::negacyclic_conv8);
    }

    #[inline(always)]
    fn negacyclic_conv16(lhs: [T; 16], rhs: [U; 16], output: &mut [T]) {
        Self::negacyclic_conv_n_recursive(lhs, rhs, output, Self::negacyclic_conv8);
    }

    #[inline(always)]
    fn conv24(lhs: [T; 24], rhs: [U; 24], output: &mut [T]) {
        Self::conv_n_recursive(lhs, rhs, output, Self::conv12, Self::negacyclic_conv12);
    }

    #[inline(always)]
    fn conv32(lhs: [T; 32], rhs: [U; 32], output: &mut [T]) {
        Self::conv_n_recursive(lhs, rhs, output, Self::conv16, Self::negacyclic_conv16);
    }

    #[inline(always)]
    fn negacyclic_conv32(lhs: [T; 32], rhs: [U; 32], output: &mut [T]) {
        Self::negacyclic_conv_n_recursive(lhs, rhs, output, Self::negacyclic_conv16);
    }

    #[inline(always)]
    fn conv64(lhs: [T; 64], rhs: [U; 64], output: &mut [T]) {
        Self::conv_n_recursive(lhs, rhs, output, Self::conv32, Self::negacyclic_conv32);
    }
}

/// Convolution implementor that stays entirely within the field.
///
/// No integer lifting — all operations are native field arithmetic.
/// Used by the public Karatsuba entry points for generic field/algebra pairs.
struct FieldConvolve<F, A>(PhantomData<(F, A)>);

impl<F: Field, A: Algebra<F> + Copy> Convolve<A, A, F> for FieldConvolve<F, A> {
    const T_ZERO: A = A::ZERO;
    const U_ZERO: F = F::ZERO;

    #[inline(always)]
    fn halve(val: A) -> A {
        val.halve()
    }

    #[inline(always)]
    fn read(input: A) -> A {
        input
    }

    #[inline(always)]
    fn parity_dot<const N: usize>(lhs: [A; N], rhs: [F; N]) -> A {
        A::mixed_dot_product(&lhs, &rhs)
    }

    #[inline(always)]
    fn reduce(z: A) -> A {
        z
    }
}

/// Circulant matrix-vector multiply for width 16 via Karatsuba convolution.
#[inline]
pub fn mds_circulant_karatsuba_16<F: Field, A: Algebra<F> + Copy>(
    state: &mut [A; 16],
    col: &[F; 16],
) {
    let input = *state;
    FieldConvolve::<F, A>::conv16(input, *col, state.as_mut_slice());
}

/// Circulant matrix-vector multiply for width 24 via Karatsuba convolution.
#[inline]
pub fn mds_circulant_karatsuba_24<F: Field, A: Algebra<F> + Copy>(
    state: &mut [A; 24],
    col: &[F; 24],
) {
    let input = *state;
    FieldConvolve::<F, A>::conv24(input, *col, state.as_mut_slice());
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;

    type F = BabyBear;

    fn arb_f() -> impl Strategy<Value = F> {
        prop::num::u32::ANY.prop_map(F::from_u32)
    }

    fn naive_cyclic_conv<const N: usize>(lhs: [F; N], rhs: [F; N]) -> [F; N] {
        // O(N^2) reference: w[i] = sum_j lhs[j] * rhs[(i - j) mod N].
        core::array::from_fn(|i| {
            let mut acc = F::ZERO;
            for j in 0..N {
                acc += lhs[j] * rhs[(N + i - j) % N];
            }
            acc
        })
    }

    fn naive_negacyclic_conv<const N: usize>(lhs: [F; N], rhs: [F; N]) -> [F; N] {
        // O(N^2) reference: w(x) = lhs(x) * rhs(x) mod (x^N + 1).
        // Coefficients that wrap past degree N-1 are subtracted (negacyclic).
        let mut out = [F::ZERO; N];
        for (i, &l) in lhs.iter().enumerate() {
            for (j, &r) in rhs.iter().enumerate() {
                let k = i + j;
                if k < N {
                    out[k] += l * r;
                } else {
                    out[k - N] -= l * r;
                }
            }
        }
        out
    }

    fn check_conv<const N: usize>(
        lhs: [F; N],
        rhs: [F; N],
        conv_fn: fn([F; N], [F; N], &mut [F]),
        naive_fn: fn([F; N], [F; N]) -> [F; N],
    ) {
        let expected = naive_fn(lhs, rhs);
        let mut output = [F::ZERO; N];
        conv_fn(lhs, rhs, &mut output);
        assert_eq!(output, expected, "convolution mismatch");
    }

    macro_rules! conv_test {
        ($name:ident, $n:expr, $conv:expr, $naive:expr, $arr:ident) => {
            proptest! {
                #[test]
                fn $name(
                    lhs in prop::array::$arr(arb_f()),
                    rhs in prop::array::$arr(arb_f()),
                ) {
                    check_conv::<$n>(lhs, rhs, $conv, $naive);
                }
            }
        };
    }

    // Width 3
    conv_test!(
        conv3_matches_naive,
        3,
        FieldConvolve::<F, F>::conv3,
        naive_cyclic_conv,
        uniform3
    );
    conv_test!(
        negacyclic_conv3_matches_naive,
        3,
        FieldConvolve::<F, F>::negacyclic_conv3,
        naive_negacyclic_conv,
        uniform3
    );

    // Width 4
    conv_test!(
        conv4_matches_naive,
        4,
        FieldConvolve::<F, F>::conv4,
        naive_cyclic_conv,
        uniform4
    );
    conv_test!(
        negacyclic_conv4_matches_naive,
        4,
        FieldConvolve::<F, F>::negacyclic_conv4,
        naive_negacyclic_conv,
        uniform4
    );

    // Width 6
    conv_test!(
        conv6_matches_naive,
        6,
        FieldConvolve::<F, F>::conv6,
        naive_cyclic_conv,
        uniform6
    );
    conv_test!(
        negacyclic_conv6_matches_naive,
        6,
        FieldConvolve::<F, F>::negacyclic_conv6,
        naive_negacyclic_conv,
        uniform6
    );

    // Width 8
    conv_test!(
        conv8_matches_naive,
        8,
        FieldConvolve::<F, F>::conv8,
        naive_cyclic_conv,
        uniform8
    );
    conv_test!(
        negacyclic_conv8_matches_naive,
        8,
        FieldConvolve::<F, F>::negacyclic_conv8,
        naive_negacyclic_conv,
        uniform8
    );

    // Width 12
    conv_test!(
        conv12_matches_naive,
        12,
        FieldConvolve::<F, F>::conv12,
        naive_cyclic_conv,
        uniform12
    );
    conv_test!(
        negacyclic_conv12_matches_naive,
        12,
        FieldConvolve::<F, F>::negacyclic_conv12,
        naive_negacyclic_conv,
        uniform12
    );

    // Width 16
    conv_test!(
        conv16_matches_naive,
        16,
        FieldConvolve::<F, F>::conv16,
        naive_cyclic_conv,
        uniform16
    );
    conv_test!(
        negacyclic_conv16_matches_naive,
        16,
        FieldConvolve::<F, F>::negacyclic_conv16,
        naive_negacyclic_conv,
        uniform16
    );

    // Width 24
    conv_test!(
        conv24_matches_naive,
        24,
        FieldConvolve::<F, F>::conv24,
        naive_cyclic_conv,
        uniform24
    );

    // Width 32
    conv_test!(
        conv32_matches_naive,
        32,
        FieldConvolve::<F, F>::conv32,
        naive_cyclic_conv,
        uniform32
    );
    conv_test!(
        negacyclic_conv32_matches_naive,
        32,
        FieldConvolve::<F, F>::negacyclic_conv32,
        naive_negacyclic_conv,
        uniform32
    );

    #[test]
    fn conv64_matches_naive_fixed() {
        let lhs: [F; 64] = core::array::from_fn(|i| F::from_u32(i as u32 + 1));
        let rhs: [F; 64] = core::array::from_fn(|i| F::from_u32(64 - i as u32));
        check_conv::<64>(lhs, rhs, FieldConvolve::<F, F>::conv64, naive_cyclic_conv);
    }

    #[test]
    fn conv64_all_ones() {
        let ones = [F::ONE; 64];
        let expected = naive_cyclic_conv(ones, ones);
        let mut output = [F::ZERO; 64];
        FieldConvolve::<F, F>::conv64(ones, ones, &mut output);
        assert_eq!(output, expected);
    }

    proptest! {
        #[test]
        fn karatsuba_16_matches_naive(
            col in prop::array::uniform16(arb_f()),
            state in prop::array::uniform16(arb_f()),
        ) {
            let expected = naive_cyclic_conv(state, col);
            let mut actual = state;
            mds_circulant_karatsuba_16(&mut actual, &col);
            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn karatsuba_24_matches_naive(
            col in prop::array::uniform24(arb_f()),
            state in prop::array::uniform24(arb_f()),
        ) {
            let expected = naive_cyclic_conv(state, col);
            let mut actual = state;
            mds_circulant_karatsuba_24(&mut actual, &col);
            prop_assert_eq!(actual, expected);
        }
    }

    proptest! {
        #[test]
        fn conv8_commutative(
            a in prop::array::uniform8(arb_f()),
            b in prop::array::uniform8(arb_f()),
        ) {
            // Cyclic convolution is commutative: a * b = b * a.
            let mut ab = [F::ZERO; 8];
            let mut ba = [F::ZERO; 8];
            FieldConvolve::<F, F>::conv8(a, b, &mut ab);
            FieldConvolve::<F, F>::conv8(b, a, &mut ba);
            prop_assert_eq!(ab, ba);
        }

        #[test]
        fn conv8_identity(a in prop::array::uniform8(arb_f())) {
            // The delta impulse [1, 0, 0, ...] is the convolution identity.
            let mut id = [F::ZERO; 8];
            id[0] = F::ONE;
            let mut out = [F::ZERO; 8];
            FieldConvolve::<F, F>::conv8(a, id, &mut out);
            prop_assert_eq!(out, a);
        }

        #[test]
        fn conv8_zero(a in prop::array::uniform8(arb_f())) {
            // Convolving with the zero vector must produce all zeros.
            let zeros = [F::ZERO; 8];
            let mut out = [F::ZERO; 8];
            FieldConvolve::<F, F>::conv8(a, zeros, &mut out);
            prop_assert_eq!(out, zeros);
        }
    }
}
