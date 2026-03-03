//! Circulant matrix multiplication via Karatsuba convolution over field elements.
//!
//! # Overview
//!
//! A **circulant matrix** is fully defined by its first column. Every subsequent column
//! is a cyclic downward shift of the previous one:
//!
//! ```text
//!     ┌                  ┐   ┌    ┐       ┌    ┐
//!     │ c_0  c_3  c_2  c_1 │   │ s_0 │       │ r_0 │
//!     │ c_1  c_0  c_3  c_2 │ x │ s_1 │   =   │ r_1 │
//!     │ c_2  c_1  c_0  c_3 │   │ s_2 │       │ r_2 │
//!     │ c_3  c_2  c_1  c_0 │   │ s_3 │       │ r_3 │
//!     └                  ┘   └    ┘       └    ┘
//! ```
//!
//! Multiplying this matrix by a vector is a **cyclic convolution**:
//!
//! ```text
//!     r[i]  =  sum_j  c[(i - j) mod N] * s[j]
//! ```
//!
//! The naive approach costs O(N^2) multiplications. This module reduces it to roughly
//! O(N log N) by recursively decomposing the convolution via the **Chinese Remainder
//! Theorem** (CRT) and the **Karatsuba trick**.
//!
//! # Motivation
//!
//! This crate already provides an integer-based Karatsuba convolution that works on
//! `i64` / `i128` intermediates. That approach avoids intermediate modular reductions
//! entirely, and the halving step in the CRT recombination is a simple right bit-shift
//! (`>>= 1`).
//!
//! However, packed SIMD field types (e.g. `PackedMontyField31Neon`, `PackedMontyField31AVX2`)
//! have no meaningful bit-shift. They only support field arithmetic. This module provides
//! the same algorithm expressed entirely in field operations, using `halve()` (multiplication
//! by the inverse of 2) in place of bit-shifts.
//!
//! The trade-off is straightforward: more modular reductions, but compatible with any type
//! that implements `Algebra<F>`.
//!
//! # Algorithm
//!
//! ## CRT Decomposition (Cyclic -> Half-Size)
//!
//! The polynomial `x^N - 1` factors as `(x^{N/2} - 1)(x^{N/2} + 1)`. By the CRT,
//! a cyclic convolution of size N decomposes into:
//!
//! - **w_0**: a cyclic convolution of size N/2 (mod `x^{N/2} - 1`)
//! - **w_1**: a negacyclic convolution of size N/2 (mod `x^{N/2} + 1`)
//!
//! The recombination step reconstructs the full-size result:
//!
//! ```text
//!     lower half  =  (w_0 + w_1) / 2
//!     upper half  =  (w_0 - w_1) / 2
//! ```
//!
//! The division by 2 is where `halve()` is used.
//!
//! ## Karatsuba Decomposition (Negacyclic -> Three Sub-Problems)
//!
//! A negacyclic convolution of size N splits via even/odd index separation:
//!
//! ```text
//!     v(x)  =  v_even(x^2) + x * v_odd(x^2)
//! ```
//!
//! This reduces to **three** negacyclic convolutions of size N/2 (even x even,
//! odd x odd, and sum x sum), then recombines the results.
//!
//! ## Recursion Tree
//!
//! Applying both decompositions recursively builds a tree from the target size
//! down to small base cases (sizes 3 and 4), which are computed directly:
//!
//! ```text
//!     Cyclic 16 --> Cyclic 8 + Negacyclic 8
//!                        |              |
//!                        v              v
//!                    C4 + NC4      3 x NC4
//!
//!     Cyclic 24 --> Cyclic 12 + Negacyclic 12
//!                        |              |
//!                        v              v
//!                    C6 + NC6      3 x NC6
//!                     |    |          |
//!                     v    v          v
//!                    C3  NC3       3 x NC3
//! ```
//!
//! # References
//!
//! - <https://cr.yp.to/lineartime/multapps-20080515.pdf>
//! - <https://2π.com/23/convolution/>

use p3_field::{Algebra, Field};

/// Circulant matrix-vector multiply for width 16 via Karatsuba convolution.
///
/// Computes `state <- circulant(col) x state` where `col` is the first column
/// of the circulant matrix.
#[inline]
pub fn mds_circulant_karatsuba_16<F: Field, A: Algebra<F> + Copy>(
    state: &mut [A; 16],
    col: &[F; 16],
) {
    let input = *state;
    cyclic_conv16(input, *col, state.as_mut_slice());
}

/// Circulant matrix-vector multiply for width 24 via Karatsuba convolution.
///
/// Computes `state <- circulant(col) x state` where `col` is the first column
/// of the circulant matrix.
#[inline]
pub fn mds_circulant_karatsuba_24<F: Field, A: Algebra<F> + Copy>(
    state: &mut [A; 24],
    col: &[F; 24],
) {
    let input = *state;
    cyclic_conv24(input, *col, state.as_mut_slice());
}

/// Cyclic convolution of size N via CRT splitting into half-size sub-problems.
///
/// Decomposes `x^N - 1 = (x^{N/2} - 1)(x^{N/2} + 1)`, recurses on each factor,
/// then recombines with field-level halving.
#[inline(always)]
fn cyclic_conv_recursive<F, A, const N: usize, const HALF: usize>(
    lhs: [A; N],
    rhs: [F; N],
    output: &mut [A],
    inner_cyclic: fn([A; HALF], [F; HALF], &mut [A]),
    inner_negacyclic: fn([A; HALF], [F; HALF], &mut [A]),
) where
    F: Field,
    A: Algebra<F> + Copy,
{
    debug_assert_eq!(2 * HALF, N);

    // Reduce inputs modulo (x^{N/2} - 1) and (x^{N/2} + 1).
    let mut lhs_pos = [A::ZERO; HALF];
    let mut lhs_neg = [A::ZERO; HALF];
    let mut rhs_pos = [F::ZERO; HALF];
    let mut rhs_neg = [F::ZERO; HALF];

    for i in 0..HALF {
        lhs_pos[i] = lhs[i] + lhs[i + HALF];
        lhs_neg[i] = lhs[i] - lhs[i + HALF];
        rhs_pos[i] = rhs[i] + rhs[i + HALF];
        rhs_neg[i] = rhs[i] - rhs[i + HALF];
    }

    let (left, right) = output.split_at_mut(HALF);

    // w_1 = negacyclic convolution (mod x^{N/2} + 1).
    inner_negacyclic(lhs_neg, rhs_neg, left);

    // w_0 = cyclic convolution (mod x^{N/2} - 1).
    inner_cyclic(lhs_pos, rhs_pos, right);

    // CRT recombination with field halving.
    for i in 0..HALF {
        left[i] += right[i]; //  w_0 + w_1
        left[i] = left[i].halve(); // (w_0 + w_1) / 2
        right[i] -= left[i]; // (w_0 - w_1) / 2
    }
}

/// Negacyclic convolution of size N via even/odd Karatsuba decomposition.
///
/// Splits each polynomial into even- and odd-indexed coefficients, performs
/// three recursive negacyclic convolutions of half size, then recombines.
#[inline(always)]
fn negacyclic_conv_recursive<F, A, const N: usize, const HALF: usize>(
    lhs: [A; N],
    rhs: [F; N],
    output: &mut [A],
    inner_negacyclic: fn([A; HALF], [F; HALF], &mut [A]),
) where
    F: Field,
    A: Algebra<F> + Copy,
{
    debug_assert_eq!(2 * HALF, N);

    // Deinterleave into even and odd coefficients.
    let mut lhs_even = [A::ZERO; HALF];
    let mut lhs_odd = [A::ZERO; HALF];
    let mut lhs_sum = [A::ZERO; HALF];
    let mut rhs_even = [F::ZERO; HALF];
    let mut rhs_odd = [F::ZERO; HALF];
    let mut rhs_sum = [F::ZERO; HALF];

    for i in 0..HALF {
        let le = lhs[2 * i];
        let lo = lhs[2 * i + 1];
        lhs_even[i] = le;
        lhs_odd[i] = lo;
        lhs_sum[i] = le + lo;

        let re = rhs[2 * i];
        let ro = rhs[2 * i + 1];
        rhs_even[i] = re;
        rhs_odd[i] = ro;
        rhs_sum[i] = re + ro;
    }

    // Three sub-convolutions: even x even, odd x odd, sum x sum.
    let mut conv_even = [A::ZERO; HALF];
    let (left, right) = output.split_at_mut(HALF);

    inner_negacyclic(lhs_even, rhs_even, &mut conv_even);
    inner_negacyclic(lhs_odd, rhs_odd, left);
    inner_negacyclic(lhs_sum, rhs_sum, right);

    // Karatsuba recombination: extract the cross term and interleave.
    right[0] -= conv_even[0] + left[0];
    conv_even[0] -= left[HALF - 1];

    for i in 1..HALF {
        right[i] -= conv_even[i] + left[i];
        conv_even[i] += left[i - 1];
    }

    for i in 0..HALF {
        output[2 * i] = conv_even[i];
        output[2 * i + 1] = output[i + HALF];
    }
}

/// Cyclic convolution of size 3.
#[inline(always)]
fn cyclic_conv3<F: Field, A: Algebra<F> + Copy>(lhs: [A; 3], rhs: [F; 3], output: &mut [A]) {
    output[0] = A::mixed_dot_product(&lhs, &[rhs[0], rhs[2], rhs[1]]);
    output[1] = A::mixed_dot_product(&lhs, &[rhs[1], rhs[0], rhs[2]]);
    output[2] = A::mixed_dot_product(&lhs, &[rhs[2], rhs[1], rhs[0]]);
}

/// Negacyclic convolution of size 3.
#[inline(always)]
fn negacyclic_conv3<F: Field, A: Algebra<F> + Copy>(lhs: [A; 3], rhs: [F; 3], output: &mut [A]) {
    output[0] = A::mixed_dot_product(&lhs, &[rhs[0], -rhs[2], -rhs[1]]);
    output[1] = A::mixed_dot_product(&lhs, &[rhs[1], rhs[0], -rhs[2]]);
    output[2] = A::mixed_dot_product(&lhs, &[rhs[2], rhs[1], rhs[0]]);
}

/// Cyclic convolution of size 4 (hand-written CRT with field halving).
#[inline(always)]
fn cyclic_conv4<F: Field, A: Algebra<F> + Copy>(lhs: [A; 4], rhs: [F; 4], output: &mut [A]) {
    let lhs_pos = [lhs[0] + lhs[2], lhs[1] + lhs[3]];
    let lhs_neg = [lhs[0] - lhs[2], lhs[1] - lhs[3]];
    let rhs_pos = [rhs[0] + rhs[2], rhs[1] + rhs[3]];
    let rhs_neg = [rhs[0] - rhs[2], rhs[1] - rhs[3]];

    output[0] = A::mixed_dot_product(&lhs_neg, &[rhs_neg[0], -rhs_neg[1]]);
    output[1] = A::mixed_dot_product(&lhs_neg, &[rhs_neg[1], rhs_neg[0]]);
    output[2] = A::mixed_dot_product(&lhs_pos, &rhs_pos);
    output[3] = A::mixed_dot_product(&lhs_pos, &[rhs_pos[1], rhs_pos[0]]);

    output[0] += output[2];
    output[1] += output[3];

    output[0] = output[0].halve();
    output[1] = output[1].halve();

    output[2] -= output[0];
    output[3] -= output[1];
}

/// Negacyclic convolution of size 4.
#[inline(always)]
fn negacyclic_conv4<F: Field, A: Algebra<F> + Copy>(lhs: [A; 4], rhs: [F; 4], output: &mut [A]) {
    output[0] = A::mixed_dot_product(&lhs, &[rhs[0], -rhs[3], -rhs[2], -rhs[1]]);
    output[1] = A::mixed_dot_product(&lhs, &[rhs[1], rhs[0], -rhs[3], -rhs[2]]);
    output[2] = A::mixed_dot_product(&lhs, &[rhs[2], rhs[1], rhs[0], -rhs[3]]);
    output[3] = A::mixed_dot_product(&lhs, &[rhs[3], rhs[2], rhs[1], rhs[0]]);
}

// Size 6 = 2 x 3

#[inline(always)]
fn cyclic_conv6<F: Field, A: Algebra<F> + Copy>(lhs: [A; 6], rhs: [F; 6], output: &mut [A]) {
    cyclic_conv_recursive::<F, A, 6, 3>(
        lhs,
        rhs,
        output,
        cyclic_conv3::<F, A>,
        negacyclic_conv3::<F, A>,
    );
}

#[inline(always)]
fn negacyclic_conv6<F: Field, A: Algebra<F> + Copy>(lhs: [A; 6], rhs: [F; 6], output: &mut [A]) {
    negacyclic_conv_recursive::<F, A, 6, 3>(lhs, rhs, output, negacyclic_conv3::<F, A>);
}

// Size 8 = 2 x 4

#[inline(always)]
fn cyclic_conv8<F: Field, A: Algebra<F> + Copy>(lhs: [A; 8], rhs: [F; 8], output: &mut [A]) {
    cyclic_conv_recursive::<F, A, 8, 4>(
        lhs,
        rhs,
        output,
        cyclic_conv4::<F, A>,
        negacyclic_conv4::<F, A>,
    );
}

#[inline(always)]
fn negacyclic_conv8<F: Field, A: Algebra<F> + Copy>(lhs: [A; 8], rhs: [F; 8], output: &mut [A]) {
    negacyclic_conv_recursive::<F, A, 8, 4>(lhs, rhs, output, negacyclic_conv4::<F, A>);
}

// Size 12 = 2 x 6

#[inline(always)]
fn cyclic_conv12<F: Field, A: Algebra<F> + Copy>(lhs: [A; 12], rhs: [F; 12], output: &mut [A]) {
    cyclic_conv_recursive::<F, A, 12, 6>(
        lhs,
        rhs,
        output,
        cyclic_conv6::<F, A>,
        negacyclic_conv6::<F, A>,
    );
}

#[inline(always)]
fn negacyclic_conv12<F: Field, A: Algebra<F> + Copy>(
    lhs: [A; 12],
    rhs: [F; 12],
    output: &mut [A],
) {
    negacyclic_conv_recursive::<F, A, 12, 6>(lhs, rhs, output, negacyclic_conv6::<F, A>);
}

// Size 16 = 2 x 8

#[inline(always)]
fn cyclic_conv16<F: Field, A: Algebra<F> + Copy>(lhs: [A; 16], rhs: [F; 16], output: &mut [A]) {
    cyclic_conv_recursive::<F, A, 16, 8>(
        lhs,
        rhs,
        output,
        cyclic_conv8::<F, A>,
        negacyclic_conv8::<F, A>,
    );
}

// Size 24 = 2 x 12

#[inline(always)]
fn cyclic_conv24<F: Field, A: Algebra<F> + Copy>(lhs: [A; 24], rhs: [F; 24], output: &mut [A]) {
    cyclic_conv_recursive::<F, A, 24, 12>(
        lhs,
        rhs,
        output,
        cyclic_conv12::<F, A>,
        negacyclic_conv12::<F, A>,
    );
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;

    type F = BabyBear;

    /// Map an arbitrary `u32` into a field element.
    fn arb_f() -> impl Strategy<Value = F> {
        prop::num::u32::ANY.prop_map(F::from_u32)
    }

    /// Naive O(N^2) circulant multiply used as the reference oracle.
    ///
    /// For each output index `i`, computes the dot product of the
    /// cyclically shifted column with the state vector:
    ///   r[i] = sum_j col[(i - j) mod N] * state[j]
    fn naive_circulant<const N: usize>(col: [F; N], state: [F; N]) -> [F; N] {
        core::array::from_fn(|i| {
            let mut acc = F::ZERO;
            for j in 0..N {
                acc += col[(N + i - j) % N] * state[j];
            }
            acc
        })
    }

    /// Fixed circulant column for width-16 tests.
    /// Uses small distinct integers so the MDS property is easy to verify by inspection.
    fn col_16() -> [F; 16] {
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17].map(F::from_i64)
    }

    /// Fixed circulant column for width-24 tests.
    fn col_24() -> [F; 24] {
        [
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25,
        ]
        .map(F::from_i64)
    }

    proptest! {
        /// Karatsuba width-16 must match the naive circulant multiply
        /// for every random state vector.
        #[test]
        fn karatsuba_16_matches_naive(state in prop::array::uniform16(arb_f())) {
            let col = col_16();

            // Compute the expected result via naive O(N^2) circulant multiply.
            let expected = naive_circulant(col, state);

            // Compute the actual result via Karatsuba convolution.
            let mut actual = state;
            mds_circulant_karatsuba_16(&mut actual, &col);

            prop_assert_eq!(actual, expected);
        }

        /// Karatsuba width-24 must match the naive circulant multiply
        /// for every random state vector.
        #[test]
        fn karatsuba_24_matches_naive(state in prop::array::uniform24(arb_f())) {
            let col = col_24();

            // Compute the expected result via naive O(N^2) circulant multiply.
            let expected = naive_circulant(col, state);

            // Compute the actual result via Karatsuba convolution.
            let mut actual = state;
            mds_circulant_karatsuba_24(&mut actual, &col);

            prop_assert_eq!(actual, expected);
        }

        /// Karatsuba width-16 with a random circulant column.
        /// Tests that the algorithm is correct beyond a single fixed matrix.
        #[test]
        fn karatsuba_16_random_col(
            col in prop::array::uniform16(arb_f()),
            state in prop::array::uniform16(arb_f()),
        ) {
            let expected = naive_circulant(col, state);

            let mut actual = state;
            mds_circulant_karatsuba_16(&mut actual, &col);

            prop_assert_eq!(actual, expected);
        }

        /// Karatsuba width-24 with a random circulant column.
        /// Tests that the algorithm is correct beyond a single fixed matrix.
        #[test]
        fn karatsuba_24_random_col(
            col in prop::array::uniform24(arb_f()),
            state in prop::array::uniform24(arb_f()),
        ) {
            let expected = naive_circulant(col, state);

            let mut actual = state;
            mds_circulant_karatsuba_24(&mut actual, &col);

            prop_assert_eq!(actual, expected);
        }
    }
}
