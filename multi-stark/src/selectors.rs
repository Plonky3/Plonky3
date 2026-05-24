//! Closed-form multilinear extensions used by the multilinear AIR prover.
//!
//! # Overview
//!
//! - The verifier needs to evaluate selector and shift multilinears at the
//!   random sumcheck point.
//! - The prover never needs them at the cube as multilinears, only as
//!   per-row constants (`0` or `1`); the cube case is exposed for completeness.
//! - Every formula evaluates in `O(k)` field operations, where `k` is the
//!   number of binary trace variables.
//!
//! # Endianness
//!
//! - This crate follows the `p3_multilinear_util` convention.
//! - Coordinate `0` is the high-order bit and coordinate `k - 1` is the low-order bit.
//! - A polynomial of `k` variables stored in lexicographic order evaluates at
//!   integer index `i` to the hypercube point whose coordinate `j` is bit `k - 1 - j` of `i`.

use alloc::vec::Vec;

use p3_field::{Algebra, Field, PrimeCharacteristicRing};

/// Evaluate the multilinear equality indicator at the boolean encoding of an index.
///
/// # Arguments
///
/// - `rs`: challenge coordinates, one per binary trace variable.
/// - `idx`: integer in `0..2^k`, with `k = rs.len()`, encoded big-endian.
///
/// # Returns
///
/// The product `prod_j (rs[j] * b_j + (1 - rs[j]) * (1 - b_j))` where
/// `b_j` is bit `k - 1 - j` of `idx`.
///
/// # Use
///
/// - With `idx = 0`, this is the first-row selector.
/// - With `idx = m - 1`, this is the last-row selector.
/// - At boolean inputs it agrees with the discrete equality predicate.
///
/// # Panics
///
/// Panics in debug mode if `idx >= 2^k`.
///
/// # Examples
///
/// ```
/// use p3_baby_bear::BabyBear;
/// use p3_field::PrimeCharacteristicRing;
/// use p3_multi_stark::selectors::eq_eval;
///
/// // Big-endian: rs = (0, 1) encodes the integer 1.
/// let rs = [BabyBear::ZERO, BabyBear::ONE];
/// assert_eq!(eq_eval(&rs, 1), BabyBear::ONE);
/// assert_eq!(eq_eval(&rs, 0), BabyBear::ZERO);
/// ```
#[inline]
#[must_use]
pub fn eq_eval<EF: Field>(rs: &[EF], idx: usize) -> EF {
    let k = rs.len();
    debug_assert!(
        k == 0 || idx < (1usize << k),
        "eq_eval: idx ({idx}) must be < 2^k ({})",
        1usize << k,
    );

    // Initialize the running product at 1; the empty product evaluates to 1.
    let mut acc = EF::ONE;
    for (j, &r) in rs.iter().enumerate() {
        // Big-endian: coordinate j holds bit (k - 1 - j) of idx.
        let bit_is_one = (idx >> (k - 1 - j)) & 1 == 1;
        // Multiply by r when the bit is 1, by (1 - r) when the bit is 0.
        acc *= if bit_is_one { r } else { EF::ONE - r };
    }
    acc
}

/// First-row selector at a challenge point.
///
/// Equivalent to `eq_eval` against the integer `0`.
#[inline]
#[must_use]
pub fn is_first_row_eval<EF: Field>(rs: &[EF]) -> EF {
    eq_eval(rs, 0)
}

/// Last-row selector at a challenge point.
///
/// Equivalent to `eq_eval` against the integer `2^k - 1`.
#[inline]
#[must_use]
pub fn is_last_row_eval<EF: Field>(rs: &[EF]) -> EF {
    let k = rs.len();
    // Last hypercube index; an empty `rs` collapses to 0.
    let last = if k == 0 { 0 } else { (1usize << k) - 1 };
    eq_eval(rs, last)
}

/// Transition selector at a challenge point.
///
/// Equal to `1 - is_last_row_eval(rs)`.
#[inline]
#[must_use]
pub fn is_transition_eval<EF: Field>(rs: &[EF]) -> EF {
    EF::ONE - is_last_row_eval(rs)
}

/// First-row Lagrange basis polynomial at a challenge point.
///
/// # Why
///
/// - First-row public inputs are *spliced* into a committed column.
/// - The verifier recovers the original column evaluation as
///   `committed(r) + p * lagrange_zero(r)`.
/// - That makes this a basis vector, distinct in role from a selector.
#[inline]
#[must_use]
pub fn lagrange_zero<EF: Field>(rs: &[EF]) -> EF {
    is_first_row_eval(rs)
}

/// Last-row Lagrange basis polynomial at a challenge point.
///
/// Mirrors `lagrange_zero` for boundary values at the last row.
#[inline]
#[must_use]
pub fn lagrange_last<EF: Field>(rs: &[EF]) -> EF {
    is_last_row_eval(rs)
}

/// Multilinear extension of the "add one in binary" relation.
///
/// # Definition
///
/// On the boolean cube `{0, 1}^k x {0, 1}^k` the relation outputs `1`
/// exactly when the integer encoded by `ry` is the integer encoded by
/// `rx` plus one, and `0` otherwise. The relation does not wrap: the
/// last hypercube vertex has no successor.
///
/// # Arguments
///
/// - `rx`: challenge coordinates for the source row.
/// - `ry`: challenge coordinates for the target row.
///
/// # Returns
///
/// The unique multilinear extension of the relation evaluated at `(rx, ry)`.
///
/// # Algorithm
///
/// Adding one in binary toggles a run of trailing one-bits to zero and
/// the bit just above to one. The polynomial is the sum of one term
/// per possible carry depth:
///
/// ```text
///     depth 0:           x = ...?0   ->   y = ...?1
///     depth 1:           x = ...?01  ->   y = ...?10
///     depth d:           x = ...?0 11..1   ->   y = ...?1 00..0
///                                    \--d--/        \--d--/
/// ```
///
/// The depth-zero term costs `O(k)`; each higher depth extends a
/// running carry product by one more position so the total cost stays
/// `O(k)`.
///
/// # Performance
///
/// - One pre-pass builds prefix products of the high-bits equality factors.
/// - The depth loop reuses those prefix products in `O(1)` per depth.
/// - Total cost is `O(k)` field operations and one `Vec` of length `k`.
///
/// # Panics
///
/// Panics if the two argument slices have different lengths.
#[inline]
#[must_use]
pub fn next_eval<EF: Field>(rx: &[EF], ry: &[EF]) -> EF {
    assert_eq!(
        rx.len(),
        ry.len(),
        "next_eval: rx and ry must have the same length"
    );
    let k = rx.len();

    // Edge case: a single-row trace has no successor.
    if k == 0 {
        return EF::ZERO;
    }

    // Precompute prefix products of "high coordinates match".
    //
    //     eq_prefix[j] = product over i in 0..j of (rx[i]*ry[i] + (1-rx[i])*(1-ry[i]))
    //
    // The single-variable equality factor is degree 2 in (rx[i], ry[i]).
    let mut eq_prefix: Vec<EF> = Vec::with_capacity(k);
    let mut acc = EF::ONE;
    eq_prefix.push(acc);
    for j in 0..k - 1 {
        let term = rx[j] * ry[j] + (EF::ONE - rx[j]) * (EF::ONE - ry[j]);
        acc *= term;
        eq_prefix.push(acc);
    }

    // Depth-zero term:
    //
    //     (1 - rx[k-1]) * ry[k-1] * eq_prefix[k-1]
    //
    // It encodes "low bit of x is 0, low bit of y is 1, higher bits match".
    let mut total = (EF::ONE - rx[k - 1]) * ry[k - 1] * eq_prefix[k - 1];

    // Higher-depth terms: extend a running carry across the trailing bits.
    //
    //     carry_d = product over i in 0..d of rx[k-1-i] * (1 - ry[k-1-i])
    //
    // For each depth d we accumulate:
    //
    //     carry_d * flip * eq_prefix[k-1-d]
    //
    // where `flip` encodes the position just above the carry that flips
    // from 0 in x to 1 in y.
    let mut carry = EF::ONE;
    for d in 1..k {
        // Grow the carry by one more trailing position.
        carry *= rx[k - d] * (EF::ONE - ry[k - d]);
        // The "flip" position: 0 in x, 1 in y.
        let flip = (EF::ONE - rx[k - 1 - d]) * ry[k - 1 - d];
        // Bits above the flip position must match between x and y.
        let high_match = eq_prefix[k - 1 - d];
        total += carry * flip * high_match;
    }

    total
}

/// Build the dense next-row rotation of a length-`m` column.
///
/// # Arguments
///
/// - `column`: column values in row order, length must be a power of two.
///
/// # Returns
///
/// A new vector where entry `i` holds the original entry at row `i + 1`
/// and the last entry is zero. The convention matches `next_eval`: the
/// last row has no successor, so its shifted value is zero rather than
/// wrapping around.
///
/// # Performance
///
/// - `O(m)` field operations dominated by the memcpy of `m - 1` elements.
/// - One allocation of length `m`.
///
/// # Panics
///
/// Panics if the column length is not a power of two.
#[must_use]
pub fn shift_column<F: PrimeCharacteristicRing + Copy>(column: &[F]) -> Vec<F> {
    assert!(
        column.len().is_power_of_two(),
        "shift_column: column length ({}) must be a power of two",
        column.len(),
    );
    let m = column.len();
    // Allocate the output and pre-fill it with zeros.
    let mut out: Vec<F> = F::zero_vec(m);
    // Copy original entries 1..m into output positions 0..m-1; the last entry stays zero.
    if m > 1 {
        out[..m - 1].copy_from_slice(&column[1..]);
    }
    out
}

/// Multilinear equality indicator with base-field challenges.
///
/// # Arguments
///
/// - `rs`: base-field challenge coordinates.
/// - `idx`: target hypercube index.
///
/// # Returns
///
/// Same closed form as `eq_eval`, but accepts base-field challenges and
/// returns an extension-field result. Useful where part of the prover's
/// transcript has been narrowed back into the base field.
#[inline]
#[must_use]
pub fn eq_eval_base<F, EF>(rs: &[F], idx: usize) -> EF
where
    F: PrimeCharacteristicRing + Copy,
    EF: Algebra<F> + Copy + PrimeCharacteristicRing,
{
    let k = rs.len();
    debug_assert!(k == 0 || idx < (1usize << k));
    // Initialize the running product at 1.
    let mut acc = EF::ONE;
    for (j, &r) in rs.iter().enumerate() {
        // Same big-endian bit selection as the extension-field variant.
        let bit = (idx >> (k - 1 - j)) & 1 == 1;
        // Lift the base challenge into the extension before multiplying.
        let r_ef: EF = r.into();
        acc *= if bit { r_ef } else { EF::ONE - r_ef };
    }
    acc
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_multilinear_util::point::Point;
    use p3_multilinear_util::poly::Poly;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn eq_eval_reference(rs: &[EF], idx: usize) -> EF {
        // Reference: build the full eq table and read off entry `idx`.
        if rs.is_empty() {
            return EF::ONE;
        }
        Poly::new_from_point(rs, EF::ONE).as_slice()[idx]
    }

    fn next_eval_reference(rx: &[EF], ry: &[EF]) -> EF {
        // Reference: materialize the explicit indicator on the cube and evaluate it.
        assert_eq!(rx.len(), ry.len());
        let k = rx.len();
        if k == 0 {
            return EF::ZERO;
        }
        let m = 1usize << k;
        // Mark every successor pair with a 1, leaving the rest as zero.
        let mut table = EF::zero_vec(m * m);
        for x in 0..m - 1 {
            let y = x + 1;
            table[x * m + y] = EF::ONE;
        }
        // Concatenate rx and ry; rx is the high half, ry is the low half.
        let mut full = Vec::with_capacity(2 * k);
        full.extend_from_slice(rx);
        full.extend_from_slice(ry);
        Poly::new(table).eval_base(&Point::new(full))
    }

    fn shift_reference(column: &[F]) -> Vec<F> {
        // Reference: write the rotated column entry-by-entry with a zero at the end.
        let m = column.len();
        let mut out = F::zero_vec(m);
        for i in 0..m - 1 {
            out[i] = column[i + 1];
        }
        out
    }

    #[test]
    fn eq_eval_zero_vars_is_one() {
        // Invariant: the empty product evaluates to ONE.
        assert_eq!(eq_eval::<EF>(&[], 0), EF::ONE);
    }

    #[test]
    fn eq_eval_at_boolean_inputs_is_indicator() {
        // Invariant: at hypercube inputs the polynomial returns the discrete indicator.
        //
        //     for every k in 0..=4 and every idx in 0..2^k:
        //         eq_eval(hypercube(idx), query) == 1 iff idx == query
        for k in 0..=4 {
            for idx in 0..(1usize << k) {
                let rs = Point::<EF>::hypercube(idx, k);
                for query in 0..(1usize << k) {
                    let v = eq_eval(rs.as_slice(), query);
                    let expected = if idx == query { EF::ONE } else { EF::ZERO };
                    assert_eq!(v, expected, "k={k} idx={idx} query={query}");
                }
            }
        }
    }

    proptest! {
        #[test]
        fn eq_eval_matches_reference(k in 0usize..6, seed: u64, idx_seed: u64) {
            // Build random challenges in EF and a random target hypercube index.
            let mut rng = SmallRng::seed_from_u64(seed);
            let rs: Vec<EF> = (0..k).map(|_| rng.random()).collect();
            let idx = if k == 0 { 0 } else { idx_seed as usize % (1usize << k) };

            // Closed-form result must agree with the dense reference.
            let got = eq_eval(&rs, idx);
            let want = eq_eval_reference(&rs, idx);
            prop_assert_eq!(got, want);
        }
    }

    #[test]
    fn first_last_transition_at_corners() {
        // Fixture state: k = 3, so the cube has 8 vertices.
        //
        // Invariant per vertex:
        //
        //     idx == 0     -> first row selector returns 1
        //     idx == 2^k-1 -> last row selector returns 1
        //     idx != 2^k-1 -> transition selector returns 1
        let k = 3usize;
        let last = (1usize << k) - 1;
        for idx in 0..(1usize << k) {
            let rs = Point::<EF>::hypercube(idx, k);
            let rs = rs.as_slice();
            // First-row selector must vanish off the first row.
            assert_eq!(
                is_first_row_eval(rs),
                if idx == 0 { EF::ONE } else { EF::ZERO },
                "first idx={idx}"
            );
            // Last-row selector must vanish off the last row.
            assert_eq!(
                is_last_row_eval(rs),
                if idx == last { EF::ONE } else { EF::ZERO },
                "last idx={idx}"
            );
            // Transition selector must vanish exactly on the last row.
            assert_eq!(
                is_transition_eval(rs),
                if idx == last { EF::ZERO } else { EF::ONE },
                "transition idx={idx}"
            );
            // The Lagrange aliases must be bit-identical to the selector functions.
            assert_eq!(lagrange_zero(rs), is_first_row_eval(rs));
            assert_eq!(lagrange_last(rs), is_last_row_eval(rs));
        }
    }

    #[test]
    fn next_eval_zero_vars_is_zero() {
        // Invariant: a single-row trace has no successor, so the relation is 0 everywhere.
        assert_eq!(next_eval::<EF>(&[], &[]), EF::ZERO);
    }

    #[test]
    fn next_eval_on_corners_matches_definition() {
        // Fixture state: k = 2, so the cube has 4 vertices.
        //
        // Invariant: next(bin(x), bin(y)) == 1 iff y == x + 1, else 0.
        //
        //     valid successor pairs: (0, 1), (1, 2), (2, 3)
        //     no wrap from (3, *)
        let k = 2usize;
        let m = 1usize << k;
        for x in 0..m {
            for y in 0..m {
                let rx = Point::<EF>::hypercube(x, k);
                let ry = Point::<EF>::hypercube(y, k);
                let got = next_eval(rx.as_slice(), ry.as_slice());
                let want = if y == x + 1 && x + 1 < m {
                    EF::ONE
                } else {
                    EF::ZERO
                };
                assert_eq!(got, want, "next(bin({x}), bin({y}))");
            }
        }
    }

    proptest! {
        #[test]
        fn next_eval_matches_reference(k in 0usize..4, seed_x: u64, seed_y: u64) {
            // Build independent random EF points for each side of the relation.
            let mut rng_x = SmallRng::seed_from_u64(seed_x);
            let mut rng_y = SmallRng::seed_from_u64(seed_y);
            let rx: Vec<EF> = (0..k).map(|_| rng_x.random()).collect();
            let ry: Vec<EF> = (0..k).map(|_| rng_y.random()).collect();

            // Closed-form result must match the naive dense-table reference.
            let got = next_eval(&rx, &ry);
            let want = next_eval_reference(&rx, &ry);
            prop_assert_eq!(got, want);
        }
    }

    #[test]
    fn shift_column_rotates_and_zero_fills() {
        // Fixture state: column = [1, 2, 3, 4, 5, 6, 7, 8].
        //
        // Invariant: rotate by one to the left, zero-fill the tail.
        //
        //     input : [1, 2, 3, 4, 5, 6, 7, 8]
        //     output: [2, 3, 4, 5, 6, 7, 8, 0]
        let col: Vec<F> = (1u64..=8).map(F::from_u64).collect();
        let want = shift_reference(&col);
        let got = shift_column(&col);
        assert_eq!(got, want);
        // Spot check: tail is zero, head is the original second entry.
        assert_eq!(got.last().copied().unwrap(), F::ZERO);
        assert_eq!(got[0], col[1]);
    }

    #[test]
    fn shift_column_single_row_is_zero() {
        // Invariant: a one-row trace has no successor, so its rotation is zero.
        let got = shift_column(&[F::from_u64(7)]);
        assert_eq!(got, vec![F::ZERO]);
    }

    #[test]
    #[should_panic]
    fn shift_column_panics_on_non_power_of_two() {
        // Mutation: pass a 3-element column.
        //
        //     length 3 is not a power of two -> panic
        let _ = shift_column(&[F::ZERO, F::ZERO, F::ZERO]);
    }

    proptest! {
        #[test]
        fn shift_column_matches_reference(log_m in 0usize..=6, seed: u64) {
            // Build a random base-field column of length 2^log_m.
            let mut rng = SmallRng::seed_from_u64(seed);
            let m = 1usize << log_m;
            let col: Vec<F> = (0..m).map(|_| rng.random()).collect();
            // The fast path must agree with the explicit element-by-element rotation.
            prop_assert_eq!(shift_column(&col), shift_reference(&col));
        }
    }
}
