//! Closed-form multilinear extensions used by the multilinear AIR prover.

use alloc::vec::Vec;

use p3_field::{Field, PrimeCharacteristicRing};

/// Boundary selectors evaluated at a sumcheck challenge.
#[derive(Copy, Clone, Debug)]
pub struct BoundaryEvals<EF> {
    /// First-row selector: `1` at row `0`, `0` elsewhere.
    pub first: EF,
    /// Last-row selector: `1` at row `m - 1`, `0` elsewhere.
    pub last: EF,
    /// Transition selector: `0` at row `m - 1`, `1` elsewhere.
    pub transition: EF,
}

impl<EF: Field> BoundaryEvals<EF> {
    /// Evaluate all three boundary selectors at the same challenge point.
    ///
    /// # Arguments
    ///
    /// - `rs`: challenge coordinates, one per binary trace variable.
    ///
    /// # Performance
    ///
    /// One pass over `rs` accumulates `first` and `last` in lockstep.
    /// `transition` falls out as a single subtraction.
    pub fn at(rs: &[EF]) -> Self {
        // Thread both running products through the same loop:
        //
        //     first := first * (1 - r)
        //     last  := last  * r
        let mut first = EF::ONE;
        let mut last = EF::ONE;
        for &r in rs {
            first *= EF::ONE - r;
            last *= r;
        }
        Self {
            first,
            last,
            transition: EF::ONE - last,
        }
    }
}

/// Multilinear extension of the "add one in binary" relation on pairs of `k`-bit boolean points.
///
/// The relation is `1` when `to_int(ry) = to_int(rx) + 1` and `0` otherwise.
/// The last hypercube vertex has no successor: the relation does not wrap.
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
/// Adding one in binary flips a run of trailing one-bits to zero and the bit just above to one.
/// The polynomial sums one term per *carry depth* `d` in `0..k`:
///
/// - The trailing `d` bits of `rx` are all `1`; in `ry` they are all `0`.
/// - The bit at offset `d` from the low end flips from `0` in `rx` to `1` in `ry`.
/// - Every position above that flip matches between `rx` and `ry`.
///
/// Prefix products of "high bits match" are precomputed once up front, so each depth costs `O(1)`.
///
/// # Performance
///
/// `O(k)` field operations and one `Vec` of length `k`.
///
/// # Panics
///
/// Panics if `rx` and `ry` have different lengths.
#[inline]
#[must_use]
pub fn next_eval<EF: Field>(rx: &[EF], ry: &[EF]) -> EF {
    assert_eq!(
        rx.len(),
        ry.len(),
        "next_eval: rx and ry must have the same length"
    );
    let k = rx.len();

    // A single-row trace has no successor; the relation vanishes everywhere.
    if k == 0 {
        return EF::ZERO;
    }

    // Prefix products of "high coordinates match" up to index j:
    //
    //     eq_prefix[j] = prod_{i < j} (rx[i]*ry[i] + (1-rx[i])*(1-ry[i]))
    let mut eq_prefix: Vec<EF> = Vec::with_capacity(k);
    let mut acc = EF::ONE;
    eq_prefix.push(acc);
    for j in 0..k - 1 {
        let term = rx[j] * ry[j] + (EF::ONE - rx[j]) * (EF::ONE - ry[j]);
        acc *= term;
        eq_prefix.push(acc);
    }

    // Depth-zero term: low bit of rx is 0, low bit of ry is 1, higher bits match.
    //
    //     (1 - rx[k-1]) * ry[k-1] * eq_prefix[k-1]
    let mut total = (EF::ONE - rx[k - 1]) * ry[k - 1] * eq_prefix[k - 1];

    // Higher-depth terms: each adds `carry_d * flip * eq_prefix[k-1-d]`.
    //
    //     carry_d = prod_{i < d} rx[k-1-i] * (1 - ry[k-1-i])   // carry chain
    //     flip    = (1 - rx[k-1-d]) * ry[k-1-d]                // bit that flips
    //     eq_prefix[k-1-d]                                     // high bits match
    let mut carry = EF::ONE;
    for d in 1..k {
        carry *= rx[k - d] * (EF::ONE - ry[k - d]);
        let flip = (EF::ONE - rx[k - 1 - d]) * ry[k - 1 - d];
        let high_match = eq_prefix[k - 1 - d];
        total += carry * flip * high_match;
    }

    total
}

/// Dense next-row rotation of a length-`m` column.
///
/// # Arguments
///
/// - `column`: column values in row order; length must be a power of two.
///
/// # Returns
///
/// A length-`m` vector where:
///
/// - entry `i` (for `i < m - 1`) holds `column[i + 1]`,
/// - the last entry is zero — the last row has no successor (no wrap).
///
/// # Performance
///
/// - `O(m)` field operations dominated by a `memcpy` of `m - 1` elements.
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
    // Pre-fill the output with zeros; the tail entry will stay zero.
    let mut out: Vec<F> = F::zero_vec(m);
    // Copy entries 1..m into output positions 0..m-1.
    if m > 1 {
        out[..m - 1].copy_from_slice(&column[1..]);
    }
    out
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
    fn boundary_evals_at_corners() {
        // Fixture state: k = 3, so the cube has 8 vertices.
        //
        // Invariant per vertex:
        //
        //     idx == 0     -> first      = 1, others 0
        //     idx == 2^k-1 -> last       = 1, others 0
        //     idx != 2^k-1 -> transition = 1
        let k = 3usize;
        let last_idx = (1usize << k) - 1;
        for idx in 0..(1usize << k) {
            let rs = Point::<EF>::hypercube(idx, k);
            let evals = BoundaryEvals::at(rs.as_slice());
            assert_eq!(
                evals.first,
                if idx == 0 { EF::ONE } else { EF::ZERO },
                "first idx={idx}"
            );
            assert_eq!(
                evals.last,
                if idx == last_idx { EF::ONE } else { EF::ZERO },
                "last idx={idx}"
            );
            assert_eq!(
                evals.transition,
                if idx == last_idx { EF::ZERO } else { EF::ONE },
                "transition idx={idx}"
            );
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
