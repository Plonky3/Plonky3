//! Closed-form multilinear extensions used by the multilinear AIR prover.

use core::ops::{AddAssign, Sub};

use p3_field::{Field, PackedValue, PrimeCharacteristicRing};

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

impl<EF> BoundaryEvals<EF> {
    /// Bundle three already-computed selector values.
    pub(super) const fn new(first: EF, last: EF, transition: EF) -> Self {
        Self {
            first,
            last,
            transition,
        }
    }
}

impl<Packed> BoundaryEvals<Packed>
where
    Packed: PackedValue,
    Packed::Value: PrimeCharacteristicRing,
{
    /// Build the three selectors for a packed block of consecutive rows, one lane per row.
    ///
    /// Lane `lane` carries the selector value at global row `row + lane`:
    /// ```text
    ///     first      = 1 iff row + lane == 0
    ///     last       = 1 iff row + lane == height - 1
    ///     transition = 1 iff row + lane <  height - 1
    /// ```
    ///
    /// # Arguments
    ///
    /// - `row`: index of the first row in this packed block.
    /// - `height`: total number of trace rows.
    pub(super) fn from_packed_row(row: usize, height: usize) -> Self {
        Self::new(
            Packed::from_fn(|lane| Packed::Value::from_bool(row + lane == 0)),
            Packed::from_fn(|lane| Packed::Value::from_bool(row + lane + 1 == height)),
            Packed::from_fn(|lane| Packed::Value::from_bool(row + lane + 1 < height)),
        )
    }

    /// Packed `(value, per-step difference)` pair for folding one variable.
    ///
    /// The active variable splits the residual cube into a low half and a high half.
    /// The first element is the selector block at the low rows.
    /// The second is the lane-wise difference to the matching high rows:
    /// ```text
    ///     value(t) = low + t * (high - low)
    /// ```
    /// Adding the difference once advances each lane from the low row to the high row.
    ///
    /// # Arguments
    ///
    /// - `row`: index of the first low-half row in this packed block.
    /// - `half`: distance from a low row to its matching high row.
    /// - `height`: total number of trace rows.
    pub(super) fn row_pair_packed(row: usize, half: usize, height: usize) -> (Self, Self)
    where
        Packed: Sub<Output = Packed>,
    {
        // Selector block at the low rows row .. row + WIDTH.
        let boundary = Self::from_packed_row(row, height);
        // Selector block at the matching high rows row + half .. row + half + WIDTH.
        let hi_boundary = Self::from_packed_row(row + half, height);
        (
            boundary,
            Self::new(
                hi_boundary.first - boundary.first,
                hi_boundary.last - boundary.last,
                hi_boundary.transition - boundary.transition,
            ),
        )
    }
}

impl<EF: Field> BoundaryEvals<EF> {
    /// Evaluate all three boundary selectors at the same challenge point.
    ///
    /// # Arguments
    ///
    /// - `rs`: challenge coordinates, one per binary trace variable.
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

    /// Fold one more bound coordinate into the running prefix accumulator.
    ///
    /// The accumulator tracks the partial products over the coordinates bound so far:
    /// ```text
    ///     first = prod_j (1 - r_j)
    ///     last  = prod_j r_j
    /// ```
    ///
    /// The transition value is kept as the dependent quantity `1 - last`.
    /// This invariant is what makes the prefix-aware row evaluation exact.
    ///
    /// # Arguments
    ///
    /// - `r`: the challenge that binds the next coordinate.
    pub(super) fn apply(&mut self, r: EF) {
        self.first *= EF::ONE - r;
        self.last *= r;
        self.transition = EF::ONE - self.last;
    }

    /// The three selector values at a single residual-cube row, with no prefix applied.
    ///
    /// Each value is the indicator of that row over the residual cube:
    /// ```text
    ///     first      = 1 iff row == 0
    ///     last       = 1 iff row == height - 1
    ///     transition = 1 iff row <  height - 1
    /// ```
    ///
    /// # Arguments
    ///
    /// - `row`: index into the residual cube.
    /// - `height`: number of rows in the residual cube.
    pub(super) fn from_row(row: usize, height: usize) -> Self {
        Self::new(
            EF::from_bool(row == 0),
            EF::from_bool(row + 1 == height),
            EF::from_bool(row + 1 < height),
        )
    }

    /// The full selector values at a residual-cube row, combined with the bound-coordinate prefix.
    ///
    /// The first-row and last-row selectors factor across bound and residual coordinates:
    /// ```text
    ///     first = prefix.first * [row == 0]
    ///     last  = prefix.last  * [row == height - 1]
    /// ```
    ///
    /// The transition selector is `1 - last` over all coordinates:
    /// ```text
    ///     transition = 1 - prefix.last * [row == height - 1]
    /// ```
    ///
    /// The implementation form reaches the same value through the residual indicators:
    /// ```text
    ///     transition = [row < height - 1] + [row == height - 1] * prefix.transition
    /// ```
    /// The two forms agree exactly because the prefix maintains `transition = 1 - last`.
    ///
    /// # Arguments
    ///
    /// - `row`: index into the residual cube.
    /// - `height`: number of rows in the residual cube.
    /// - `prefix`: partial products over the coordinates bound so far.
    pub(super) fn from_row_with_prefix(row: usize, height: usize, prefix: Self) -> Self {
        let suffix = Self::from_row(row, height);
        Self::new(
            prefix.first * suffix.first,
            prefix.last * suffix.last,
            suffix.transition + suffix.last * prefix.transition,
        )
    }

    /// `(value, per-step difference)` pair for folding one variable, with no prefix.
    ///
    /// - The first element is the selector at the low-half row.
    /// - The second is the difference to the matching high-half row:
    /// ```text
    ///     value(t) = low + t * (high - low)
    /// ```
    ///
    /// # Arguments
    ///
    /// - `row`: index of the low-half row in the residual cube.
    /// - `half`: distance from a low row to its matching high row.
    /// - `height`: number of rows in the residual cube.
    pub(super) fn row_pair(row: usize, half: usize, height: usize) -> (Self, Self) {
        let boundary = Self::from_row(row, height);
        let hi_boundary = Self::from_row(row + half, height);
        (
            boundary,
            Self::new(
                hi_boundary.first - boundary.first,
                hi_boundary.last - boundary.last,
                hi_boundary.transition - boundary.transition,
            ),
        )
    }

    /// `(value, per-step difference)` pair for folding one variable, with a bound-coordinate prefix.
    ///
    /// Same interpolation shape as the prefix-free pair, with the prefix folded in:
    /// ```text
    ///     value(t) = low + t * (high - low)
    /// ```
    ///
    /// # Arguments
    ///
    /// - `row`: index of the low-half row in the residual cube.
    /// - `half`: distance from a low row to its matching high row.
    /// - `height`: number of rows in the residual cube.
    /// - `prefix`: partial products over the coordinates bound so far.
    pub(super) fn row_pair_with_prefix(
        row: usize,
        half: usize,
        height: usize,
        prefix: Self,
    ) -> (Self, Self) {
        let boundary = Self::from_row_with_prefix(row, height, prefix);
        let hi_boundary = Self::from_row_with_prefix(row + half, height, prefix);
        (
            boundary,
            Self::new(
                hi_boundary.first - boundary.first,
                hi_boundary.last - boundary.last,
                hi_boundary.transition - boundary.transition,
            ),
        )
    }
}

impl<EF> AddAssign for BoundaryEvals<EF>
where
    EF: AddAssign,
{
    /// Advance all three selectors by one interpolation step.
    ///
    /// Adding the per-step difference moves each selector to the next integer node.
    fn add_assign(&mut self, rhs: Self) {
        self.first += rhs.first;
        self.last += rhs.last;
        self.transition += rhs.transition;
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
    use p3_multilinear_util::point::Point;
    use p3_multilinear_util::poly::Poly;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

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
    fn virtual_selectors_match_materialized_fold() {
        // Invariant: the closed-form selectors equal the materialized indicator columns folded the same way.
        //
        // Fixture state:
        //   cube of k variables, height = 2^k.
        //   interpolation nodes {0, 1, 2, 3}.
        //
        // The nodes mirror how the prover steps a round polynomial by adding the per-step difference.
        let mut rng = SmallRng::seed_from_u64(0x5E1EC7);
        let nodes = [EF::ZERO, EF::ONE, EF::from_u64(2), EF::from_u64(3)];

        for k in 1..=6usize {
            let height = 1usize << k;

            // Materialized indicator columns over the full cube:
            //
            //     first      = [1, 0, ..., 0]
            //     last       = [0, ..., 0, 1]
            //     transition = [1, ..., 1, 0]
            let mut first = vec![EF::ZERO; height];
            first[0] = EF::ONE;
            let mut last = vec![EF::ZERO; height];
            last[height - 1] = EF::ONE;
            let mut transition = vec![EF::ONE; height];
            transition[height - 1] = EF::ZERO;
            let mut first = Poly::new(first);
            let mut last = Poly::new(last);
            let mut transition = Poly::new(transition);

            // Round 0 binds no coordinate yet; later rounds carry a prefix accumulator.
            let mut prefix: Option<BoundaryEvals<EF>> = None;
            let mut num_evals = height;

            while num_evals > 1 {
                // The active variable splits the residual cube into matching halves.
                let half = num_evals / 2;

                for s in 0..half {
                    // Closed-form value at the low row, plus the step to the high row.
                    // Round 0 has no prefix; later rounds fold the accumulator in.
                    let (mut value, diff) = prefix.map_or_else(
                        || BoundaryEvals::row_pair(s, half, num_evals),
                        |p| BoundaryEvals::row_pair_with_prefix(s, half, num_evals, p),
                    );

                    // value(t) = low + t * (high - low), checked against the folded columns.
                    for &t in &nodes {
                        assert_eq!(value.first, first.fix_prefix_var_at(t, s));
                        assert_eq!(value.last, last.fix_prefix_var_at(t, s));
                        assert_eq!(value.transition, transition.fix_prefix_var_at(t, s));
                        value += diff;
                    }
                }

                // Bind this round with a random challenge on both representations.
                let r: EF = rng.random();
                first.fix_prefix_var_mut(r);
                last.fix_prefix_var_mut(r);
                transition.fix_prefix_var_mut(r);

                // First fold seeds the accumulator; later folds extend it.
                prefix = Some(prefix.map_or_else(
                    || BoundaryEvals::new(EF::ONE - r, r, EF::ONE - r),
                    |mut p| {
                        p.apply(r);
                        p
                    },
                ));

                num_evals = half;
            }
        }
    }

    #[test]
    fn packed_row_pair_matches_scalar_lanes() {
        // Invariant: each lane of the packed selector pair equals the scalar pair at that lane's row.
        //
        // Fixture state:
        //   height = 64 rows.
        //   one packed block per WIDTH consecutive low-half rows.
        let height = 1usize << 6;
        let half = height / 2;
        let width = <F as Field>::Packing::WIDTH;

        for block in 0..(half / width) {
            // First low-half row covered by this packed block.
            let row = block * width;

            // Packed pair over WIDTH lanes at once.
            let (packed_value, packed_diff) =
                BoundaryEvals::<<F as Field>::Packing>::row_pair_packed(row, half, height);

            for lane in 0..width {
                // Scalar pair at the single global row this lane represents.
                let (scalar_value, scalar_diff) =
                    BoundaryEvals::<F>::row_pair(row + lane, half, height);

                assert_eq!(packed_value.first.as_slice()[lane], scalar_value.first);
                assert_eq!(packed_value.last.as_slice()[lane], scalar_value.last);
                assert_eq!(
                    packed_value.transition.as_slice()[lane],
                    scalar_value.transition
                );

                assert_eq!(packed_diff.first.as_slice()[lane], scalar_diff.first);
                assert_eq!(packed_diff.last.as_slice()[lane], scalar_diff.last);
                assert_eq!(
                    packed_diff.transition.as_slice()[lane],
                    scalar_diff.transition
                );
            }
        }
    }
}
