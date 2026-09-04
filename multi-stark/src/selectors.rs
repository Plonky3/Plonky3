//! Closed-form multilinear extensions used by the multilinear AIR prover.

use alloc::vec::Vec;
use core::ops::{AddAssign, Sub};

use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::PolyView;
use p3_util::log2_strict_usize;
use thiserror::Error;

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

    /// The full selector values at a residual-cube row, combined with the
    /// bound-coordinate prefix, spread across `WIDTH` SIMD lanes — one lane
    /// per consecutive residual row starting at `row`.
    ///
    /// Same construction as [`Self::from_row_with_prefix`], but every
    /// per-row indicator is built lane-by-lane via
    /// [`PackedFieldExtension::from_ext_fn`] instead of once for a single row.
    fn from_ext_packed_row_with_prefix<F>(
        row: usize,
        height: usize,
        prefix: Self,
    ) -> BoundaryEvals<EF::ExtensionPacking>
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        BoundaryEvals::new(
            EF::ExtensionPacking::from_ext_fn(|lane| prefix.first * EF::from_bool(row + lane == 0)),
            EF::ExtensionPacking::from_ext_fn(|lane| {
                prefix.last * EF::from_bool(row + lane + 1 == height)
            }),
            EF::ExtensionPacking::from_ext_fn(|lane| {
                let suffix_last = EF::from_bool(row + lane + 1 == height);
                EF::from_bool(row + lane + 1 < height) + suffix_last * prefix.transition
            }),
        )
    }

    /// Packed `(value, per-step difference)` pair for folding one variable,
    /// with a bound-coordinate prefix — the SIMD-packed extension-field twin
    /// of [`Self::row_pair_with_prefix`].
    ///
    /// # Arguments
    ///
    /// - `row`: index of the first low-half row in this packed block.
    /// - `half`: distance from a low row to its matching high row.
    /// - `height`: number of rows in the residual cube.
    /// - `prefix`: partial products over the coordinates bound so far.
    pub(super) fn row_pair_with_prefix_packed<F>(
        row: usize,
        half: usize,
        height: usize,
        prefix: Self,
    ) -> (
        BoundaryEvals<EF::ExtensionPacking>,
        BoundaryEvals<EF::ExtensionPacking>,
    )
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        let boundary = Self::from_ext_packed_row_with_prefix::<F>(row, height, prefix);
        let hi_boundary = Self::from_ext_packed_row_with_prefix::<F>(row + half, height, prefix);
        (
            boundary,
            BoundaryEvals::new(
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

/// Reasons a declared periodic column cannot be laid out on a trace.
///
/// Each variant is a statement about the declaration and the trace height alone.
/// No witness data is involved.
#[derive(Copy, Clone, Debug, Error, PartialEq, Eq)]
pub enum PeriodicError {
    /// The advertised periodic column count disagrees with the period vectors supplied.
    #[error("periodic column count mismatch: declared {declared}, supplied {supplied}")]
    CountMismatch {
        /// Count the AIR advertises.
        declared: usize,
        /// Number of period vectors the AIR supplies.
        supplied: usize,
    },
    /// A period vector is empty, naming no values to repeat.
    #[error("periodic column {column} is empty")]
    Empty {
        /// Index of the column in declaration order.
        column: usize,
    },
    /// A period is not a power of two.
    ///
    /// Only such a period covers a whole block of row bits.
    #[error("periodic column {column} has period {period}, which is not a power of two")]
    PeriodNotPowerOfTwo {
        /// Index of the column in declaration order.
        column: usize,
        /// Period supplied for it.
        period: usize,
    },
    /// A period exceeds the trace height.
    ///
    /// The column then never completes one cycle inside the trace.
    #[error("periodic column {column} has period {period}, above the trace height 2^{log_height}")]
    PeriodAboveHeight {
        /// Index of the column in declaration order.
        column: usize,
        /// Period supplied for it.
        period: usize,
        /// Base-two logarithm of the trace height.
        log_height: usize,
    },
}

/// Row-bit count each periodic column depends on, once the declaration is known to fit the trace.
///
/// A period-`p` column repeats every `p` rows.
/// It therefore depends only on the low `log2(p)` bits of the row index.
///
/// A declaration fits a trace of `2^log_height` rows exactly when every period has the form:
///
/// ```text
///     p = 2^j     with     0 <= j <= log_height
/// ```
///
/// Prover and verifier both go through this.
/// The two sides therefore agree on which declarations are legal.
///
/// # Arguments
///
/// - `declared`: periodic column count the AIR advertises.
/// - `columns`: one period vector per column, in declaration order.
/// - `log_height`: base-two logarithm of the trace height.
///
/// # Returns
///
/// The exponent `j` of each column's period, in declaration order.
///
/// # Errors
///
/// - The advertised count disagrees with the number of period vectors.
/// - A period vector is empty.
/// - A period is not a power of two.
/// - A period exceeds the trace height.
pub(super) fn periodic_num_variables<F>(
    declared: usize,
    columns: &[Vec<F>],
    log_height: usize,
) -> Result<Vec<usize>, PeriodicError> {
    // The opening layout steps over the advertised count to reach the next AIR's columns.
    // A disagreement there misplaces every column laid out after this AIR.
    if declared != columns.len() {
        return Err(PeriodicError::CountMismatch {
            declared,
            supplied: columns.len(),
        });
    }

    columns
        .iter()
        .enumerate()
        .map(|(column, values)| {
            let period = values.len();

            // An empty vector names no values to repeat.
            if period == 0 {
                return Err(PeriodicError::Empty { column });
            }

            // A period that is not a power of two spans no whole block of row bits.
            if !period.is_power_of_two() {
                return Err(PeriodicError::PeriodNotPowerOfTwo { column, period });
            }

            // A period above the trace height would reach past the point's coordinates.
            let j = log2_strict_usize(period);
            if j > log_height {
                return Err(PeriodicError::PeriodAboveHeight {
                    column,
                    period,
                    log_height,
                });
            }

            Ok(j)
        })
        .collect()
}

/// Evaluate every periodic column's multilinear extension at the bound point, in closed form.
///
/// A period-`p` column depends only on the low `j = log2(p)` bits of the row index.
/// The hypercube layout is big-endian.
/// Coordinate `0` is therefore the most significant bit.
/// Those low bits are therefore the last `j` coordinates of the point:
///
/// ```text
///     periodic_col(r_0, ..., r_{k-1}) = MLE_v(r_{k-j}, ..., r_{k-1})
/// ```
///
/// Here `v` is the length-`p` period vector.
///
/// The prover folds the full-height column `col[i mod p]` to the same point.
/// The leading coordinates' equality factors sum to one, collapsing that fold onto `v`:
///
/// ```text
///     sum_{high} eq(r_0..r_{k-j-1}, high) = 1
/// ```
///
/// Both sides therefore land on the same value.
///
/// The verifier computes this unaided.
/// Periodic columns are public parameters of the AIR and are never committed.
///
/// # Arguments
///
/// - `declared`: periodic column count the AIR advertises.
/// - `columns`: one period vector per column, in declaration order.
/// - `point`: the bound sumcheck point, one coordinate per trace variable.
///
/// # Returns
///
/// One evaluation per periodic column, in declaration order.
///
/// # Errors
///
/// Returns an error when the declaration does not fit a trace of `2^point.len()` rows.
pub(super) fn periodic_evals_at<F, EF>(
    declared: usize,
    columns: &[Vec<F>],
    point: &[EF],
) -> Result<Vec<EF>, PeriodicError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    // One point coordinate per trace variable.
    // The point length is therefore the height exponent.
    let k = point.len();

    // Reject a declaration that does not fit before indexing the point with it.
    let num_variables = periodic_num_variables(declared, columns, k)?;

    Ok(columns
        .iter()
        .zip(num_variables)
        .map(|(values, j)| {
            // The last j coordinates carry the low-order row bits, in order.
            PolyView::new(values).eval_base(&Point::new(point[k - j..].to_vec()))
        })
        .collect())
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

    #[test]
    fn packed_prefix_row_pair_matches_scalar_lanes() {
        // Invariant: each lane of the prefix-aware packed pair equals the scalar pair at that
        // lane's row, and a block holding neither row 0 nor row `height - 1` carries the
        // constant pair `((0, 0, 1), (0, 0, 0))`.
        //
        // Fixture state:
        //   height = 64 residual rows.
        //   a prefix accumulator obeying its own `transition = 1 - last` invariant.
        type Packing = <EF as ExtensionField<F>>::ExtensionPacking;

        let height = 1usize << 6;
        let half = height / 2;
        let width = <F as Field>::Packing::WIDTH;
        let last = EF::from_u64(5);
        let prefix = BoundaryEvals::new(EF::from_u64(3), last, EF::ONE - last);
        let num_blocks = half / width;

        for block in 0..num_blocks {
            // First low-half row covered by this packed block.
            let row = block * width;

            let (packed_value, packed_diff) =
                BoundaryEvals::<EF>::row_pair_with_prefix_packed::<F>(row, half, height, prefix);

            for lane in 0..width {
                // Scalar pair at the single residual row this lane represents.
                let (scalar_value, scalar_diff) =
                    BoundaryEvals::row_pair_with_prefix(row + lane, half, height, prefix);

                assert_eq!(
                    <Packing as PackedFieldExtension<F, EF>>::extract(&packed_value.first, lane),
                    scalar_value.first
                );
                assert_eq!(
                    <Packing as PackedFieldExtension<F, EF>>::extract(&packed_value.last, lane),
                    scalar_value.last
                );
                assert_eq!(
                    <Packing as PackedFieldExtension<F, EF>>::extract(
                        &packed_value.transition,
                        lane
                    ),
                    scalar_value.transition
                );

                assert_eq!(
                    <Packing as PackedFieldExtension<F, EF>>::extract(&packed_diff.first, lane),
                    scalar_diff.first
                );
                assert_eq!(
                    <Packing as PackedFieldExtension<F, EF>>::extract(&packed_diff.last, lane),
                    scalar_diff.last
                );
                assert_eq!(
                    <Packing as PackedFieldExtension<F, EF>>::extract(
                        &packed_diff.transition,
                        lane
                    ),
                    scalar_diff.transition
                );

                // Row 0 lives in block 0; row `height - 1` is the high twin of the last block.
                if block != 0 && block != num_blocks - 1 {
                    assert_eq!(scalar_value.first, EF::ZERO);
                    assert_eq!(scalar_value.last, EF::ZERO);
                    assert_eq!(scalar_value.transition, EF::ONE);
                    assert_eq!(scalar_diff.first, EF::ZERO);
                    assert_eq!(scalar_diff.last, EF::ZERO);
                    assert_eq!(scalar_diff.transition, EF::ZERO);
                }
            }
        }
    }

    #[test]
    fn periodic_evals_match_the_materialized_column() {
        // Invariant: the closed form on the low coordinates equals the full column's own extension.
        //
        // Fixture state: a period-4 vector on a 3-variable trace.
        //
        //     period vector : [5, 6, 7, 8]
        //     materialized  : [5, 6, 7, 8, 5, 6, 7, 8]
        //                     → depends on coordinates 1 and 2, never on coordinate 0
        let column = [5u64, 6, 7, 8].map(F::from_u64).to_vec();
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x2C), 3);

        let closed =
            periodic_evals_at::<F, EF>(1, core::slice::from_ref(&column), point.as_slice())
                .expect("a period-4 vector fits a height-8 trace");

        // Materialize the column and evaluate it directly at the same point.
        let materialized = Poly::new((0..8).map(|i| column[i % 4]).collect::<Vec<_>>());
        assert_eq!(closed, vec![materialized.eval_base(&point)]);
    }

    #[test]
    fn periodic_evals_reject_a_count_mismatch() {
        // Mutation: advertise three columns while supplying one.
        //
        //     advertised : 3 -> the opening layout steps over 3 columns
        //     supplied   : 1 -> every column laid out after this AIR would shift
        let column = vec![F::ONE, F::TWO];
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(1), 2);

        assert_eq!(
            periodic_evals_at::<F, EF>(3, &[column], point.as_slice()),
            Err(PeriodicError::CountMismatch {
                declared: 3,
                supplied: 1
            })
        );
    }

    #[test]
    fn periodic_evals_reject_an_empty_period_vector() {
        // Mutation: supply a period vector with no values to repeat.
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(2), 2);

        assert_eq!(
            periodic_evals_at::<F, EF>(1, &[Vec::new()], point.as_slice()),
            Err(PeriodicError::Empty { column: 0 })
        );
    }

    #[test]
    fn periodic_evals_reject_a_non_power_of_two_period() {
        // Mutation: use period 3, which covers no whole block of row bits.
        let column = [1u64, 2, 3].map(F::from_u64).to_vec();
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(3), 3);

        assert_eq!(
            periodic_evals_at::<F, EF>(1, &[column], point.as_slice()),
            Err(PeriodicError::PeriodNotPowerOfTwo {
                column: 0,
                period: 3
            })
        );
    }

    #[test]
    fn periodic_evals_reject_a_period_above_the_trace_height() {
        // Mutation: declare a period-8 column on a height-4 trace.
        //
        //     point coordinates : 2
        //     coordinates needed: 3
        //                         → the low bits reach past the point, hence the rejection
        let column = [1u64, 2, 3, 4, 5, 6, 7, 8].map(F::from_u64).to_vec();
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(4), 2);

        assert_eq!(
            periodic_evals_at::<F, EF>(1, &[column], point.as_slice()),
            Err(PeriodicError::PeriodAboveHeight {
                column: 0,
                period: 8,
                log_height: 2
            })
        );
    }

    #[test]
    fn periodic_evals_handle_a_constant_column() {
        // Invariant: period 1 is a constant column and needs no coordinates at all.
        //
        //     period vector : [9]
        //     materialized  : [9, 9, 9, 9]
        //                     → value 9 at every point
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x3D), 2);
        let closed = periodic_evals_at::<F, EF>(1, &[vec![F::from_u64(9)]], point.as_slice())
            .expect("a constant column fits any trace height");

        assert_eq!(closed, vec![EF::from(F::from_u64(9))]);
    }
}
