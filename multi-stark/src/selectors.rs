//! Closed-form multilinear extensions used by the multilinear AIR prover.

use core::ops::{AddAssign, Sub};

use p3_field::{Field, PackedValue};

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
    pub(super) const fn new(first: EF, last: EF, transition: EF) -> Self {
        Self {
            first,
            last,
            transition,
        }
    }
}

impl<Packed> BoundaryEvals<Packed> {
    pub(super) fn from_packed_row<F>(row: usize, height: usize) -> Self
    where
        F: Field<Packing = Packed>,
        Packed: PackedValue<Value = F>,
    {
        Self::new(
            Packed::from_fn(|lane| F::from_bool(row + lane == 0)),
            Packed::from_fn(|lane| F::from_bool(row + lane + 1 == height)),
            Packed::from_fn(|lane| F::from_bool(row + lane + 1 < height)),
        )
    }

    pub(super) fn row_pair_packed<F>(row: usize, half: usize, height: usize) -> (Self, Self)
    where
        F: Field<Packing = Packed>,
        Packed: PackedValue<Value = F> + Sub<Output = Packed> + Copy,
    {
        let boundary = Self::from_packed_row::<F>(row, height);
        let hi_boundary = Self::from_packed_row::<F>(row + half, height);
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
    pub(super) fn at(rs: &[EF]) -> Self {
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

    pub(super) fn apply(&mut self, r: EF) {
        self.first *= EF::ONE - r;
        self.last *= r;
        self.transition = EF::ONE - self.last;
    }

    pub(super) fn from_row(row: usize, height: usize) -> Self {
        Self::new(
            EF::from_bool(row == 0),
            EF::from_bool(row + 1 == height),
            EF::from_bool(row + 1 < height),
        )
    }

    pub(super) fn from_row_with_prefix(row: usize, height: usize, prefix: Self) -> Self {
        let suffix = Self::from_row(row, height);
        Self::new(
            prefix.first * suffix.first,
            prefix.last * suffix.last,
            suffix.transition + suffix.last * prefix.transition,
        )
    }

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
    fn add_assign(&mut self, rhs: Self) {
        self.first += rhs.first;
        self.last += rhs.last;
        self.transition += rhs.transition;
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_multilinear_util::point::Point;

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
}
