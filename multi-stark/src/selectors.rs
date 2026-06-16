//! Closed-form multilinear extensions used by the multilinear AIR prover.

use p3_field::Field;

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
