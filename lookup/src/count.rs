//! Signed multiplicity carrying a static bound on its per-row magnitude.

use core::ops::Neg;

use p3_field::PrimeCharacteristicRing;

/// A signed bus multiplicity paired with a per-row bound on its magnitude.
///
/// # Overview
///
/// - The expression is the signed count one row contributes to a bus.
/// - The bound is a static upper limit on that count's absolute value, per row.
/// - The bound feeds the LogUp height check `sum_i weight_i * height_i < p`.
///
/// # Soundness
///
/// - The bound holds only if the AIR constrains the count's magnitude to respect it on every row.
/// - The height check trusts that constraint; it never reads committed values to confirm it.
/// - A bound below the true per-row count lets a provided multiplicity wrap modulo `p`.
///
/// # Construction
///
/// - A signed integer constant fixes the bound to its absolute value.
/// - A variable count must state its bound explicitly.
#[derive(Clone, Debug)]
pub struct Count<E> {
    /// Signed multiplicity expression: positive sends, negative receives.
    expr: E,
    /// Per-row upper bound on the absolute value of the expression.
    weight: u32,
}

impl<E> Count<E> {
    /// Pair a variable count with an explicit per-row magnitude bound.
    ///
    /// # Arguments
    ///
    /// - `expr` — the signed multiplicity, typically a trace expression.
    /// - `weight` — a per-row upper bound on the count's magnitude, enforced by the AIR.
    pub const fn bounded(expr: E, weight: u32) -> Self {
        Self { expr, weight }
    }

    /// A provided table entry, excluded from the query height check.
    ///
    /// - The provided side supplies values rather than querying them.
    /// - Its bound is zero, so it never adds to the query-multiplicity sum.
    pub const fn provided(expr: E) -> Self {
        Self { expr, weight: 0 }
    }

    /// Per-row upper bound on the absolute value of the count.
    pub const fn weight(&self) -> u32 {
        self.weight
    }

    /// Split into the signed expression and its per-row bound.
    pub fn into_parts(self) -> (E, u32) {
        (self.expr, self.weight)
    }
}

impl<E: PrimeCharacteristicRing> From<i32> for Count<E> {
    fn from(count: i32) -> Self {
        // A compile-time constant is its own tightest per-row bound.
        // The sign only selects send versus receive, so the bound drops it.
        Self {
            expr: E::from_i32(count),
            weight: count.unsigned_abs(),
        }
    }
}

impl<E: Neg<Output = E>> Neg for Count<E> {
    type Output = Self;

    fn neg(self) -> Self {
        // Swapping send and receive flips the sign of the expression.
        // The magnitude is unchanged, so the bound carries over untouched.
        Self {
            expr: -self.expr,
            weight: self.weight,
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    #[test]
    fn constant_count_fixes_its_own_bound() {
        // A compile-time constant is its own tightest per-row bound.
        //
        //     from(5) -> expr = 5, weight = 5
        let (expr, weight) = Count::<F>::from(5).into_parts();
        assert_eq!(expr, F::from_u8(5));
        assert_eq!(weight, 5);
    }

    #[test]
    fn negative_constant_keeps_magnitude_as_bound() {
        // A receive is a negative count; the bound tracks magnitude, not sign.
        //
        //     from(-3) -> expr = -3, weight = 3
        let (expr, weight) = Count::<F>::from(-3).into_parts();
        assert_eq!(expr, -F::from_u8(3));
        assert_eq!(weight, 3);
    }

    #[test]
    fn bounded_stores_the_declared_bound() {
        // A variable count carries the bound the caller declares.
        //
        //     bounded(7, 16) -> expr = 7, weight = 16
        let count = Count::bounded(F::from_u8(7), 16);
        assert_eq!(count.weight(), 16);
        assert_eq!(count.into_parts(), (F::from_u8(7), 16));
    }

    #[test]
    fn provided_entry_carries_zero_weight() {
        // The provided side never queries, so it stays out of the height sum.
        assert_eq!(Count::provided(F::from_u8(9)).weight(), 0);
    }

    #[test]
    fn negation_flips_sign_and_preserves_bound() {
        // Invariant: swapping send and receive negates the count but not its bound.
        //
        //     -bounded(4, 10) -> expr = -4, weight = 10
        let (expr, weight) = (-Count::bounded(F::from_u8(4), 10)).into_parts();
        assert_eq!(expr, -F::from_u8(4));
        assert_eq!(weight, 10);
    }
}
