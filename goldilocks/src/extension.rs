use p3_field::extension::BinomiallyExtendable;

use crate::Goldilocks;

impl BinomiallyExtendable<2> for Goldilocks {
    // Verifiable in Sage with
    // `R.<x> = GF(p)[]; assert (x^2 - 7).is_irreducible()`.
    const W: Self = Self::new(7);

    fn ext_multiplicative_group_generator() -> [Self; 2] {
        [
            Self::new(18081566051660590251),
            Self::new(16121475356294670766),
        ]
    }
}

#[cfg(test)]
mod test_quadratic_extension {

    use p3_field_testing::test_field;

    test_field!(p3_field::extension::quadratic::QuadraticBef<crate::Goldilocks>);
}
