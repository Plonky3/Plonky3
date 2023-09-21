use p3_field::extension::BinomiallyExtendable;

use crate::{Mersenne31, Mersenne31Complex};

impl BinomiallyExtendable<2> for Mersenne31Complex<Mersenne31> {
    // Verifiable in Sage with
    // ```sage
    // p = 2**31 - 1  # Mersenne31
    // F = GF(p)  # The base field GF(p)
    // R.<x> = F[]  # The polynomial ring over F
    // K.<i> = F.extension(x^2 + 1)  # The complex extension field
    // R2.<y> = K[]
    // f2 = y^2 - i - 2
    // assert f2.is_irreducible()
    // ```
    const W: Self = Self::new(Mersenne31::new(2), Mersenne31::new(1));

    // Verifiable in Sage with
    // ```sage
    // K2.<j> = K.extension(f2)
    //  g = j + 6
    // for f in factor(p^4 - 1):
    //   assert g^((p^4-1) // f) != 1
    // ```
    fn ext_multiplicative_group_generator() -> [Self; 2] {
        [
            Self::new(Mersenne31::new(6), Mersenne31::new(0)),
            Self::new(Mersenne31::new(1), Mersenne31::new(0)),
        ]
    }
}

impl BinomiallyExtendable<3> for Mersenne31Complex<Mersenne31> {
    // Verifiable in Sage with
    // ```sage
    // p = 2**31 - 1  # Mersenne31
    // F = GF(p)  # The base field GF(p)
    // R.<x> = F[]  # The polynomial ring over F
    // K.<i> = F.extension(x^2 + 1)  # The complex extension field
    // R2.<y> = K[]
    // f2 = y^3 - 5*i
    // assert f2.is_irreducible()
    // ```
    const W: Self = Self::new(Mersenne31::new(0), Mersenne31::new(5));

    // Verifiable in Sage with
    // ```sage
    // K2.<j> = K.extension(f2)
    //  g = j + 5
    // for f in factor(p^6 - 1):
    //   assert g^((p^6-1) // f) != 1
    // ```
    fn ext_multiplicative_group_generator() -> [Self; 3] {
        [
            Self::new(Mersenne31::new(5), Mersenne31::new(0)),
            Self::new(Mersenne31::new(1), Mersenne31::new(0)),
            Self::new(Mersenne31::new(0), Mersenne31::new(0)),
        ]
    }
}

#[cfg(test)]
mod test_cubic_extension {

    use p3_field_testing::test_field;

    test_field!(p3_field::extension::cubic::CubicBef<crate::Mersenne31Complex<crate::Mersenne31>>);
}

#[cfg(test)]
mod test_quadratic_extension {

    use p3_field_testing::test_field;

    test_field!(
        p3_field::extension::quadratic::QuadraticBef<crate::Mersenne31Complex<crate::Mersenne31>>
    );
}
