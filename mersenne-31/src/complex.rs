//! Implementation of the quadratic extension of the Mersenne31 field
//! by X^2 + 1.
//!
//! Note that X^2 + 1 is irreducible over p = Mersenne31 field because
//! kronecker(-1, p) = -1, that is, -1 is not square in F_p.

use p3_field::extension::{Complex, ComplexExtendable, HasTwoAdicBionmialExtension};
use p3_field::AbstractField;

use crate::Mersenne31;

impl ComplexExtendable for Mersenne31 {
    const CIRCLE_TWO_ADICITY: usize = 31;

    // sage: p = 2^31 - 1
    // sage: F = GF(p)
    // sage: R.<x> = F[]
    // sage: F2.<u> = F.extension(x^2 + 1)
    // sage: F2.multiplicative_generator()
    // u + 12
    fn complex_generator() -> Complex<Self> {
        Complex::new(Mersenne31::new(12), Mersenne31::one())
    }

    fn circle_two_adic_generator(bits: usize) -> Complex<Self> {
        // Generator of the whole 2^TWO_ADICITY group
        // sage: p = 2^31 - 1
        // sage: F = GF(p)
        // sage: R.<x> = F[]
        // sage: F2.<u> = F.extension(x^2 + 1)
        // sage: g = F2.multiplicative_generator()^((p^2 - 1) / 2^31); g
        // 1584694829*u + 311014874
        // sage: assert(g.multiplicative_order() == 2^31)
        // sage: assert(g.norm() == 1)
        let base = Complex::new(Mersenne31::new(311_014_874), Mersenne31::new(1_584_694_829));
        base.exp_power_of_2(Self::CIRCLE_TWO_ADICITY - bits)
    }
}

impl HasTwoAdicBionmialExtension<2> for Mersenne31 {
    const EXT_TWO_ADICITY: usize = 32;

    fn ext_two_adic_generator(bits: usize) -> [Self; 2] {
        // TODO: Consider a `match` which may speed this up.
        assert!(bits <= Self::EXT_TWO_ADICITY);
        // Generator of the whole 2^TWO_ADICITY group
        // sage: p = 2^31 - 1
        // sage: F = GF(p)
        // sage: R.<x> = F[]
        // sage: F2.<u> = F.extension(x^2 + 1)
        // sage: g = F2.multiplicative_generator()^((p^2 - 1) / 2^32); g
        // 1117296306*u + 1166849849
        // sage: assert(g.multiplicative_order() == 2^32)
        let base = Complex::<Self>::new(
            Mersenne31::new(1_166_849_849),
            Mersenne31::new(1_117_296_306),
        );
        base.exp_power_of_2(Self::EXT_TWO_ADICITY - bits).to_array()
    }
}

#[cfg(test)]
mod tests {
    use p3_field::{Field, PrimeField32};
    use p3_field_testing::{test_field, test_two_adic_field};

    use super::*;

    type Fi = Complex<Mersenne31>;
    type F = Mersenne31;

    #[test]
    fn add() {
        // real part
        assert_eq!(Fi::one() + Fi::one(), Fi::two());
        assert_eq!(Fi::neg_one() + Fi::one(), Fi::zero());
        assert_eq!(Fi::neg_one() + Fi::two(), Fi::one());
        assert_eq!(
            (Fi::neg_one() + Fi::neg_one()).real(),
            F::new(F::ORDER_U32 - 2)
        );

        // complex part
        assert_eq!(
            Fi::new_imag(F::one()) + Fi::new_imag(F::one()),
            Fi::new_imag(F::two())
        );
        assert_eq!(
            Fi::new_imag(F::neg_one()) + Fi::new_imag(F::one()),
            Fi::new_imag(F::zero())
        );
        assert_eq!(
            Fi::new_imag(F::neg_one()) + Fi::new_imag(F::two()),
            Fi::new_imag(F::one())
        );
        assert_eq!(
            (Fi::new_imag(F::neg_one()) + Fi::new_imag(F::neg_one())).imag(),
            F::new(F::ORDER_U32 - 2)
        );

        // further tests
        assert_eq!(
            Fi::new(F::one(), F::two()) + Fi::new(F::one(), F::one()),
            Fi::new(F::two(), F::new(3))
        );
        assert_eq!(
            Fi::new(F::neg_one(), F::neg_one()) + Fi::new(F::one(), F::one()),
            Fi::zero()
        );
        assert_eq!(
            Fi::new(F::neg_one(), F::one()) + Fi::new(F::two(), F::new(F::ORDER_U32 - 2)),
            Fi::new(F::one(), F::neg_one())
        );
    }

    #[test]
    fn sub() {
        // real part
        assert_eq!(Fi::one() - Fi::one(), Fi::zero());
        assert_eq!(Fi::two() - Fi::two(), Fi::zero());
        assert_eq!(Fi::neg_one() - Fi::neg_one(), Fi::zero());
        assert_eq!(Fi::two() - Fi::one(), Fi::one());
        assert_eq!(Fi::neg_one() - Fi::zero(), Fi::neg_one());

        // complex part
        assert_eq!(Fi::new_imag(F::one()) - Fi::new_imag(F::one()), Fi::zero());
        assert_eq!(Fi::new_imag(F::two()) - Fi::new_imag(F::two()), Fi::zero());
        assert_eq!(
            Fi::new_imag(F::neg_one()) - Fi::new_imag(F::neg_one()),
            Fi::zero()
        );
        assert_eq!(
            Fi::new_imag(F::two()) - Fi::new_imag(F::one()),
            Fi::new_imag(F::one())
        );
        assert_eq!(
            Fi::new_imag(F::neg_one()) - Fi::zero(),
            Fi::new_imag(F::neg_one())
        );
    }

    #[test]
    fn mul() {
        assert_eq!(
            Fi::new(F::two(), F::two()) * Fi::new(F::new(4), F::new(5)),
            Fi::new(-F::two(), F::new(18))
        );
    }

    #[test]
    fn mul_2exp_u64() {
        // real part
        // 1 * 2^0 = 1.
        assert_eq!(Fi::one().mul_2exp_u64(0), Fi::one());
        // 2 * 2^30 = 2^31 = 1.
        assert_eq!(Fi::two().mul_2exp_u64(30), Fi::one());
        // 5 * 2^2 = 20.
        assert_eq!(
            Fi::new_real(F::new(5)).mul_2exp_u64(2),
            Fi::new_real(F::new(20))
        );

        // complex part
        // i * 2^0 = i.
        assert_eq!(
            Fi::new_imag(F::one()).mul_2exp_u64(0),
            Fi::new_imag(F::one())
        );
        // (2i) * 2^30 = (2^31) * i = i.
        assert_eq!(
            Fi::new_imag(F::two()).mul_2exp_u64(30),
            Fi::new_imag(F::one())
        );
        // 5i * 2^2 = 20i.
        assert_eq!(
            Fi::new_imag(F::new(5)).mul_2exp_u64(2),
            Fi::new_imag(F::new(20))
        );
    }

    test_field!(p3_field::extension::Complex<crate::Mersenne31>);
    test_two_adic_field!(p3_field::extension::Complex<crate::Mersenne31>);
}
