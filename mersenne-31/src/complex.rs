//! Implementation of the quadratic extension of the Mersenne31 field
//! by X^2 + 1.
//!
//! Note that X^2 + 1 is irreducible over p = Mersenne31 field because
//! kronecker(-1, p) = -1, that is, -1 is not square in F_p.

use p3_field::PrimeCharacteristicRing;
use p3_field::extension::{Complex, ComplexExtendable, HasTwoAdicBinomialExtension};

use crate::Mersenne31;

impl Mersenne31 {
    /// Precomputed table of generators for two-adic subgroups of the circle group
    /// (the norm-1 subgroup of the degree two extension field over Mersenne31).
    /// The `i`'th element is a generator of the subgroup of order `2^i`.
    const CIRCLE_TWO_ADIC_GENERATORS: [Complex<Self>; 32] = [
        Complex::new_complex(Self::new(1), Self::new(0)),
        Complex::new_complex(Self::new(2_147_483_646), Self::new(0)),
        Complex::new_complex(Self::new(0), Self::new(2_147_483_646)),
        Complex::new_complex(Self::new(32_768), Self::new(2_147_450_879)),
        Complex::new_complex(Self::new(590_768_354), Self::new(978_592_373)),
        Complex::new_complex(Self::new(1_179_735_656), Self::new(1_241_207_368)),
        Complex::new_complex(Self::new(1_567_857_810), Self::new(456_695_729)),
        Complex::new_complex(Self::new(1_774_253_895), Self::new(1_309_288_441)),
        Complex::new_complex(Self::new(736_262_640), Self::new(1_553_669_210)),
        Complex::new_complex(Self::new(1_819_216_575), Self::new(1_662_816_114)),
        Complex::new_complex(Self::new(1_323_191_254), Self::new(1_936_974_060)),
        Complex::new_complex(Self::new(605_622_498), Self::new(1_964_232_216)),
        Complex::new_complex(Self::new(343_674_985), Self::new(501_786_993)),
        Complex::new_complex(Self::new(1_995_316_534), Self::new(149_306_621)),
        Complex::new_complex(Self::new(2_107_600_913), Self::new(1_378_821_388)),
        Complex::new_complex(Self::new(541_476_169), Self::new(2_101_081_972)),
        Complex::new_complex(Self::new(2_135_874_973), Self::new(483_411_332)),
        Complex::new_complex(Self::new(2_097_144_245), Self::new(1_684_033_590)),
        Complex::new_complex(Self::new(1_662_322_247), Self::new(670_236_780)),
        Complex::new_complex(Self::new(1_172_215_635), Self::new(595_888_646)),
        Complex::new_complex(Self::new(241_940_101), Self::new(323_856_519)),
        Complex::new_complex(Self::new(1_957_194_259), Self::new(2_139_647_100)),
        Complex::new_complex(Self::new(1_957_419_629), Self::new(1_541_039_442)),
        Complex::new_complex(Self::new(1_062_045_235), Self::new(1_824_580_421)),
        Complex::new_complex(Self::new(1_929_382_196), Self::new(1_664_698_822)),
        Complex::new_complex(Self::new(1_889_294_251), Self::new(331_248_939)),
        Complex::new_complex(Self::new(1_214_231_414), Self::new(1_646_302_518)),
        Complex::new_complex(Self::new(1_765_392_370), Self::new(461_136_547)),
        Complex::new_complex(Self::new(1_629_751_483), Self::new(66_485_474)),
        Complex::new_complex(Self::new(1_501_355_827), Self::new(1_439_063_420)),
        Complex::new_complex(Self::new(509_778_402), Self::new(800_467_507)),
        Complex::new_complex(Self::new(311_014_874), Self::new(1_584_694_829)),
    ];
}

impl ComplexExtendable for Mersenne31 {
    const CIRCLE_TWO_ADICITY: usize = 31;

    // sage: p = 2^31 - 1
    // sage: F = GF(p)
    // sage: R.<x> = F[]
    // sage: F2.<u> = F.extension(x^2 + 1)
    // sage: F2.multiplicative_generator()
    // u + 12
    const COMPLEX_GENERATOR: Complex<Self> = Complex::new_complex(Self::new(12), Self::ONE);

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
        assert!(bits <= Self::CIRCLE_TWO_ADICITY);
        Self::CIRCLE_TWO_ADIC_GENERATORS[bits]
    }
}

impl HasTwoAdicBinomialExtension<2> for Mersenne31 {
    const EXT_TWO_ADICITY: usize = 32;

    fn ext_two_adic_generator(bits: usize) -> [Self; 2] {
        assert!(bits <= Self::EXT_TWO_ADICITY);
        Self::EXT_TWO_ADIC_GENERATORS[bits]
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;
    use p3_field::{ExtensionField, PrimeField32};
    use p3_field_testing::{
        test_extension_field, test_field, test_packed_extension_field, test_two_adic_field,
    };

    use super::*;

    type Fi = Complex<Mersenne31>;
    type F = Mersenne31;

    #[test]
    fn add() {
        // real part
        assert_eq!(Fi::ONE + Fi::ONE, Fi::TWO);
        assert_eq!(Fi::NEG_ONE + Fi::ONE, Fi::ZERO);
        assert_eq!(Fi::NEG_ONE + Fi::TWO, Fi::ONE);
        assert_eq!((Fi::NEG_ONE + Fi::NEG_ONE).real(), F::new(F::ORDER_U32 - 2));

        // complex part
        assert_eq!(
            Fi::new_imag(F::ONE) + Fi::new_imag(F::ONE),
            Fi::new_imag(F::TWO)
        );
        assert_eq!(
            Fi::new_imag(F::NEG_ONE) + Fi::new_imag(F::ONE),
            Fi::new_imag(F::ZERO)
        );
        assert_eq!(
            Fi::new_imag(F::NEG_ONE) + Fi::new_imag(F::TWO),
            Fi::new_imag(F::ONE)
        );
        assert_eq!(
            (Fi::new_imag(F::NEG_ONE) + Fi::new_imag(F::NEG_ONE)).imag(),
            F::new(F::ORDER_U32 - 2)
        );

        // further tests
        assert_eq!(
            Fi::new_complex(F::ONE, F::TWO) + Fi::new_complex(F::ONE, F::ONE),
            Fi::new_complex(F::TWO, F::new(3))
        );
        assert_eq!(
            Fi::new_complex(F::NEG_ONE, F::NEG_ONE) + Fi::new_complex(F::ONE, F::ONE),
            Fi::ZERO
        );
        assert_eq!(
            Fi::new_complex(F::NEG_ONE, F::ONE) + Fi::new_complex(F::TWO, F::new(F::ORDER_U32 - 2)),
            Fi::new_complex(F::ONE, F::NEG_ONE)
        );
    }

    #[test]
    fn sub() {
        // real part
        assert_eq!(Fi::ONE - Fi::ONE, Fi::ZERO);
        assert_eq!(Fi::TWO - Fi::TWO, Fi::ZERO);
        assert_eq!(Fi::NEG_ONE - Fi::NEG_ONE, Fi::ZERO);
        assert_eq!(Fi::TWO - Fi::ONE, Fi::ONE);
        assert_eq!(Fi::NEG_ONE - Fi::ZERO, Fi::NEG_ONE);

        // complex part
        assert_eq!(Fi::new_imag(F::ONE) - Fi::new_imag(F::ONE), Fi::ZERO);
        assert_eq!(Fi::new_imag(F::TWO) - Fi::new_imag(F::TWO), Fi::ZERO);
        assert_eq!(
            Fi::new_imag(F::NEG_ONE) - Fi::new_imag(F::NEG_ONE),
            Fi::ZERO
        );
        assert_eq!(
            Fi::new_imag(F::TWO) - Fi::new_imag(F::ONE),
            Fi::new_imag(F::ONE)
        );
        assert_eq!(
            Fi::new_imag(F::NEG_ONE) - Fi::ZERO,
            Fi::new_imag(F::NEG_ONE)
        );
    }

    #[test]
    fn mul() {
        assert_eq!(
            Fi::new_complex(F::TWO, F::TWO) * Fi::new_complex(F::new(4), F::new(5)),
            Fi::new_complex(-F::TWO, F::new(18))
        );
    }

    #[test]
    fn mul_2exp_u64() {
        // real part
        // 1 * 2^0 = 1.
        assert_eq!(Fi::ONE.mul_2exp_u64(0), Fi::ONE);
        // 2 * 2^30 = 2^31 = 1.
        assert_eq!(Fi::TWO.mul_2exp_u64(30), Fi::ONE);
        // 5 * 2^2 = 20.
        assert_eq!(
            Fi::new_real(F::new(5)).mul_2exp_u64(2),
            Fi::new_real(F::new(20))
        );

        // complex part
        // i * 2^0 = i.
        assert_eq!(Fi::new_imag(F::ONE).mul_2exp_u64(0), Fi::new_imag(F::ONE));
        // (2i) * 2^30 = (2^31) * i = i.
        assert_eq!(Fi::new_imag(F::TWO).mul_2exp_u64(30), Fi::new_imag(F::ONE));
        // 5i * 2^2 = 20i.
        assert_eq!(
            Fi::new_imag(F::new(5)).mul_2exp_u64(2),
            Fi::new_imag(F::new(20))
        );
    }

    #[test]
    fn circle_two_adic_generators_table_matches_repeated_squaring() {
        let base = Fi::new_complex(F::new(311_014_874), F::new(1_584_694_829));
        for bits in 0..=Mersenne31::CIRCLE_TWO_ADICITY {
            assert_eq!(
                Mersenne31::CIRCLE_TWO_ADIC_GENERATORS[bits],
                base.exp_power_of_2(Mersenne31::CIRCLE_TWO_ADICITY - bits)
            );
        }
    }

    // There is a redundant representation of zero but we already tested it
    // when testing the base field.
    const ZEROS: [Fi; 1] = [Fi::ZERO];
    const ONES: [Fi; 1] = [Fi::ONE];

    // Get the prime factorization of the order of the multiplicative group.
    // i.e. the prime factorization of P^2 - 1.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 7] {
        [
            (BigUint::from(2u8), 32),
            (BigUint::from(3u8), 2),
            (BigUint::from(7u8), 1),
            (BigUint::from(11u8), 1),
            (BigUint::from(31u8), 1),
            (BigUint::from(151u8), 1),
            (BigUint::from(331u16), 1),
        ]
    }

    test_field!(
        super::Fi,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );

    test_extension_field!(super::F, super::Fi);
    test_two_adic_field!(super::Fi);

    type Pef = <Fi as ExtensionField<F>>::ExtensionPacking;
    const PACKED_ZEROS: [Pef; 1] = [Pef::ZERO];
    const PACKED_ONES: [Pef; 1] = [Pef::ONE];
    test_packed_extension_field!(
        super::F,
        super::Fi,
        super::Pef,
        &super::PACKED_ZEROS,
        &super::PACKED_ONES
    );
    p3_field_testing::test_packed_binomial_extension_division!(F, 2);
}
