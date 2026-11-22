use p3_field::extension::{
    Binomial, BinomiallyExtendable, Complex, ExtensionAlgebra, HasComplexBinomialExtension,
    HasTwoAdicComplexBinomialExtension, binomial_mul, binomial_square,
};
use p3_field::{PrimeCharacteristicRing, TwoAdicField, field_to_array};

use crate::Mersenne31;
use crate::mersenne_31::{partial_reduce, reduce_64};

const P64: u64 = (1 << 31) - 1;

/// Reduce a value in `[0, 2^34)` modulo `P = 2^31 - 1` into `[0, P)`.
#[inline(always)]
const fn reduce_34(val: u64) -> u32 {
    let lo = (val & P64) as u32;
    let hi = (val >> 31) as u32;
    let sum = lo + hi;
    let sub = sum.wrapping_sub(P64 as u32);
    if sum < sub { sum } else { sub }
}

/// Unreduced 2-multiply Gauss squaring in `Mersenne31[i]`.
///
/// Returns `(re, im)` congruent to `(a0 + a1 i)^2 (mod P)` with `0 <= re, im < 4 * P`.
#[inline(always)]
fn cm31_square_unreduced(a: Complex<Mersenne31>) -> (u64, u64) {
    let a0 = a.real().value as u64;
    let a1 = a.imag().value as u64;
    (
        partial_reduce((a0 + a1) * (a0 + (P64 ^ a1))),
        partial_reduce((a0 << 1) * a1),
    )
}

/// Branchless 4-channel `u64` delayed-reduction kernel for `QM31` (`Complex<Mersenne31>[Y] / (Y^2 - (2 + i))`).
#[inline(always)]
fn qm31_mul_u64_dot4(
    a: &[Complex<Mersenne31>; 2],
    b: &[Complex<Mersenne31>; 2],
    res: &mut [Complex<Mersenne31>; 2],
) {
    let wb = <Mersenne31 as HasComplexBinomialExtension<2>>::mul_by_w(b[1]);
    let (a0, a1) = (a[0].real().value as u64, a[0].imag().value as u64);
    let (a2, a3) = (a[1].real().value as u64, a[1].imag().value as u64);
    let (b0, b1) = (b[0].real().value as u64, b[0].imag().value as u64);
    let (b2, b3) = (b[1].real().value as u64, b[1].imag().value as u64);
    let (wb2, wb3) = (wb.real().value as u64, wb.imag().value as u64);
    let na1 = P64 ^ a1;
    let na3 = P64 ^ a3;

    let c0 = reduce_64(a0 * b0 + na1 * b1 + a2 * wb2 + na3 * wb3);
    let c1 = reduce_64(a0 * b1 + a1 * b0 + a2 * wb3 + a3 * wb2);
    let c2 = reduce_64(a0 * b2 + na1 * b3 + a2 * b0 + na3 * b1);
    let c3 = reduce_64(a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0);

    res[0] = Complex::new_complex(Mersenne31::new_reduced(c0), Mersenne31::new_reduced(c1));
    res[1] = Complex::new_complex(Mersenne31::new_reduced(c2), Mersenne31::new_reduced(c3));
}

/// Branchless parallel `u64` squaring kernel for `QM31` (`Complex<Mersenne31>[Y] / (Y^2 - (2 + i))`).
#[inline(always)]
fn qm31_square_u64_parallel8(a: &[Complex<Mersenne31>; 2], res: &mut [Complex<Mersenne31>; 2]) {
    const Q16: u64 = P64 << 4;
    let a0 = a[0].real().value as u64;
    let a1 = a[0].imag().value as u64;
    let a2 = a[1].real().value as u64;
    let a3 = a[1].imag().value as u64;

    let s0 = cm31_square_unreduced(a[0]);
    let s1 = cm31_square_unreduced(a[1]);

    let c0 = reduce_64(s0.0 + (s1.0 << 1) + Q16 - s1.1);
    let c1 = reduce_64(s0.1 + s1.0 + (s1.1 << 1));
    let c2 = reduce_64((a0 * a2 + (P64 ^ a1) * a3) << 1);
    let c3 = reduce_64((a0 * a3 + a1 * a2) << 1);

    res[0] = Complex::new_complex(Mersenne31::new_reduced(c0), Mersenne31::new_reduced(c1));
    res[1] = Complex::new_complex(Mersenne31::new_reduced(c2), Mersenne31::new_reduced(c3));
}

/// Branchless 6-channel `u64` delayed-reduction kernel for degree-6 `Complex<Mersenne31>[Y] / (Y^3 - 5i)`.
#[inline(always)]
fn sextic_mul_u64_dot6(
    a: &[Complex<Mersenne31>; 3],
    b: &[Complex<Mersenne31>; 3],
    res: &mut [Complex<Mersenne31>; 3],
) {
    let wb1 = <Mersenne31 as HasComplexBinomialExtension<3>>::mul_by_w(b[1]);
    let wb2 = <Mersenne31 as HasComplexBinomialExtension<3>>::mul_by_w(b[2]);

    let (a0, a1) = (a[0].real().value as u64, a[0].imag().value as u64);
    let (a2, a3) = (a[1].real().value as u64, a[1].imag().value as u64);
    let (a4, a5) = (a[2].real().value as u64, a[2].imag().value as u64);

    let (b0, b1) = (b[0].real().value as u64, b[0].imag().value as u64);
    let (b2, b3) = (b[1].real().value as u64, b[1].imag().value as u64);
    let (b4, b5) = (b[2].real().value as u64, b[2].imag().value as u64);

    let (wb1_0, wb1_1) = (wb1.real().value as u64, wb1.imag().value as u64);
    let (wb2_0, wb2_1) = (wb2.real().value as u64, wb2.imag().value as u64);

    let na1 = P64 ^ a1;
    let na3 = P64 ^ a3;
    let na5 = P64 ^ a5;

    let c0_re = reduce_64(
        partial_reduce(a0 * b0 + na1 * b1 + a2 * wb2_0 + na3 * wb2_1) + a4 * wb1_0 + na5 * wb1_1,
    );
    let c0_im = reduce_64(
        partial_reduce(a0 * b1 + a1 * b0 + a2 * wb2_1 + a3 * wb2_0) + a4 * wb1_1 + a5 * wb1_0,
    );
    let c1_re = reduce_64(
        partial_reduce(a0 * b2 + na1 * b3 + a2 * b0 + na3 * b1) + a4 * wb2_0 + na5 * wb2_1,
    );
    let c1_im =
        reduce_64(partial_reduce(a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0) + a4 * wb2_1 + a5 * wb2_0);
    let c2_re =
        reduce_64(partial_reduce(a0 * b4 + na1 * b5 + a2 * b2 + na3 * b3) + a4 * b0 + na5 * b1);
    let c2_im =
        reduce_64(partial_reduce(a0 * b5 + a1 * b4 + a2 * b3 + a3 * b2) + a4 * b1 + a5 * b0);

    res[0] = Complex::new_complex(
        Mersenne31::new_reduced(c0_re),
        Mersenne31::new_reduced(c0_im),
    );
    res[1] = Complex::new_complex(
        Mersenne31::new_reduced(c1_re),
        Mersenne31::new_reduced(c1_im),
    );
    res[2] = Complex::new_complex(
        Mersenne31::new_reduced(c2_re),
        Mersenne31::new_reduced(c2_im),
    );
}

/// Branchless parallel `u64` squaring kernel for degree-6 `Complex<Mersenne31>[Y] / (Y^3 - 5i)`.
#[inline(always)]
fn sextic_square_u64_fast(a: &[Complex<Mersenne31>; 3], res: &mut [Complex<Mersenne31>; 3]) {
    const Q128: u64 = P64 << 7;
    let (a0, a1) = (a[0].real().value as u64, a[0].imag().value as u64);
    let (a2, a3) = (a[1].real().value as u64, a[1].imag().value as u64);
    let (a4, a5) = (a[2].real().value as u64, a[2].imag().value as u64);

    let na1 = P64 ^ a1;
    let na3 = P64 ^ a3;
    let na5 = P64 ^ a5;

    let s0_re = partial_reduce((a0 + a1) * (a0 + na1));
    let s0_im = partial_reduce((a0 << 1) * a1);
    let s1_re = partial_reduce((a2 + a3) * (a2 + na3));
    let s1_im = partial_reduce((a2 << 1) * a3);
    let s2_re = partial_reduce((a4 + a5) * (a4 + na5));
    let s2_im = partial_reduce((a4 << 1) * a5);

    let c01_re = partial_reduce(a0 * a2 + na1 * a3);
    let c01_im = partial_reduce(a0 * a3 + a1 * a2);
    let c02_re = partial_reduce(a0 * a4 + na1 * a5);
    let c02_im = partial_reduce(a0 * a5 + a1 * a4);
    let c12_re = partial_reduce(a2 * a4 + na3 * a5);
    let c12_im = partial_reduce(a2 * a5 + a3 * a4);

    let c0_re = reduce_64(s0_re + Q128 - 10 * c12_im);
    let c0_im = reduce_64(s0_im + 10 * c12_re);
    let c1_re = reduce_64((c01_re << 1) + Q128 - 5 * s2_im);
    let c1_im = reduce_64((c01_im << 1) + 5 * s2_re);
    let c2_re = reduce_64(s1_re + (c02_re << 1));
    let c2_im = reduce_64(s1_im + (c02_im << 1));

    res[0] = Complex::new_complex(
        Mersenne31::new_reduced(c0_re),
        Mersenne31::new_reduced(c0_im),
    );
    res[1] = Complex::new_complex(
        Mersenne31::new_reduced(c1_re),
        Mersenne31::new_reduced(c1_im),
    );
    res[2] = Complex::new_complex(
        Mersenne31::new_reduced(c2_re),
        Mersenne31::new_reduced(c2_im),
    );
}

impl ExtensionAlgebra<Self, 3, Binomial<Self>> for Mersenne31 {
    #[inline]
    fn ext_mul(a: &[Self; 3], b: &[Self; 3], res: &mut [Self; 3]) {
        binomial_mul::<Self, Self, Self, 3>(a, b, res, <Self as BinomiallyExtendable<3>>::W);
    }

    #[inline]
    fn ext_square(a: &[Self; 3], res: &mut [Self; 3]) {
        binomial_square::<Self, Self, 3>(a, res, <Self as BinomiallyExtendable<3>>::W);
    }
}

impl BinomiallyExtendable<3> for Mersenne31 {
    fn binomial_algebra_id() -> alloc::vec::Vec<u8> {
        use p3_field::PrimeField32;
        alloc::format!(
            "p3-power-basis-v1:X^3-{}",
            <Self as BinomiallyExtendable<3>>::W.as_canonical_u32()
        )
        .into_bytes()
    }

    // ```sage
    // p = 2^31 - 1
    // F = GF(p)
    // R.<x> = F[]
    // assert (x^3 - 5).is_irreducible()
    // ```
    const W: Self = Self::new(5);

    // ```sage
    // F(5)^((p-1)/3)
    // ```
    const DTH_ROOT: Self = Self::new(1513477735);

    // ```sage
    // F.extension(x^3 - 5, 'u').multiplicative_generator()
    // ```
    const EXT_GENERATOR: [Self; 3] = [Self::new(10), Self::ONE, Self::ZERO];
}

impl HasComplexBinomialExtension<2> for Mersenne31 {
    fn complex_binomial_algebra_id() -> alloc::vec::Vec<u8> {
        use p3_field::PrimeField32;
        let w = <Self as HasComplexBinomialExtension<2>>::W;
        alloc::format!(
            "p3-complex-power-basis-v1:X^2-({}+{}i)",
            w.real().as_canonical_u32(),
            w.imag().as_canonical_u32()
        )
        .into_bytes()
    }

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
    const W: Complex<Self> = Complex::new_complex(Self::TWO, Self::ONE);

    // DTH_ROOT = W^((p^2 - 1)/2).
    const DTH_ROOT: Complex<Self> = Complex::new_real(Self::new(2147483646));

    // Verifiable in Sage with
    // ```sage
    // K2.<j> = K.extension(f2)
    //  g = j + 6
    // for f in factor(p^4 - 1):
    //   assert g^((p^4-1) // f) != 1
    // ```
    const EXT_GENERATOR: [Complex<Self>; 2] = [Complex::new_real(Self::new(6)), Complex::ONE];

    /// Multiply a `Complex<Mersenne31>` element by `W = 2 + i` using only additions:
    /// `(a + bi)(2 + i) = (2a - b) + (a + 2b)i`.
    #[inline(always)]
    fn mul_by_w(z: Complex<Self>) -> Complex<Self> {
        let re = z.real();
        let im = z.imag();
        Complex::new_complex(re + re - im, re + im + im)
    }

    #[inline(always)]
    fn complex_ext_mul(
        a: &[Complex<Self>; 2],
        b: &[Complex<Self>; 2],
        res: &mut [Complex<Self>; 2],
    ) {
        qm31_mul_u64_dot4(a, b, res);
    }

    #[inline(always)]
    fn complex_ext_square(a: &[Complex<Self>; 2], res: &mut [Complex<Self>; 2]) {
        qm31_square_u64_parallel8(a, res);
    }
}

impl HasTwoAdicComplexBinomialExtension<2> for Mersenne31 {
    const COMPLEX_EXT_TWO_ADICITY: usize = 33;

    fn complex_ext_two_adic_generator(bits: usize) -> [Complex<Self>; 2] {
        assert!(bits <= 33);
        if bits == 33 {
            [
                Complex::ZERO,
                Complex::new_complex(Self::new_reduced(1437746044), Self::new_reduced(946469285)),
            ]
        } else {
            [Complex::two_adic_generator(bits), Complex::ZERO]
        }
    }
}

impl HasComplexBinomialExtension<3> for Mersenne31 {
    fn complex_binomial_algebra_id() -> alloc::vec::Vec<u8> {
        use p3_field::PrimeField32;
        let w = <Self as HasComplexBinomialExtension<3>>::W;
        alloc::format!(
            "p3-complex-power-basis-v1:X^3-({}+{}i)",
            w.real().as_canonical_u32(),
            w.imag().as_canonical_u32()
        )
        .into_bytes()
    }

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
    const W: Complex<Self> = Complex::new_imag(Self::new(5));

    // DTH_ROOT = W^((p^2 - 1)/3).
    const DTH_ROOT: Complex<Self> = Complex::new_real(Self::new(634005911));

    // Verifiable in Sage with
    // ```sage
    // K2.<j> = K.extension(f2)
    //  g = j + 5
    // for f in factor(p^6 - 1):
    //   assert g^((p^6-1) // f) != 1
    // ```
    const EXT_GENERATOR: [Complex<Self>; 3] = [
        Complex::new_real(Self::new(5)),
        Complex::new_real(Self::ONE),
        Complex::ZERO,
    ];

    /// Multiply a `Complex<Mersenne31>` element by `W = 5i` branchlessly:
    /// `(a + bi)(5i) = -5b + 5ai`.
    #[inline(always)]
    fn mul_by_w(z: Complex<Self>) -> Complex<Self> {
        let five_re = reduce_34(5 * (z.real().value as u64));
        let five_im = reduce_34(5 * (z.imag().value as u64));
        Complex::new_complex(
            Self::new_reduced((P64 as u32) ^ five_im),
            Self::new_reduced(five_re),
        )
    }

    #[inline(always)]
    fn complex_ext_mul(
        a: &[Complex<Self>; 3],
        b: &[Complex<Self>; 3],
        res: &mut [Complex<Self>; 3],
    ) {
        sextic_mul_u64_dot6(a, b, res);
    }

    #[inline(always)]
    fn complex_ext_square(a: &[Complex<Self>; 3], res: &mut [Complex<Self>; 3]) {
        sextic_square_u64_fast(a, res);
    }
}

impl HasTwoAdicComplexBinomialExtension<3> for Mersenne31 {
    const COMPLEX_EXT_TWO_ADICITY: usize = 32;

    fn complex_ext_two_adic_generator(bits: usize) -> [Complex<Self>; 3] {
        field_to_array(Complex::two_adic_generator(bits))
    }
}

#[cfg(test)]
mod test_cubic_extension {
    use num_bigint::BigUint;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{ExtensionField, PrimeCharacteristicRing};
    use p3_field_testing::{test_extension_field, test_field, test_packed_extension_field};

    use crate::Mersenne31;

    type F = Mersenne31;
    type EF = BinomialExtensionField<F, 3>;

    // There is a redundant representation of zero but we already tested it
    // when testing the base field.
    const ZEROS: [EF; 1] = [EF::ZERO];
    const ONES: [EF; 1] = [EF::ONE];

    // Get the prime factorization of the order of the multiplicative group.
    // i.e. the prime factorization of P^3 - 1.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 9] {
        [
            (BigUint::from(2u8), 1),
            (BigUint::from(3u8), 3),
            (BigUint::from(7u8), 1),
            (BigUint::from(11u8), 1),
            (BigUint::from(31u8), 1),
            (BigUint::from(151u8), 1),
            (BigUint::from(331u16), 1),
            (BigUint::from(529510939u32), 1),
            (BigUint::from(2903110321u32), 1),
        ]
    }

    test_extension_field!(super::F, super::EF);

    test_field!(
        super::EF,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );

    type Pef = <EF as ExtensionField<F>>::ExtensionPacking;
    const PACKED_ZEROS: [Pef; 1] = [Pef::ZERO];
    const PACKED_ONES: [Pef; 1] = [Pef::ONE];
    test_packed_extension_field!(
        super::F,
        super::EF,
        super::Pef,
        &super::PACKED_ZEROS,
        &super::PACKED_ONES
    );
    p3_field_testing::test_packed_binomial_extension_division!(F, 3);
}

#[cfg(test)]
mod test_cubic_complex_extension {
    use num_bigint::BigUint;
    use p3_field::extension::{BinomialExtensionField, Complex};
    use p3_field::{ExtensionField, PrimeCharacteristicRing};
    use p3_field_testing::{
        test_extension_field, test_field, test_packed_extension_field,
        test_two_adic_extension_field,
    };

    use crate::Mersenne31;

    type F = Complex<Mersenne31>;
    type EF = BinomialExtensionField<F, 3>;

    // There is a redundant representation of zero but we already tested it
    // when testing the base field.
    const ZEROS: [EF; 1] = [EF::ZERO];
    const ONES: [EF; 1] = [EF::ONE];

    // Get the prime factorization of the order of the multiplicative group.
    // i.e. the prime factorization of P^6 - 1.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 14] {
        [
            (BigUint::from(2u8), 32),
            (BigUint::from(3u8), 3),
            (BigUint::from(7u8), 1),
            (BigUint::from(11u8), 1),
            (BigUint::from(13u8), 1),
            (BigUint::from(31u8), 1),
            (BigUint::from(43u8), 2),
            (BigUint::from(79u8), 1),
            (BigUint::from(151u8), 1),
            (BigUint::from(331u16), 1),
            (BigUint::from(1381u16), 1),
            (BigUint::from(529510939u32), 1),
            (BigUint::from(1758566101u32), 1),
            (BigUint::from(2903110321u32), 1),
        ]
    }

    test_field!(
        super::EF,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );

    test_extension_field!(super::F, super::EF);

    test_two_adic_extension_field!(super::F, super::EF);

    type Pef = <EF as ExtensionField<F>>::ExtensionPacking;
    const PACKED_ZEROS: [Pef; 1] = [Pef::ZERO];
    const PACKED_ONES: [Pef; 1] = [Pef::ONE];
    test_packed_extension_field!(
        super::F,
        super::EF,
        super::Pef,
        &super::PACKED_ZEROS,
        &super::PACKED_ONES
    );
    p3_field_testing::test_packed_binomial_extension_division!(F, 3);
}

#[cfg(test)]
mod test_quadratic_complex_extension {

    use num_bigint::BigUint;
    use p3_field::extension::{BinomialExtensionField, Complex};
    use p3_field::{ExtensionField, PrimeCharacteristicRing};
    use p3_field_testing::{
        test_extension_field, test_field, test_packed_extension_field,
        test_two_adic_extension_field,
    };

    use crate::Mersenne31;

    type F = Complex<Mersenne31>;
    type EF = BinomialExtensionField<F, 2>;

    // There is a redundant representation of zero but we already tested it
    // when testing the base field.
    const ZEROS: [EF; 1] = [EF::ZERO];
    const ONES: [EF; 1] = [EF::ONE];

    // Get the prime factorization of the order of the multiplicative group.
    // i.e. the prime factorization of P^4 - 1.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 11] {
        [
            (BigUint::from(2u8), 33),
            (BigUint::from(3u8), 2),
            (BigUint::from(5u8), 1),
            (BigUint::from(7u8), 1),
            (BigUint::from(11u8), 1),
            (BigUint::from(31u8), 1),
            (BigUint::from(151u8), 1),
            (BigUint::from(331u16), 1),
            (BigUint::from(733u16), 1),
            (BigUint::from(1709u16), 1),
            (BigUint::from(368140581013u64), 1),
        ]
    }

    test_field!(
        super::EF,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );

    test_extension_field!(super::F, super::EF);

    test_two_adic_extension_field!(super::F, super::EF);

    type Pef = <EF as ExtensionField<F>>::ExtensionPacking;
    const PACKED_ZEROS: [Pef; 1] = [Pef::ZERO];
    const PACKED_ONES: [Pef; 1] = [Pef::ONE];
    test_packed_extension_field!(
        super::F,
        super::EF,
        super::Pef,
        &super::PACKED_ZEROS,
        &super::PACKED_ONES
    );
    p3_field_testing::test_packed_binomial_extension_division!(F, 2);
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::{Complex, HasComplexBinomialExtension};

    use crate::Mersenne31;

    #[test]
    fn complex_binomial_dth_roots() {
        // DTH_ROOT = W^((p^2 - 1)/D) for the degree-D extension of the complex field.
        //
        // Here p^2 - 1 is the order of the complex field's multiplicative group.
        let p = (1u64 << 31) - 1;
        let order = p * p - 1;

        let w2 = <Mersenne31 as HasComplexBinomialExtension<2>>::W;
        let dth2 = <Mersenne31 as HasComplexBinomialExtension<2>>::DTH_ROOT;
        assert_eq!(w2.exp_u64(order / 2), dth2);

        let w3 = <Mersenne31 as HasComplexBinomialExtension<3>>::W;
        let dth3 = <Mersenne31 as HasComplexBinomialExtension<3>>::DTH_ROOT;
        assert_eq!(w3.exp_u64(order / 3), dth3);

        // The cubic root is a primitive cube root of unity.
        assert_ne!(dth3, Complex::ONE);
        assert_eq!(dth3.cube(), Complex::ONE);
    }

    #[test]
    fn qm31_and_sextic_kernels_match_schoolbook_across_extremes_and_random() {
        use core::array;

        use p3_field::extension::{BinomialExtensionField, binomial_mul, binomial_square};
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        use super::*;

        type EF2 = BinomialExtensionField<Complex<Mersenne31>, 2>;
        type EF3 = BinomialExtensionField<Complex<Mersenne31>, 3>;

        let w2 = <Mersenne31 as HasComplexBinomialExtension<2>>::W;
        let w3 = <Mersenne31 as HasComplexBinomialExtension<3>>::W;

        // Include non-canonical zero P = 2^31 - 1 alongside 0, 1, 2, (P - 1) / 2 (maximizer of
        // `(a0 + a1) * (a0 + (P ^ a1))` at `a0 = P`), 858_993_459 (where `5 * x = 2^32 - 1` hits
        // `lo = P` in `reduce_34`), and P - 1.
        let p_zero = Mersenne31::new_reduced((1 << 31) - 1);
        let edges = [
            Mersenne31::ZERO,
            Mersenne31::ONE,
            Mersenne31::TWO,
            Mersenne31::new_reduced(858_993_459),
            Mersenne31::new_reduced(1_073_741_823),
            Mersenne31::NEG_ONE,
            p_zero,
        ];

        for &a0 in &edges {
            for &a1 in &edges {
                let c = Complex::new_complex(a0, a1);
                let sq = c.square();
                assert_eq!(sq, c * c);
                assert_eq!(sq.real().value >> 31, 0);
                assert_eq!(sq.imag().value >> 31, 0);
                let mw2 = <Mersenne31 as HasComplexBinomialExtension<2>>::mul_by_w(c);
                let mw3 = <Mersenne31 as HasComplexBinomialExtension<3>>::mul_by_w(c);
                assert_eq!(mw2, w2 * c);
                assert_eq!(mw3, w3 * c);
                assert_eq!(mw2.real().value >> 31, 0);
                assert_eq!(mw2.imag().value >> 31, 0);
                assert_eq!(mw3.real().value >> 31, 0);
                assert_eq!(mw3.imag().value >> 31, 0);
            }
        }

        // Exhaustive extreme-value stress test (including non-canonical P) against binomial_mul/square
        for &x in &edges {
            for &y in &edges {
                for &z in &edges {
                    let a = [Complex::new_complex(x, y), Complex::new_complex(z, x)];
                    let b = [Complex::new_complex(y, p_zero), Complex::new_complex(x, z)];
                    let mut ref2 = [Complex::ZERO; 2];
                    binomial_mul::<Complex<Mersenne31>, _, _, 2>(&a, &b, &mut ref2, w2);
                    assert_eq!(EF2::new(a) * EF2::new(b), EF2::new(ref2));

                    let mut ref2_sq = [Complex::ZERO; 2];
                    binomial_square::<Complex<Mersenne31>, _, 2>(&a, &mut ref2_sq, w2);
                    assert_eq!(EF2::new(a).square(), EF2::new(ref2_sq));

                    let a3 = [a[0], a[1], b[0]];
                    let b3 = [b[0], b[1], a[1]];
                    let mut ref3 = [Complex::ZERO; 3];
                    binomial_mul::<Complex<Mersenne31>, _, _, 3>(&a3, &b3, &mut ref3, w3);
                    assert_eq!(EF3::new(a3) * EF3::new(b3), EF3::new(ref3));

                    let mut ref3_sq = [Complex::ZERO; 3];
                    binomial_square::<Complex<Mersenne31>, _, 3>(&a3, &mut ref3_sq, w3);
                    assert_eq!(EF3::new(a3).square(), EF3::new(ref3_sq));
                }
            }
        }

        let mut rng = SmallRng::seed_from_u64(0x1234_5678_9ABC_DEF0);
        for _ in 0..2_000 {
            let a: [Complex<Mersenne31>; 2] = array::from_fn(|_| rng.random());
            let b: [Complex<Mersenne31>; 2] = array::from_fn(|_| rng.random());
            let mut ref2 = [Complex::ZERO; 2];
            binomial_mul::<Complex<Mersenne31>, _, _, 2>(&a, &b, &mut ref2, w2);
            assert_eq!(EF2::new(a) * EF2::new(b), EF2::new(ref2));
            assert_eq!(EF2::new(a).square(), EF2::new(a) * EF2::new(a));

            let a3: [Complex<Mersenne31>; 3] = array::from_fn(|_| rng.random());
            let b3: [Complex<Mersenne31>; 3] = array::from_fn(|_| rng.random());
            let mut ref3 = [Complex::ZERO; 3];
            binomial_mul::<Complex<Mersenne31>, _, _, 3>(&a3, &b3, &mut ref3, w3);
            assert_eq!(EF3::new(a3) * EF3::new(b3), EF3::new(ref3));
            assert_eq!(EF3::new(a3).square(), EF3::new(a3) * EF3::new(a3));
        }
    }
}
