use p3_field::extension::{
    BinomiallyExtendable, BinomiallyExtendableAlgebra, HasTwoAdicBinomialExtension,
    HasTwoAdicQuinticExtension, QuinticExtendableAlgebra, QuinticTrinomialExtendable,
};
use p3_field::{
    PrimeCharacteristicRing, TwoAdicField, field_to_array, packed_mod_add, packed_mod_sub,
};

use crate::utils::{add, sub};
use crate::{
    BinomialExtensionData, FieldParameters, MontyField31, TrinomialQuinticData, TwoAdicData,
    base_mul_packed, octic_mul_packed, quartic_mul_packed, quintic_mul_packed,
};

// If a field implements BinomialExtensionData<WIDTH> then there is a natural
// field extension of degree WIDTH we can define.
// We perform no checks to make sure the data given in BinomialExtensionData<WIDTH> is valid and
// corresponds to an actual field extension. Ensuring that is left to the implementer.

impl<const WIDTH: usize, FP> BinomiallyExtendableAlgebra<Self, WIDTH> for MontyField31<FP>
where
    FP: BinomialExtensionData<WIDTH> + FieldParameters,
{
    #[inline(always)]
    fn binomial_mul(a: &[Self; WIDTH], b: &[Self; WIDTH], res: &mut [Self; WIDTH], _w: Self) {
        match WIDTH {
            4 => quartic_mul_packed(a, b, res),
            5 => quintic_mul_packed(a, b, res),
            8 => octic_mul_packed(a, b, res),
            _ => panic!("Unsupported binomial extension degree: {}", WIDTH),
        }
    }

    #[inline(always)]
    fn binomial_add(a: &[Self; WIDTH], b: &[Self; WIDTH]) -> [Self; WIDTH] {
        let mut res = [Self::ZERO; WIDTH];
        unsafe {
            // Safe as Self is repr(transparent) and stores a single u32.
            let a: &[u32; WIDTH] = &*(a.as_ptr() as *const [u32; WIDTH]);
            let b: &[u32; WIDTH] = &*(b.as_ptr() as *const [u32; WIDTH]);
            let res: &mut [u32; WIDTH] = &mut *(res.as_mut_ptr() as *mut [u32; WIDTH]);

            packed_mod_add(a, b, res, FP::PRIME, add::<FP>);
        }
        res
    }

    #[inline(always)]
    fn binomial_sub(a: &[Self; WIDTH], b: &[Self; WIDTH]) -> [Self; WIDTH] {
        let mut res = [Self::ZERO; WIDTH];
        unsafe {
            // Safe as Self is repr(transparent) and stores a single u32.
            let a: &[u32; WIDTH] = &*(a.as_ptr() as *const [u32; WIDTH]);
            let b: &[u32; WIDTH] = &*(b.as_ptr() as *const [u32; WIDTH]);
            let res: &mut [u32; WIDTH] = &mut *(res.as_mut_ptr() as *mut [u32; WIDTH]);

            packed_mod_sub(a, b, res, FP::PRIME, sub::<FP>);
        }
        res
    }

    #[inline(always)]
    fn binomial_base_mul(lhs: [Self; WIDTH], rhs: Self) -> [Self; WIDTH] {
        let mut res = [Self::ZERO; WIDTH];
        base_mul_packed(lhs, rhs, &mut res);
        res
    }
}

impl<const WIDTH: usize, FP> BinomiallyExtendable<WIDTH> for MontyField31<FP>
where
    FP: BinomialExtensionData<WIDTH> + FieldParameters,
{
    const W: Self = FP::W;

    const DTH_ROOT: Self = FP::DTH_ROOT;

    const EXT_GENERATOR: [Self; WIDTH] = FP::EXT_GENERATOR;
}

impl<const WIDTH: usize, FP> HasTwoAdicBinomialExtension<WIDTH> for MontyField31<FP>
where
    FP: BinomialExtensionData<WIDTH> + TwoAdicData + FieldParameters,
{
    const EXT_TWO_ADICITY: usize = FP::EXT_TWO_ADICITY;

    fn ext_two_adic_generator(bits: usize) -> [Self; WIDTH] {
        assert!(bits <= Self::EXT_TWO_ADICITY);
        if bits <= FP::TWO_ADICITY {
            field_to_array(Self::two_adic_generator(bits))
        } else {
            FP::TWO_ADIC_EXTENSION_GENERATORS.as_ref()[bits - FP::TWO_ADICITY - 1]
        }
    }
}

impl<FP> QuinticExtendableAlgebra<Self> for MontyField31<FP> where
    FP: TrinomialQuinticData + FieldParameters
{
}

impl<FP> QuinticTrinomialExtendable for MontyField31<FP>
where
    FP: TrinomialQuinticData + FieldParameters,
{
    const FROBENIUS_COEFFS: [[Self; 5]; 4] = FP::FROBENIUS_COEFFS;

    const EXT_GENERATOR: [Self; 5] = FP::EXT_GENERATOR;
}

impl<FP> HasTwoAdicQuinticExtension for MontyField31<FP>
where
    FP: TrinomialQuinticData + TwoAdicData + FieldParameters,
{
    const EXT_TWO_ADICITY: usize = FP::EXT_TWO_ADICITY;

    type ArrayLike = <FP as TrinomialQuinticData>::ArrayLike;
    const TWO_ADIC_EXTENSION_GENERATORS: Self::ArrayLike =
        <FP as TrinomialQuinticData>::TWO_ADIC_EXTENSION_GENERATORS;

    fn ext_two_adic_generator(bits: usize) -> [Self; 5] {
        assert!(bits <= Self::EXT_TWO_ADICITY);
        if bits <= FP::TWO_ADICITY {
            field_to_array(Self::two_adic_generator(bits))
        } else {
            FP::TWO_ADIC_EXTENSION_GENERATORS.as_ref()[bits - FP::TWO_ADICITY - 1]
        }
    }
}
