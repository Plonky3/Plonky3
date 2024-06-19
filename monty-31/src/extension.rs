use p3_field::extension::{BinomiallyExtendable, HasTwoAdicBionmialExtension};
use p3_field::{AbstractField, TwoAdicField};

use crate::{BinomialExtensionData, FieldParameters, MontyField31, TwoAdicData};

// One clear issue for defining extensions is that, depending on the field order,
// Binomial extensions of a given size may or may not exist.
// For now we set up the code to handle allow for Binomials extensions of degree 4 and 5.

impl<const WIDTH: usize, FP> BinomiallyExtendable<WIDTH> for MontyField31<FP>
where
    FP: BinomialExtensionData<WIDTH> + FieldParameters,
{
    fn w() -> Self {
        <FP as BinomialExtensionData<WIDTH>>::W
    }
    fn dth_root() -> Self {
        <FP as BinomialExtensionData<WIDTH>>::DTH_ROOT
    }
    fn ext_generator() -> [Self; WIDTH] {
        FP::EXT_GENERATOR
    }
}

impl<const WIDTH: usize, FP> HasTwoAdicBionmialExtension<WIDTH> for MontyField31<FP>
where
    FP: BinomialExtensionData<WIDTH> + TwoAdicData + FieldParameters,
{
    const EXT_TWO_ADICITY: usize = <FP as BinomialExtensionData<WIDTH>>::EXT_TWO_ADICITY;

    fn ext_two_adic_generator(bits: usize) -> [Self; WIDTH] {
        assert!(bits <= Self::EXT_TWO_ADICITY);
        if bits <= FP::TWO_ADICITY {
            // TODO: Probably a cleaner way to do this?
            let mut output = [Self::zero(); WIDTH];
            output[0] = Self::two_adic_generator(bits);
            output
        } else {
            FP::TWO_ADIC_EXTENSION_GENERATORS.as_ref()[bits - FP::TWO_ADICITY - 1]
        }
    }
}
