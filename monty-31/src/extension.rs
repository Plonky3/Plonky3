use p3_field::extension::{BinomiallyExtendable, HasTwoAdicBionmialExtension};
use p3_field::{field_to_array, TwoAdicField};

use crate::{BinomialExtensionData, FieldParameters, MontyField31, TwoAdicData};

// If a field implements BinomialExtensionData<WIDTH> then there is a natural
// field extension of degree WIDTH we can define.
// We perform no checks to make sure the data given in BinomialExtensionData<WIDTH> is valid and
// corresponds to an actual field extension. Ensuring that is left to the implementor.

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
            field_to_array(Self::two_adic_generator(bits))
        } else {
            FP::TWO_ADIC_EXTENSION_GENERATORS.as_ref()[bits - FP::TWO_ADICITY - 1]
        }
    }
}
