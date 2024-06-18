use p3_field::extension::{BinomiallyExtendable, HasTwoAdicBionmialExtension};
use p3_field::AbstractField;

use crate::{BinomialExtensionData, FieldParameters, MontyField31};

// One clear issue for defining extensions is that, depending on the field order,
// Binomial extensions of a given size may or may not exist.
// For now we set up the code to handle allow for Binomials extensions of degree 4 and 5.

impl<const WIDTH: usize, FP> BinomiallyExtendable<WIDTH> for MontyField31<FP>
where
    FP: BinomialExtensionData<WIDTH> + FieldParameters,
{
    fn w() -> Self {
        Self::new_monty(<FP as BinomialExtensionData<WIDTH>>::MONTY_W)
    }
    fn dth_root() -> Self {
        Self::new_monty(<FP as BinomialExtensionData<WIDTH>>::MONTY_DTH_ROOT)
    }
    fn ext_generator() -> [Self; WIDTH] {
        FP::MONTY_EXT_GENERATOR.map(Self::new_monty)
    }
}

impl<const WIDTH: usize, FP> HasTwoAdicBionmialExtension<WIDTH> for MontyField31<FP>
where
    FP: BinomialExtensionData<WIDTH> + FieldParameters,
{
    const EXT_TWO_ADICITY: usize = <FP as BinomialExtensionData<WIDTH>>::EXT_TWO_ADICITY;

    fn ext_two_adic_generator(bits: usize) -> [Self; WIDTH] {
        assert!(bits <= <FP as BinomialExtensionData<WIDTH>>::EXT_TWO_ADICITY);
        FP::u32_ext_two_adic_generator(bits).map(Self::from_canonical_u32)
    }
}
