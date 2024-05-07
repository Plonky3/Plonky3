use p3_field::extension::{BinomiallyExtendable, HasTwoAdicBionmialExtension};
use p3_field::AbstractField;

use crate::{BinomialExtensionData, FieldParameters, MontyField31};

// One clear issue for defining extensions is that, depending on the field order,
// Binomial extensions of a given size may or may not exist.
// For now we set up the code to handle allow for Binomials extensions of degree 4 and 5.

impl<FP: FieldParameters> BinomiallyExtendable<4> for MontyField31<FP> {
    fn w() -> Self {
        Self::new_monty(<FP as BinomialExtensionData<4>>::MONTY_W)
    }
    fn dth_root() -> Self {
        Self::new_monty(<FP as BinomialExtensionData<4>>::MONTY_DTH_ROOT)
    }
    fn ext_generator() -> [Self; 4] {
        FP::MONTY_EXT_GENERATOR.map(Self::new_monty)
    }
}

impl<FP: FieldParameters> HasTwoAdicBionmialExtension<4> for MontyField31<FP> {
    const EXT_TWO_ADICITY: usize = <FP as BinomialExtensionData<4>>::EXT_TWO_ADICITY;

    fn ext_two_adic_generator(bits: usize) -> [Self; 4] {
        assert!(bits <= <FP as BinomialExtensionData<4>>::EXT_TWO_ADICITY);
        FP::u32_ext_two_adic_generator(bits).map(Self::from_canonical_u32)
    }
}

impl<FP: FieldParameters> BinomiallyExtendable<5> for MontyField31<FP> {
    fn w() -> Self {
        Self::new_monty(<FP as BinomialExtensionData<5>>::MONTY_W)
    }
    fn dth_root() -> Self {
        Self::new_monty(<FP as BinomialExtensionData<5>>::MONTY_DTH_ROOT)
    }
    fn ext_generator() -> [Self; 5] {
        FP::MONTY_EXT_GENERATOR.map(Self::new_monty)
    }
}

impl<FP: FieldParameters> HasTwoAdicBionmialExtension<5> for MontyField31<FP> {
    const EXT_TWO_ADICITY: usize = <FP as BinomialExtensionData<5>>::EXT_TWO_ADICITY;

    fn ext_two_adic_generator(bits: usize) -> [Self; 5] {
        assert!(bits <= <FP as BinomialExtensionData<5>>::EXT_TWO_ADICITY);
        FP::u32_ext_two_adic_generator(bits).map(Self::from_canonical_u32)
    }
}
