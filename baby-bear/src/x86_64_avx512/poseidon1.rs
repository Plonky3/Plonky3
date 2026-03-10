use p3_monty_31::PartialRoundParametersAVX512;

use crate::{BabyBearParameters, BabyBearPoseidonParameters};

impl PartialRoundParametersAVX512<BabyBearParameters, 16> for BabyBearPoseidonParameters {}
impl PartialRoundParametersAVX512<BabyBearParameters, 24> for BabyBearPoseidonParameters {}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::Permutation;
    use proptest::prelude::*;

    use crate::poseidon1::{default_babybear_poseidon1_16, default_babybear_poseidon1_24};
    use crate::{BabyBear, PackedBabyBearAVX512};

    type F = BabyBear;

    fn arb_f() -> impl Strategy<Value = F> {
        prop::num::u32::ANY.prop_map(F::from_u32)
    }

    proptest! {
        #[test]
        fn poseidon_avx512_matches_scalar_width_16(
            input in prop::array::uniform16(arb_f())
        ) {
            let perm = default_babybear_poseidon1_16();

            let mut expected = input;
            perm.permute_mut(&mut expected);

            let mut avx512_input = input.map(Into::<PackedBabyBearAVX512>::into);
            perm.permute_mut(&mut avx512_input);
            let avx512_output = avx512_input.map(|x| x.0[0]);

            prop_assert_eq!(avx512_output, expected);
        }

        #[test]
        fn poseidon_avx512_matches_scalar_width_24(
            input in prop::array::uniform24(arb_f())
        ) {
            let perm = default_babybear_poseidon1_24();

            let mut expected = input;
            perm.permute_mut(&mut expected);

            let mut avx512_input = input.map(Into::<PackedBabyBearAVX512>::into);
            perm.permute_mut(&mut avx512_input);
            let avx512_output = avx512_input.map(|x| x.0[0]);

            prop_assert_eq!(avx512_output, expected);
        }
    }
}
