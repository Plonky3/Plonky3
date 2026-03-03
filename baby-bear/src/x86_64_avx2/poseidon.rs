use p3_monty_31::PartialRoundParametersAVX2;

use crate::{BabyBearParameters, BabyBearPoseidonParameters};

impl PartialRoundParametersAVX2<BabyBearParameters, 16> for BabyBearPoseidonParameters {}
impl PartialRoundParametersAVX2<BabyBearParameters, 24> for BabyBearPoseidonParameters {}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::Permutation;
    use proptest::prelude::*;

    use crate::poseidon::{default_babybear_poseidon_16, default_babybear_poseidon_24};
    use crate::{BabyBear, PackedBabyBearAVX2};

    type F = BabyBear;

    fn arb_f() -> impl Strategy<Value = F> {
        prop::num::u32::ANY.prop_map(F::from_u32)
    }

    proptest! {
        #[test]
        fn poseidon_avx2_matches_scalar_width_16(
            input in prop::array::uniform16(arb_f())
        ) {
            let perm = default_babybear_poseidon_16();

            let mut expected = input;
            perm.permute_mut(&mut expected);

            let mut avx2_input = input.map(Into::<PackedBabyBearAVX2>::into);
            perm.permute_mut(&mut avx2_input);
            let avx2_output = avx2_input.map(|x| x.0[0]);

            prop_assert_eq!(avx2_output, expected);
        }

        #[test]
        fn poseidon_avx2_matches_scalar_width_24(
            input in prop::array::uniform24(arb_f())
        ) {
            let perm = default_babybear_poseidon_24();

            let mut expected = input;
            perm.permute_mut(&mut expected);

            let mut avx2_input = input.map(Into::<PackedBabyBearAVX2>::into);
            perm.permute_mut(&mut avx2_input);
            let avx2_output = avx2_input.map(|x| x.0[0]);

            prop_assert_eq!(avx2_output, expected);
        }
    }
}
