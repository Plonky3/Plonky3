use p3_monty_31::PartialRoundParametersAVX2;

use crate::{BabyBearParameters, BabyBearPoseidonParameters};

impl PartialRoundParametersAVX2<BabyBearParameters, 16> for BabyBearPoseidonParameters {}
impl PartialRoundParametersAVX2<BabyBearParameters, 24> for BabyBearPoseidonParameters {}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::Permutation;
    use proptest::prelude::*;

    use crate::poseidon1::{default_babybear_poseidon1_16, default_babybear_poseidon1_24};
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
            let perm = default_babybear_poseidon1_16();

            let mut packed_input = core::array::from_fn(|i| {
                let mut packed = PackedBabyBearAVX2::ZERO;
                for lane in 0..packed.0.len() {
                    packed.0[lane] = input[i] + F::from_u32((lane + 1) as u32);
                }
                packed
            });
            perm.permute_mut(&mut packed_input);

            for lane in 0..packed_input[0].0.len() {
                let mut expected = input.map(|x| x + F::from_u32((lane + 1) as u32));
                perm.permute_mut(&mut expected);
                let packed_output = packed_input.map(|x| x.0[lane]);

                prop_assert_eq!(packed_output, expected, "lane {} mismatch", lane);
            }
        }

        #[test]
        fn poseidon_avx2_matches_scalar_width_24(
            input in prop::array::uniform24(arb_f())
        ) {
            let perm = default_babybear_poseidon1_24();

            let mut packed_input = core::array::from_fn(|i| {
                let mut packed = PackedBabyBearAVX2::ZERO;
                for lane in 0..packed.0.len() {
                    packed.0[lane] = input[i] + F::from_u32((lane + 1) as u32);
                }
                packed
            });
            perm.permute_mut(&mut packed_input);

            for lane in 0..packed_input[0].0.len() {
                let mut expected = input.map(|x| x + F::from_u32((lane + 1) as u32));
                perm.permute_mut(&mut expected);
                let packed_output = packed_input.map(|x| x.0[lane]);

                prop_assert_eq!(packed_output, expected, "lane {} mismatch", lane);
            }
        }
    }
}
