use p3_monty_31::PartialRoundParametersAVX2;

use crate::{KoalaBearParameters, KoalaBearPoseidonParameters};

impl PartialRoundParametersAVX2<KoalaBearParameters, 16> for KoalaBearPoseidonParameters {}
impl PartialRoundParametersAVX2<KoalaBearParameters, 24> for KoalaBearPoseidonParameters {}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::Permutation;
    use proptest::prelude::*;

    use crate::poseidon1::{default_koalabear_poseidon1_16, default_koalabear_poseidon1_24};
    use crate::{KoalaBear, PackedKoalaBearAVX2};

    type F = KoalaBear;

    fn arb_f() -> impl Strategy<Value = F> {
        prop::num::u32::ANY.prop_map(F::from_u32)
    }

    proptest! {
        #[test]
        fn poseidon_avx2_matches_scalar_width_16(
            input in prop::array::uniform16(arb_f())
        ) {
            let perm = default_koalabear_poseidon1_16();

            let mut expected = input;
            perm.permute_mut(&mut expected);

            let mut avx2_input = input.map(Into::<PackedKoalaBearAVX2>::into);
            perm.permute_mut(&mut avx2_input);
            let avx2_output = avx2_input.map(|x| x.0[0]);

            prop_assert_eq!(avx2_output, expected);
        }

        #[test]
        fn poseidon_avx2_matches_scalar_width_24(
            input in prop::array::uniform24(arb_f())
        ) {
            let perm = default_koalabear_poseidon1_24();

            let mut expected = input;
            perm.permute_mut(&mut expected);

            let mut avx2_input = input.map(Into::<PackedKoalaBearAVX2>::into);
            perm.permute_mut(&mut avx2_input);
            let avx2_output = avx2_input.map(|x| x.0[0]);

            prop_assert_eq!(avx2_output, expected);
        }
    }
}
