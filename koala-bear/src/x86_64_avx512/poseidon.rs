use p3_monty_31::PartialRoundParametersAVX512;

use crate::{KoalaBearParameters, KoalaBearPoseidonParameters};

impl PartialRoundParametersAVX512<KoalaBearParameters, 16> for KoalaBearPoseidonParameters {}
impl PartialRoundParametersAVX512<KoalaBearParameters, 24> for KoalaBearPoseidonParameters {}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::Permutation;
    use proptest::prelude::*;

    use crate::poseidon::{default_koalabear_poseidon_16, default_koalabear_poseidon_24};
    use crate::{KoalaBear, PackedKoalaBearAVX512};

    type F = KoalaBear;

    fn arb_f() -> impl Strategy<Value = F> {
        prop::num::u32::ANY.prop_map(F::from_u32)
    }

    proptest! {
        #[test]
        fn poseidon_avx512_matches_scalar_width_16(
            input in prop::array::uniform16(arb_f())
        ) {
            let perm = default_koalabear_poseidon_16();

            let mut expected = input;
            perm.permute_mut(&mut expected);

            let mut avx512_input = input.map(Into::<PackedKoalaBearAVX512>::into);
            perm.permute_mut(&mut avx512_input);
            let avx512_output = avx512_input.map(|x| x.0[0]);

            prop_assert_eq!(avx512_output, expected);
        }

        #[test]
        fn poseidon_avx512_matches_scalar_width_24(
            input in prop::array::uniform24(arb_f())
        ) {
            let perm = default_koalabear_poseidon_24();

            let mut expected = input;
            perm.permute_mut(&mut expected);

            let mut avx512_input = input.map(Into::<PackedKoalaBearAVX512>::into);
            perm.permute_mut(&mut avx512_input);
            let avx512_output = avx512_input.map(|x| x.0[0]);

            prop_assert_eq!(avx512_output, expected);
        }
    }
}
