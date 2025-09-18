use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::{KoalaBear, QuinticExtendable, QuinticExtensionFieldKB};

#[test]
fn test_frobenius_matrix_quintic_koala_bear() {
    for i in 1..5 {
        let mut x = QuinticExtensionFieldKB::ZERO;
        x.value[i] = KoalaBear::ONE;
        let x = x.exp_u64(KoalaBear::ORDER_U64);
        for j in 0..5 {
            assert_eq!(
                x.value[j],
                <KoalaBear as QuinticExtendable>::FROBENIUS_MATRIX[i - 1][j]
            )
        }
    }
}
