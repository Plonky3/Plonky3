use p3_field::{exp_1725656503, exp_u64_by_squaring, AbstractField, Field};
use p3_monty_31_bit_field::{
    BinomialExtensionData, FieldConstants, FieldParameters, MontyField31, MontyParameters,
    TwoAdicData,
};

// We are defining MU = P^-1 (mod 2^MONTY_BITS). This is different from the usual convention
// (MU = -P^-1 (mod 2^MONTY_BITS)) but it avoids a carry.

/// The prime field `2^31 - 2^27 + 1`, a.k.a. the Baby Bear field.
pub type BabyBear = MontyField31<BabyBearParameters>;

#[derive(Copy, Clone, Default, Eq, Hash, PartialEq)]
pub struct BabyBearParameters;

impl FieldParameters for BabyBearParameters {}

impl MontyParameters for BabyBearParameters {
    /// The Baby Bear prime
    /// This is the unique 31-bit prime with the highest possible 2 adicity (27).
    const PRIME: u32 = 0x78000001;

    const MONTY_BITS: u32 = 32;
    const MONTY_MU: u32 = 0x88000001;
}

impl FieldConstants for BabyBearParameters {
    fn exp_u64_generic<AF: AbstractField>(val: AF, power: u64) -> AF {
        match power {
            1725656503 => exp_1725656503(val), // used to compute x^{1/7}
            _ => exp_u64_by_squaring(val, power),
        }
    }

    fn try_inverse<F: Field>(p1: F) -> Option<F> {
        if p1.is_zero() {
            return None;
        }

        // From Fermat's little theorem, in a prime field `F_p`, the inverse of `a` is `a^(p-2)`.
        // Here p-2 = 2013265919 = 1110111111111111111111111111111_2.
        // Uses 30 Squares + 7 Multiplications => 37 Operations total.

        let p100000000 = p1.exp_power_of_2(8);
        let p100000001 = p100000000 * p1;
        let p10000000000000000 = p100000000.exp_power_of_2(8);
        let p10000000100000001 = p10000000000000000 * p100000001;
        let p10000000100000001000 = p10000000100000001.exp_power_of_2(3);
        let p1000000010000000100000000 = p10000000100000001000.exp_power_of_2(5);
        let p1000000010000000100000001 = p1000000010000000100000000 * p1;
        let p1000010010000100100001001 = p1000000010000000100000001 * p10000000100000001000;
        let p10000000100000001000000010 = p1000000010000000100000001.square();
        let p11000010110000101100001011 = p10000000100000001000000010 * p1000010010000100100001001;
        let p100000001000000010000000100 = p10000000100000001000000010.square();
        let p111000011110000111100001111 =
            p100000001000000010000000100 * p11000010110000101100001011;
        let p1110000111100001111000011110000 = p111000011110000111100001111.exp_power_of_2(4);
        let p1110111111111111111111111111111 =
            p1110000111100001111000011110000 * p111000011110000111100001111;

        Some(p1110111111111111111111111111111)
    }

    const GEN: u32 = 31;
}

const TWO_ADIC_GENERATORS: [u32; 28] = [
    0x1, 0x78000000, 0x67055c21, 0x5ee99486, 0xbb4c4e4, 0x2d4cc4da, 0x669d6090, 0x17b56c64,
    0x67456167, 0x688442f9, 0x145e952d, 0x4fe61226, 0x4c734715, 0x11c33e2a, 0x62c3d2b1, 0x77cad399,
    0x54c131f4, 0x4cabd6a6, 0x5cf5713f, 0x3e9430e8, 0xba067a3, 0x18adc27d, 0x21fd55bc, 0x4b859b3d,
    0x3bd57996, 0x4483d85a, 0x3a26eef8, 0x1a427a41,
];

impl TwoAdicData for BabyBearParameters {
    const TWO_ADICITY: usize = 27;

    fn u32_two_adic_generator(bits: usize) -> u32 {
        assert!(bits <= Self::TWO_ADICITY);

        TWO_ADIC_GENERATORS[bits]
    }
}

const EXT_TWO_ADIC_GENERATORS4: [[u32; 4]; 2] = [[0, 0, 1996171314, 0], [0, 0, 0, 124907976]];

impl BinomialExtensionData for BabyBearParameters {
    const W4: u32 = 11;
    const DTH_ROOT4: u32 = 1728404513;
    const EXT_GENERATOR_4: [u32; 4] = [8, 1, 0, 0];
    const EXT_TWO_ADICITY4: usize = 29;
    fn u32_ext_two_adic_generator4(bits: usize) -> [u32; 4] {
        assert!(bits <= Self::EXT_TWO_ADICITY4);
        if bits > Self::TWO_ADICITY {
            EXT_TWO_ADIC_GENERATORS4[bits - Self::TWO_ADICITY - 1]
        } else {
            [TWO_ADIC_GENERATORS[bits], 0, 0, 0]
        }
    }

    const W5: u32 = 2;
    const DTH_ROOT5: u32 = 815036133;
    const EXT_GENERATOR_5: [u32; 5] = [8, 1, 0, 0, 0];
    const EXT_TWO_ADICITY5: usize = 27;

    fn u32_ext_two_adic_generator5(bits: usize) -> [u32; 5] {
        assert!(bits <= Self::EXT_TWO_ADICITY5);

        [TWO_ADIC_GENERATORS[bits], 0, 0, 0, 0]
    }
}

#[cfg(test)]
mod tests {
    use core::array;

    use p3_field::{AbstractField, Field, PrimeField32, PrimeField64, TwoAdicField};
    use p3_field_testing::{test_field, test_two_adic_field};
    use p3_monty_31_bit_field::to_monty_array;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_baby_bear_two_adicity_generators() {
        let base = BabyBear::from_canonical_u32(0x1a427a41);
        for bits in 0..=BabyBear::TWO_ADICITY {
            assert_eq!(
                BabyBear::two_adic_generator(bits),
                base.exp_power_of_2(BabyBear::TWO_ADICITY - bits)
            );
        }
    }

    #[test]
    fn test_to_babybear_array() {
        let range_array: [u32; 32] = array::from_fn(|i| i as u32);
        assert_eq!(
            to_monty_array::<32, BabyBearParameters>(range_array),
            range_array.map(F::from_canonical_u32)
        )
    }

    #[test]
    fn test_baby_bear() {
        let f = F::from_canonical_u32(100);
        assert_eq!(f.as_canonical_u64(), 100);

        let f = F::from_canonical_u32(0);
        assert!(f.is_zero());

        let f = F::from_wrapped_u32(F::ORDER_U32);
        assert!(f.is_zero());

        let f_1 = F::one();
        let f_1_copy = F::from_canonical_u32(1);

        let expected_result = F::zero();
        assert_eq!(f_1 - f_1_copy, expected_result);

        let expected_result = F::two();
        assert_eq!(f_1 + f_1_copy, expected_result);

        let f_2 = F::from_canonical_u32(2);
        let expected_result = F::from_canonical_u32(3);
        assert_eq!(f_1 + f_1_copy * f_2, expected_result);

        let expected_result = F::from_canonical_u32(5);
        assert_eq!(f_1 + f_2 * f_2, expected_result);

        let f_p_minus_1 = F::from_canonical_u32(F::ORDER_U32 - 1);
        let expected_result = F::zero();
        assert_eq!(f_1 + f_p_minus_1, expected_result);

        let f_p_minus_2 = F::from_canonical_u32(F::ORDER_U32 - 2);
        let expected_result = F::from_canonical_u32(F::ORDER_U32 - 3);
        assert_eq!(f_p_minus_1 + f_p_minus_2, expected_result);

        let expected_result = F::from_canonical_u32(1);
        assert_eq!(f_p_minus_1 - f_p_minus_2, expected_result);

        let expected_result = f_p_minus_1;
        assert_eq!(f_p_minus_2 - f_p_minus_1, expected_result);

        let expected_result = f_p_minus_2;
        assert_eq!(f_p_minus_1 - f_1, expected_result);

        let m1 = F::from_canonical_u32(0x34167c58);
        let m2 = F::from_canonical_u32(0x61f3207b);
        let expected_prod = F::from_canonical_u32(0x1b5c8046);
        assert_eq!(m1 * m2, expected_prod);

        assert_eq!(m1.exp_u64(1725656503).exp_const_u64::<7>(), m1);
        assert_eq!(m2.exp_u64(1725656503).exp_const_u64::<7>(), m2);
        assert_eq!(f_2.exp_u64(1725656503).exp_const_u64::<7>(), f_2);

        let f_serialized = serde_json::to_string(&f).unwrap();
        let f_deserialized: F = serde_json::from_str(&f_serialized).unwrap();
        assert_eq!(f, f_deserialized);

        let f_1_serialized = serde_json::to_string(&f_1).unwrap();
        let f_1_deserialized: F = serde_json::from_str(&f_1_serialized).unwrap();
        let f_1_serialized_again = serde_json::to_string(&f_1_deserialized).unwrap();
        let f_1_deserialized_again: F = serde_json::from_str(&f_1_serialized_again).unwrap();
        assert_eq!(f_1, f_1_deserialized);
        assert_eq!(f_1, f_1_deserialized_again);

        let f_2_serialized = serde_json::to_string(&f_2).unwrap();
        let f_2_deserialized: F = serde_json::from_str(&f_2_serialized).unwrap();
        assert_eq!(f_2, f_2_deserialized);

        let f_p_minus_1_serialized = serde_json::to_string(&f_p_minus_1).unwrap();
        let f_p_minus_1_deserialized: F = serde_json::from_str(&f_p_minus_1_serialized).unwrap();
        assert_eq!(f_p_minus_1, f_p_minus_1_deserialized);

        let f_p_minus_2_serialized = serde_json::to_string(&f_p_minus_2).unwrap();
        let f_p_minus_2_deserialized: F = serde_json::from_str(&f_p_minus_2_serialized).unwrap();
        assert_eq!(f_p_minus_2, f_p_minus_2_deserialized);

        let m1_serialized = serde_json::to_string(&m1).unwrap();
        let m1_deserialized: F = serde_json::from_str(&m1_serialized).unwrap();
        assert_eq!(m1, m1_deserialized);

        let m2_serialized = serde_json::to_string(&m2).unwrap();
        let m2_deserialized: F = serde_json::from_str(&m2_serialized).unwrap();
        assert_eq!(m2, m2_deserialized);
    }

    test_field!(crate::BabyBear);
    test_two_adic_field!(crate::BabyBear);
}
