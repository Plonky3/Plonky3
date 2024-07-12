use p3_field::{exp_1725656503, exp_u64_by_squaring, AbstractField, Field};
use p3_monty_31::{
    BarrettParameters, BinomialExtensionData, FieldParameters, MontyField31, MontyParameters,
    PackedMontyParameters, TwoAdicData,
};

/// The prime field `2^31 - 2^27 + 1`, a.k.a. the Baby Bear field.
pub type BabyBear = MontyField31<BabyBearParameters>;

#[derive(Copy, Clone, Default, Debug, Eq, Hash, PartialEq)]
pub struct BabyBearParameters;

impl MontyParameters for BabyBearParameters {
    /// The Baby Bear prime: 2^31 - 2^27 + 1.
    /// This is the unique 31-bit prime with the highest possible 2 adicity (27).
    const PRIME: u32 = 0x78000001;

    const MONTY_BITS: u32 = 32;
    const MONTY_MU: u32 = 0x88000001;
}

impl PackedMontyParameters for BabyBearParameters {}

impl BarrettParameters for BabyBearParameters {}

impl FieldParameters for BabyBearParameters {
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

    const MONTY_GEN: BabyBear = BabyBear::new(31);
}

impl TwoAdicData for BabyBearParameters {
    const TWO_ADICITY: usize = 27;

    type ArrayLike = [BabyBear; Self::TWO_ADICITY + 1];

    const TWO_ADIC_GENERATORS: Self::ArrayLike = BabyBear::new_array([
        0x1, 0x78000000, 0x67055c21, 0x5ee99486, 0xbb4c4e4, 0x2d4cc4da, 0x669d6090, 0x17b56c64,
        0x67456167, 0x688442f9, 0x145e952d, 0x4fe61226, 0x4c734715, 0x11c33e2a, 0x62c3d2b1,
        0x77cad399, 0x54c131f4, 0x4cabd6a6, 0x5cf5713f, 0x3e9430e8, 0xba067a3, 0x18adc27d,
        0x21fd55bc, 0x4b859b3d, 0x3bd57996, 0x4483d85a, 0x3a26eef8, 0x1a427a41,
    ]);
}

impl BinomialExtensionData<4> for BabyBearParameters {
    const W: BabyBear = BabyBear::new(11);
    const DTH_ROOT: BabyBear = BabyBear::new(1728404513);
    const EXT_GENERATOR: [BabyBear; 4] = BabyBear::new_array([8, 1, 0, 0]);
    const EXT_TWO_ADICITY: usize = 29;

    type ArrayLike = [[BabyBear; 4]; 2];
    const TWO_ADIC_EXTENSION_GENERATORS: Self::ArrayLike =
        BabyBear::new_2d_array([[0, 0, 1996171314, 0], [0, 0, 0, 124907976]]);
}

impl BinomialExtensionData<5> for BabyBearParameters {
    const W: BabyBear = BabyBear::new(2);
    const DTH_ROOT: BabyBear = BabyBear::new(815036133);
    const EXT_GENERATOR: [BabyBear; 5] = BabyBear::new_array([8, 1, 0, 0, 0]);
    const EXT_TWO_ADICITY: usize = 27;

    type ArrayLike = [[BabyBear; 5]; 0];
    const TWO_ADIC_EXTENSION_GENERATORS: Self::ArrayLike = [];
}

#[cfg(test)]
mod tests {
    use core::array;

    use p3_field::{AbstractField, Field, PrimeField32, PrimeField64, TwoAdicField};
    use p3_field_testing::{test_field, test_two_adic_field};

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
            BabyBear::new_array(range_array),
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

    // TODO: Refactor these FFT tests with macros as for test_field! etc above.
    use p3_dft::TwoAdicSubgroupDft;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_monty_31::dft::Radix2Dit;
    use rand::distributions::{Distribution, Standard};
    use rand::{thread_rng, Rng};

    fn _naive_convolve(us: &[BabyBear], vs: &[BabyBear]) -> Vec<BabyBear> {
        let n = us.len();
        assert_eq!(n, vs.len());

        let mut conv = Vec::with_capacity(n);
        for i in 0..n {
            let mut t = BabyBear::zero();
            for j in 0..n {
                t = t + us[j] * vs[(n + i - j) % n];
            }
            conv.push(t);
        }
        conv
    }

    fn randvec(n: usize) -> Vec<BabyBear>
    where
        Standard: Distribution<BabyBear>,
    {
        thread_rng()
            .sample_iter(Standard)
            .take(n)
            .collect::<Vec<_>>()
    }

    /*
    #[test]
    fn test_forward_16() {
        const NITERS: usize = 100;
        let len = 16;
        let root_table = BabyBear::roots_of_unity_table(len);

        for _ in 0..NITERS {
            let us = randvec(len);
            /*
            //let us = vec![0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
            // monty form of [0..16)
            let us = vec![
                0, 268435454, 536870908, 805306362, 1073741816, 1342177270, 1610612724, 1879048178,
                134217711, 402653165, 671088619, 939524073, 1207959527, 1476394981, 1744830435,
                2013265889,
            ];
            */

            let mut vs = us.clone();
            BabyBear::forward_fft(&mut vs, &root_table);
            reverse_slice_index_bits(&mut vs);

            let mut ws = us.clone();
            BabyBear::four_step_fft(&mut ws, &root_table);

            assert!(vs.iter().zip(ws).all(|(&v, w)| v == w));
        }
    }
    */

    #[test]
    fn forward_backward_is_identity() {
        const NITERS: usize = 100;
        let mut len = 16;
        loop {
            let dft = Radix2Dit::default();

            let inv_root_table = BabyBear::inv_roots_of_unity_table(len);
            let root_inv = inv_root_table[0][0];

            for _ in 0..NITERS {
                let us = randvec(len);
                let vs = RowMajorMatrix::new_col(us.clone());
                let vs = dft.dft_batch(vs);

                // FIXME: Need this for four-step
                //p3_util::reverse_slice_index_bits(&mut vs);

                let mut ws = vs.values.clone();
                BabyBear::backward_fft(&mut ws, root_inv);

                let scale = BabyBear::new(len as u32);
                let scaled_us = us.iter().map(|&u| u * scale).collect::<Vec<_>>();
                assert_eq!(scaled_us, ws);
            }
            len *= 2;
            if len > 8192 {
                break;
            }
        }
    }

    /*
    #[test]
    fn convolution() {
        const NITERS: usize = 4;
        let mut len = 4;
        loop {
            let root_table = BabyBear::roots_of_unity_table(len);
            let inv_root_table = BabyBear::inv_roots_of_unity_table(len);
            let root_inv = inv_root_table[0][0];

            for _ in 0..NITERS {
                let us = randvec(len);
                let vs = randvec(len);

                let mut fft_us = us.clone();
                BabyBear::forward_fft(&mut fft_us, &root_table);

                let mut fft_vs = vs.clone();
                BabyBear::forward_fft(&mut fft_vs, &root_table);

                let mut pt_prods = fft_us
                    .iter()
                    .zip(fft_vs)
                    .map(|(&u, v)| {
                        let prod = BabyBear { value: u } * BabyBear { value: v };
                        prod.value
                    })
                    .collect::<Vec<_>>();

                backward_fft(&mut pt_prods, root_inv);

                let bus = us
                    .iter()
                    .map(|&u| BabyBear { value: u })
                    .collect::<Vec<_>>();
                let bvs = vs
                    .iter()
                    .map(|&v| BabyBear { value: v })
                    .collect::<Vec<_>>();
                let bconv = naive_convolve(&bus, &bvs);
                let conv = bconv
                    .iter()
                    .map(|&BabyBear { value }| value)
                    .collect::<Vec<_>>();

                assert!(conv
                    .iter()
                    .zip(pt_prods)
                    .all(|(&c, p)| p as u64 == (c as u64 * len as u64) % P as u64));
            }
            len *= 2;
            if len > 8192 {
                break;
            }
        }
    }
    */
}
