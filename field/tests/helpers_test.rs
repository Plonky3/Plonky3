mod helpers {
    use num_bigint::BigUint;
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{
        BasedVectorSpace, PrimeCharacteristicRing, PrimeField32, PrimeField64,
        add_scaled_slice_in_place, dot_product, field_to_array, halve_u32, halve_u64, reduce_32,
        scale_vec, split_32, two_adic_coset_vanishing_polynomial,
        two_adic_subgroup_vanishing_polynomial,
    };

    #[test]
    fn test_two_adic_subgroup_vanishing_polynomial() {
        // x = 3, log_n = 3 → compute x^8 - 1
        let x = BabyBear::from_u64(3);
        let log_n = 3;

        // x^8 = 3^8
        let x_pow = x * x * x * x * x * x * x * x;

        let expected = x_pow - BabyBear::ONE;
        let result = two_adic_subgroup_vanishing_polynomial(log_n, x);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_two_adic_coset_vanishing_polynomial() {
        // x = 2, shift = 5, log_n = 2 → compute x^4 - shift^4
        let x = BabyBear::from_u64(2);
        let shift = BabyBear::from_u64(5);
        let log_n = 2;

        // x^4 = 2^4
        let x_pow = x * x * x * x;

        // shift^4 = 5^4
        let shift_pow = shift * shift * shift * shift;

        let expected = x_pow - shift_pow;
        let result = two_adic_coset_vanishing_polynomial(log_n, shift, x);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_scale_vec() {
        // Scale [1, 2, 3] by 7
        let v = vec![BabyBear::new(1), BabyBear::new(2), BabyBear::new(3)];
        let s = BabyBear::new(7);
        let result = scale_vec(s, v.clone());

        let expected = vec![
            s * BabyBear::new(1),
            s * BabyBear::new(2),
            s * BabyBear::new(3),
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_add_scaled_slice_in_place() {
        // x = [1, 2], y = [10, 20], scale by 3
        let x1 = BabyBear::new(1);
        let x2 = BabyBear::new(2);
        let mut x = vec![x1, x2];

        let y1 = BabyBear::new(10);
        let y2 = BabyBear::new(20);
        let y = vec![y1, y2];
        let s = BabyBear::new(3);

        add_scaled_slice_in_place(&mut x, y.clone().into_iter(), s);

        // x = [x1 + s * y1, x2 + s * y2]
        let expected = vec![x1 + s * y1, x2 + s * y2];

        assert_eq!(x, expected);
    }

    #[test]
    fn test_add_scaled_slice_in_place_zero_scale() {
        let original = vec![BabyBear::new(4), BabyBear::new(5)];
        let mut x = original.clone();
        let y = vec![BabyBear::new(6), BabyBear::new(7)];
        let s = BabyBear::ZERO;

        add_scaled_slice_in_place(&mut x, y.into_iter(), s);

        assert_eq!(x, original);
    }

    #[test]
    fn test_field_to_array() {
        // Convert value 9 to array of size 4
        let x = BabyBear::new(9);
        let arr = field_to_array::<BabyBear, 4>(x);

        // Should yield [9, 0, 0, 0]
        assert_eq!(arr, [x, BabyBear::ZERO, BabyBear::ZERO, BabyBear::ZERO]);
    }

    #[test]
    fn test_field_to_array_single() {
        let x = BabyBear::new(99);
        let arr = field_to_array::<BabyBear, 1>(x);
        assert_eq!(arr, [x]);
    }

    #[test]
    fn test_halve_u32() {
        // Let x = 5, P = BabyBear::ORDER_U32 = 2^32 - 2^20 + 1
        let x: u32 = 5;
        let result = halve_u32::<{ BabyBear::ORDER_U32 }>(x);

        // shift = (P + 1) >> 1
        let shift = (BabyBear::ORDER_U32 + 1) >> 1;
        let expected = (x >> 1) + shift; // since x is odd

        assert_eq!(result, expected);
    }

    #[test]
    fn test_halve_u64() {
        let x: u64 = 6;
        let result = halve_u64::<{ BabyBear::ORDER_U64 }>(x);

        let expected = x >> 1; // since x is even

        assert_eq!(result, expected);
    }

    #[test]
    fn test_reduce_32() {
        // Input: vals = [1, 2, 3]
        let vals = [
            BabyBear::from_u32(1),
            BabyBear::from_u32(2),
            BabyBear::from_u32(3),
        ];

        // po2 = 2^32 mod BabyBear = 1048575
        let po2 = BabyBear::from_u64(1u64 << 32); // 2^32 mod field

        // Manual reduction process (reverse order):
        // Step 1: result = 0
        // Step 2: result = result * po2 + 3 = 0 + 3 = 3
        // Step 3: result = result * po2 + 2 = 3 * 1048575 + 2
        // Step 4: result = result * po2 + 1 = (step 3) * 1048575 + 1

        let step1 = BabyBear::ZERO;
        let step2 = step1 * po2 + vals[2]; // = 0 + 3 = 3
        let step3 = step2 * po2 + vals[1]; // = 3 * 1048575 + 2
        let expected = step3 * po2 + vals[0]; // = ... + 1

        let result = reduce_32::<BabyBear, BabyBear>(&vals);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_reduce_32_empty() {
        let vals: [BabyBear; 0] = [];
        let result = reduce_32::<BabyBear, BabyBear>(&vals);
        assert_eq!(result, BabyBear::ZERO);
    }

    #[test]
    fn test_split_32_round_trip() {
        // Choose any field element as base input (already reduced)
        let val = BabyBear::from_u32(1172168165);

        // Split it into base-2^64 "digits"
        let parts = split_32::<BabyBear, BabyBear>(val, 2);

        // Recombine it using reduce_32
        let recomposed = reduce_32::<BabyBear, BabyBear>(&parts);

        // It should match the original value
        assert_eq!(recomposed, val);
    }

    #[test]
    fn test_split_32_zero() {
        let val = BabyBear::ZERO;
        let parts = split_32::<BabyBear, BabyBear>(val, 3);

        assert_eq!(parts, vec![BabyBear::ZERO; 3]);
    }

    #[test]
    fn test_dot_product() {
        let a1 = BabyBear::new(2);
        let a2 = BabyBear::new(4);
        let a3 = BabyBear::new(6);
        let a = vec![a1, a2, a3];

        let b1 = BabyBear::new(3);
        let b2 = BabyBear::new(5);
        let b3 = BabyBear::new(7);
        let b = vec![b1, b2, b3];

        // 2*3 + 4*5 + 6*7
        let expected = a1 * b1 + a2 * b2 + a3 * b3;

        let result = dot_product::<BabyBear, _, _>(a.iter().copied(), b.iter().copied());
        assert_eq!(result, expected);
    }

    #[test]
    fn test_dot_product_empty() {
        let a: Vec<BabyBear> = vec![];
        let b: Vec<BabyBear> = vec![];
        let result = dot_product::<BabyBear, _, _>(a.into_iter(), b.into_iter());
        assert_eq!(result, BabyBear::ZERO);
    }

    #[test]
    fn test_dot_product_mismatched_lengths() {
        let a1 = BabyBear::new(2);
        let a2 = BabyBear::new(4);
        let a = vec![a1, a2];

        let b1 = BabyBear::new(3);
        let b2 = BabyBear::new(5);
        let b3 = BabyBear::new(7);
        let b = vec![b1, b2, b3];

        // Only first two elements will be multiplied
        let expected = a1 * b1 + a2 * b2;

        let result = dot_product::<BabyBear, _, _>(a.into_iter(), b.into_iter());
        assert_eq!(result, expected);
    }

    #[test]
    fn test_field_to_array_complex() {
        use p3_baby_bear::BabyBear;
        use p3_field::field_to_array;

        // Case 1: Non-zero element, D = 5
        let x = BabyBear::from_u32(123);
        let arr = field_to_array::<BabyBear, 5>(x);

        // Should produce: [123, 0, 0, 0, 0]
        assert_eq!(
            arr,
            [
                BabyBear::from_u32(123),
                BabyBear::ZERO,
                BabyBear::ZERO,
                BabyBear::ZERO,
                BabyBear::ZERO
            ]
        );

        // Case 2: Zero input value
        let x = BabyBear::ZERO;
        let arr = field_to_array::<BabyBear, 3>(x);

        // Should be all zeros: [0, 0, 0]
        assert_eq!(arr, [BabyBear::ZERO; 3]);
    }
}
