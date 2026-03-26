mod helpers {
    use p3_baby_bear::BabyBear;
    use p3_field::{
        Field, PrimeCharacteristicRing, PrimeField, PrimeField32, absorb_radix_bits,
        add_scaled_slice_in_place, dot_product, field_to_array, injective_pack_bits,
        max_absorb_injective_limbs, max_packed_injective_limbs, max_shifted_absorb_injective_limbs,
        max_shifted_packed_injective_limbs, par_add_scaled_slice_in_place,
        pf_packed_limbs_cover_order, reduce_32, reduce_packed, reduce_packed_shifted, split_32,
        split_pf_to_field_order_limbs, split_pf_to_packed_limbs, squeeze_field_order_num_limbs,
    };
    use p3_goldilocks::Goldilocks;

    #[test]
    fn test_add_scaled_slice_in_place() {
        // x = [1, 2], y = [10, 20], scale by 3
        let x1 = BabyBear::ONE;
        let x2 = BabyBear::TWO;
        let mut x = vec![x1, x2];
        let mut par_x = x.clone();

        let y1 = BabyBear::from_u8(10);
        let y2 = BabyBear::from_u8(20);
        let y = vec![y1, y2];
        let s = BabyBear::from_u8(3);

        add_scaled_slice_in_place(&mut x, &y, s);
        par_add_scaled_slice_in_place(&mut par_x, &y, s);

        // x = [x1 + s * y1, x2 + s * y2]
        let expected = vec![x1 + s * y1, x2 + s * y2];

        assert_eq!(x, expected);
        assert_eq!(par_x, expected);
    }

    #[test]
    fn test_add_scaled_slice_in_place_zero_scale() {
        let original = vec![BabyBear::from_u8(4), BabyBear::from_u8(5)];
        let mut x = original.clone();
        let mut par_x = original.clone();
        let y = vec![BabyBear::from_u8(6), BabyBear::from_u8(7)];
        let s = BabyBear::ZERO;

        add_scaled_slice_in_place(&mut x, &y, s);
        par_add_scaled_slice_in_place(&mut par_x, &y, s);

        assert_eq!(x, original);
        assert_eq!(par_x, original);
    }

    #[test]
    fn test_field_to_array() {
        // Convert value 9 to array of size 4
        let x = BabyBear::from_u8(9);
        let arr = field_to_array::<BabyBear, 4>(x);

        // Should yield [9, 0, 0, 0]
        assert_eq!(arr, [x, BabyBear::ZERO, BabyBear::ZERO, BabyBear::ZERO]);
    }

    #[test]
    fn test_field_to_array_single() {
        let x = BabyBear::from_u8(99);
        let arr = field_to_array::<BabyBear, 1>(x);
        assert_eq!(arr, [x]);
    }

    #[test]
    fn test_reduce_32() {
        // Input: vals = [1, 2, 3]
        let vals = [BabyBear::ONE, BabyBear::TWO, BabyBear::from_u32(3)];

        // po2 = 2^32 mod BabyBear = 1048575
        let po2 = BabyBear::from_u64(1u64 << 32); // 2^32 mod field

        // Manual reduction process (reverse order):
        // Step 1: result = 0
        // Step 2: result = result * po2 + 3
        // Step 3: result = result * po2 + 2
        // Step 4: result = result * po2 + 1

        let step1 = BabyBear::ZERO;
        let step2 = step1 * po2 + vals[2];
        let step3 = step2 * po2 + vals[1];
        let expected = step3 * po2 + vals[0];

        let result = reduce_32::<BabyBear, BabyBear>(&vals);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_reduce_32_large_vector_high_entropy() {
        // Input: vals = [1, 2, 3, ..., 10]
        let vals: Vec<BabyBear> = (1..=10).map(BabyBear::from_u32).collect();

        let po2 = BabyBear::from_u64(1u64 << 32); // base = 2^32

        // Manual computation step-by-step:
        let step10 = BabyBear::from_u32(10);
        let step9 = step10 * po2 + BabyBear::from_u32(9);
        let step8 = step9 * po2 + BabyBear::from_u32(8);
        let step7 = step8 * po2 + BabyBear::from_u32(7);
        let step6 = step7 * po2 + BabyBear::from_u32(6);
        let step5 = step6 * po2 + BabyBear::from_u32(5);
        let step4 = step5 * po2 + BabyBear::from_u32(4);
        let step3 = step4 * po2 + BabyBear::from_u32(3);
        let step2 = step3 * po2 + BabyBear::TWO;
        let expected = step2 * po2 + BabyBear::ONE;

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
    fn test_packed_limbs_roundtrip_goldilocks() {
        let g = Goldilocks::from_u64(12_345_678_901_234_567_890u64);
        let pb = injective_pack_bits::<BabyBear>();
        let n = Goldilocks::bits().div_ceil(pb as usize);
        let limbs = split_pf_to_packed_limbs::<Goldilocks, BabyBear>(g, n, pb);
        // Recompose using reduce_32's radix (2^32) on the 30-bit limbs:
        // each limb is in [0, 2^30), so the packed integer is the same as with radix 2^30
        // only if no digit exceeds the smaller base. Here they don't, so roundtrip works
        // via the original reduce_32 (which uses base 2^32) only when values are small.
        // Instead, manually reconstruct with the matching radix.
        let base = Goldilocks::from_u64(1u64 << pb);
        let recomposed: Goldilocks = limbs.iter().rev().fold(Goldilocks::ZERO, |acc, &limb| {
            acc * base + Goldilocks::from_u64(limb.as_canonical_u32() as u64)
        });
        assert_eq!(recomposed, g);
    }

    #[test]
    fn test_absorb_radix_bits_baby_bear() {
        assert_eq!(absorb_radix_bits::<BabyBear>(), 31);
    }

    #[test]
    fn test_max_absorb_injective_limbs_baby_bear_goldilocks() {
        // Tighter radix 2^31 still yields k=2 into Goldilocks.
        assert_eq!(max_absorb_injective_limbs::<BabyBear, Goldilocks>(), 2);
        assert_eq!(
            max_absorb_injective_limbs::<BabyBear, Goldilocks>(),
            max_packed_injective_limbs::<BabyBear, Goldilocks>(absorb_radix_bits::<BabyBear>()),
        );
    }

    #[test]
    fn test_max_shifted_absorb_injective_limbs_baby_bear_goldilocks() {
        assert_eq!(
            max_shifted_absorb_injective_limbs::<BabyBear, Goldilocks>(),
            2
        );
        assert_eq!(
            max_shifted_absorb_injective_limbs::<BabyBear, Goldilocks>(),
            max_shifted_packed_injective_limbs::<BabyBear, Goldilocks>(
                absorb_radix_bits::<BabyBear>()
            ),
        );
    }

    #[test]
    fn test_reduce_packed_matches_reduce_32_when_radix_32() {
        let vals: Vec<BabyBear> = (1..=5).map(BabyBear::from_u32).collect();
        assert_eq!(
            reduce_packed::<BabyBear, BabyBear>(&vals, 32),
            reduce_32::<BabyBear, BabyBear>(&vals),
        );
    }

    #[test]
    fn test_reduce_packed_shifted_distinguishes_trailing_zero() {
        let rb = absorb_radix_bits::<BabyBear>();
        assert_ne!(
            reduce_packed_shifted::<BabyBear, Goldilocks>(&[BabyBear::ONE], rb),
            reduce_packed_shifted::<BabyBear, Goldilocks>(&[BabyBear::ONE, BabyBear::ZERO], rb),
        );
    }

    #[test]
    fn test_squeeze_field_order_num_limbs_baby_bear_goldilocks() {
        // F::ORDER^2 ≈ 2^{61.97} < Goldilocks::ORDER ≈ 2^{64}
        // F::ORDER^3 ≈ 2^{92.8} >> Goldilocks::ORDER
        // Largest k with F::ORDER^{k+1} < Goldilocks::ORDER → k=1.
        assert_eq!(squeeze_field_order_num_limbs::<Goldilocks, BabyBear>(), 1);
    }

    #[test]
    fn test_split_pf_to_field_order_limbs_roundtrip_goldilocks() {
        use num_bigint::BigUint;
        let g = Goldilocks::from_u64(12_345_678_901_234_567_890u64);
        let num_limbs = squeeze_field_order_num_limbs::<Goldilocks, BabyBear>();
        let limbs = split_pf_to_field_order_limbs::<Goldilocks, BabyBear>(g, num_limbs);
        assert_eq!(limbs.len(), num_limbs);
        // Each limb must be a valid BabyBear element (< BabyBear::ORDER).
        for limb in &limbs {
            assert!(limb.as_canonical_u32() < BabyBear::ORDER_U32);
        }
        // Recompose in base p_F and verify.
        let p = BigUint::from(BabyBear::ORDER_U32);
        let recomposed: BigUint = limbs.iter().rev().fold(BigUint::from(0u32), |acc, limb| {
            acc * &p + BigUint::from(limb.as_canonical_u32())
        });
        assert_eq!(
            recomposed,
            g.as_canonical_biguint() % p.pow(num_limbs as u32)
        );
    }

    #[test]
    fn test_split_pf_to_field_order_limbs_covers_full_f_range() {
        // With base 2^30, limbs are confined to [0, 2^30) ≈ 50% of BabyBear.
        // With base p_F, limbs can take any value in [0, p_BabyBear).
        // Construct a Goldilocks value whose c0 = v mod p_BB falls above 2^30.
        let threshold = 1u32 << injective_pack_bits::<BabyBear>();
        // Choose a Goldilocks element large enough that v mod p_BB > threshold.
        // p_BB = 2130706433. Any v with (v mod p_BB) in (threshold, p_BB) qualifies.
        let target = threshold + 1; // a value in BabyBear above the old ceiling
        let g = Goldilocks::from_u64(target as u64);
        let limbs = split_pf_to_field_order_limbs::<Goldilocks, BabyBear>(g, 1);
        assert_eq!(limbs[0].as_canonical_u32(), target);
        assert!(limbs[0].as_canonical_u32() >= threshold);
    }

    #[test]
    fn test_pf_packed_limbs_cover_order_goldilocks_baby_bear() {
        let pb = injective_pack_bits::<BabyBear>();
        let n_observe = Goldilocks::bits().div_ceil(pb as usize);
        let n_squeeze = Goldilocks::bits() / (pb as usize);
        assert!(pf_packed_limbs_cover_order::<Goldilocks>(n_observe, pb));
        assert!(!pf_packed_limbs_cover_order::<Goldilocks>(n_squeeze, pb));
    }

    #[test]
    fn test_dot_product() {
        let a1 = BabyBear::TWO;
        let a2 = BabyBear::from_u8(4);
        let a3 = BabyBear::from_u8(6);
        let a = [a1, a2, a3];

        let b1 = BabyBear::from_u8(3);
        let b2 = BabyBear::from_u8(5);
        let b3 = BabyBear::from_u8(7);
        let b = [b1, b2, b3];

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
        let a1 = BabyBear::TWO;
        let a2 = BabyBear::from_u8(4);
        let a = vec![a1, a2];

        let b1 = BabyBear::from_u8(3);
        let b2 = BabyBear::from_u8(5);
        let b3 = BabyBear::from_u8(7);
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
