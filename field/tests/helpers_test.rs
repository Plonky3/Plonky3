mod helpers {
    use p3_baby_bear::BabyBear;
    use p3_field::{
        BasedVectorSpace, Field, HornerIter, PrimeCharacteristicRing, PrimeField, PrimeField32,
        absorb_radix_bits, add_scaled_slice_in_place, chunked_mixed_dot_product,
        dispatch_chunked_mixed_dot_product, dot_product, field_to_array, injective_pack_bits,
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

        assert_eq!(parts, BabyBear::zero_vec(3));
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
        let recomposed: Goldilocks = limbs
            .iter()
            .map(|limb| Goldilocks::from_u64(limb.as_canonical_u32() as u64))
            .horner(base);
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
        let recomposed: BigUint = limbs
            .iter()
            .map(|limb| BigUint::from(limb.as_canonical_u32()))
            .horner(&p);
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
    fn test_algebra_dot_product_matches_same_type() {
        use p3_field::Algebra;
        // Self-algebra case: Algebra::dot_product should match PrimeCharacteristicRing::dot_product
        let a = [BabyBear::TWO, BabyBear::from_u8(4), BabyBear::from_u8(6)];
        let b = [
            BabyBear::from_u8(3),
            BabyBear::from_u8(5),
            BabyBear::from_u8(7),
        ];

        let pcr_result = BabyBear::dot_product(&a, &b);
        let alg_result = <BabyBear as Algebra<BabyBear>>::mixed_dot_product(&a, &b);
        assert_eq!(pcr_result, alg_result);
    }

    #[test]
    fn test_algebra_dot_product_zeros() {
        use p3_field::Algebra;
        let a = [BabyBear::ZERO; 4];
        let f = [BabyBear::ZERO; 4];
        assert_eq!(
            <BabyBear as Algebra<BabyBear>>::mixed_dot_product(&a, &f),
            BabyBear::ZERO
        );
    }

    #[test]
    fn test_algebra_dot_product_unit_vector() {
        use p3_field::Algebra;
        // Unit vector: only one nonzero element in f.
        let a = [
            BabyBear::from_u8(10),
            BabyBear::from_u8(20),
            BabyBear::from_u8(30),
        ];
        let f = [BabyBear::ZERO, BabyBear::ONE, BabyBear::ZERO];
        assert_eq!(
            <BabyBear as Algebra<BabyBear>>::mixed_dot_product(&a, &f),
            BabyBear::from_u8(20)
        );
    }

    #[test]
    fn test_algebra_dot_product_known_values() {
        use p3_field::Algebra;
        // a = [2, 3, 5], f = [7, 11, 13] → 2*7 + 3*11 + 5*13 = 14 + 33 + 65 = 112
        let a = [BabyBear::TWO, BabyBear::from_u8(3), BabyBear::from_u8(5)];
        let f = [
            BabyBear::from_u8(7),
            BabyBear::from_u8(11),
            BabyBear::from_u8(13),
        ];
        assert_eq!(
            <BabyBear as Algebra<BabyBear>>::mixed_dot_product(&a, &f),
            BabyBear::from_u8(112)
        );
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

    #[test]
    fn test_chunked_mixed_dot_product_zero_length() {
        // Empty inputs hit the N <= CHUNK fast path with N = 0.
        // The fast path builds an empty product array and reduces it.
        // Expected: empty sum yields the additive identity.
        let a: [BabyBear; 0] = [];
        let f: [BabyBear; 0] = [];

        assert_eq!(
            chunked_mixed_dot_product::<4, BabyBear, BabyBear, 0>(&a, &f),
            BabyBear::ZERO,
        );
    }

    #[test]
    fn test_chunked_mixed_dot_product_fast_path_n_equals_chunk() {
        // Boundary: N == CHUNK. The condition `N <= CHUNK` is still true,
        // so this exercises the fast path on its upper edge — no outer
        // loop, single balanced reduction over CHUNK products.
        //
        //     a = [2, 3, 5, 7]
        //     f = [11, 13, 17, 19]
        //     expected = 2*11 + 3*13 + 5*17 + 7*19
        //              = 22  + 39  + 85  + 133  = 279
        let a = [
            BabyBear::TWO,
            BabyBear::from_u8(3),
            BabyBear::from_u8(5),
            BabyBear::from_u8(7),
        ];
        let f = [
            BabyBear::from_u8(11),
            BabyBear::from_u8(13),
            BabyBear::from_u8(17),
            BabyBear::from_u8(19),
        ];

        assert_eq!(
            chunked_mixed_dot_product::<4, BabyBear, BabyBear, 4>(&a, &f),
            BabyBear::from_u32(279),
        );
    }

    #[test]
    fn test_chunked_mixed_dot_product_no_remainder() {
        // N = 8, CHUNK = 4: exactly two complete groups, no tail.
        // Exercises the chunked main loop only.
        //
        //     a = [1, 2, 3, 4, 5, 6, 7, 8]
        //     f = [1; 8]
        //     expected = 1+2+3+...+8 = 36
        let a: [BabyBear; 8] = core::array::from_fn(|i| BabyBear::from_u8((i + 1) as u8));
        let f: [BabyBear; 8] = [BabyBear::ONE; 8];

        assert_eq!(
            chunked_mixed_dot_product::<4, BabyBear, BabyBear, 8>(&a, &f),
            BabyBear::from_u8(36),
        );
    }

    #[test]
    fn test_chunked_mixed_dot_product_with_remainder() {
        // N = 10, CHUNK = 4: two complete groups plus a tail of 2.
        // Exercises both the chunked main loop and the scalar tail.
        //
        //     layout:  [group_0 = pairs 0..=3][group_1 = pairs 4..=7][tail = pairs 8..=9]
        //     a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        //     f = [1; 10]
        //     expected = 1+2+...+10 = 55
        let a: [BabyBear; 10] = core::array::from_fn(|i| BabyBear::from_u8((i + 1) as u8));
        let f: [BabyBear; 10] = [BabyBear::ONE; 10];

        assert_eq!(
            chunked_mixed_dot_product::<4, BabyBear, BabyBear, 10>(&a, &f),
            BabyBear::from_u8(55),
        );
    }

    #[test]
    fn test_dispatch_chunked_mixed_dot_product_invariant_over_valid_chunks() {
        // Field addition is associative, so the chunk choice never changes
        // the mathematical result. Run all seven supported chunk values
        // against the same input and assert agreement.
        //
        // N = 20 was picked so different chunks land in different paths:
        //
        //     CHUNK = 1   → 20 chunked groups, no tail        (chunked path)
        //     CHUNK = 2   → 10 chunked groups, no tail        (chunked path)
        //     CHUNK = 4   →  5 chunked groups, no tail        (chunked path)
        //     CHUNK = 8   →  2 chunked groups, tail of 4      (chunked + tail)
        //     CHUNK = 16  →  1 chunked group,  tail of 4      (chunked + tail)
        //     CHUNK = 32  → fast path                          (N <= CHUNK)
        //     CHUNK = 64  → fast path                          (N <= CHUNK)
        let a: [BabyBear; 20] = core::array::from_fn(|i| BabyBear::from_u32(i as u32 * 7 + 11));
        let f: [BabyBear; 20] = core::array::from_fn(|i| BabyBear::from_u32(i as u32 * 13 + 5));

        // Reference: any valid chunk yields the same value.
        let reference = chunked_mixed_dot_product::<1, BabyBear, BabyBear, 20>(&a, &f);

        // Every supported runtime chunk size must reproduce that value.
        for chunk in [1usize, 2, 4, 8, 16, 32, 64] {
            let r = dispatch_chunked_mixed_dot_product::<BabyBear, BabyBear, 20>(&a, &f, chunk);
            assert_eq!(r, reference, "chunk={chunk} disagreed with reference");
        }
    }

    #[test]
    #[should_panic(expected = "mixed_dot_product chunk must be one of 1, 2, 4, 8, 16, 32, or 64")]
    fn test_dispatch_chunked_mixed_dot_product_panics_on_invalid_chunk() {
        // Any chunk outside the supported power-of-two set must panic.
        let a = [BabyBear::ONE; 4];
        let f = [BabyBear::ONE; 4];

        let _ = dispatch_chunked_mixed_dot_product::<BabyBear, BabyBear, 4>(&a, &f, 3);
    }

    #[test]
    fn horner_empty_iterator_is_zero() {
        // Empty iterator → empty sum, both forms return ZERO.
        let coeffs: [BabyBear; 0] = [];
        let x = BabyBear::from_u32(7);
        assert_eq!(
            coeffs.iter().copied().horner::<BabyBear, _>(x),
            BabyBear::ZERO
        );
        assert_eq!(
            coeffs.iter().copied().horner_acc(BabyBear::ZERO, x),
            BabyBear::ZERO,
        );
    }

    #[test]
    fn horner_single_coefficient_is_constant() {
        // [c] is a degree-0 polynomial; the result is c regardless of x.
        let c = BabyBear::from_u32(42);
        let coeffs = [c];
        assert_eq!(
            coeffs
                .iter()
                .copied()
                .horner::<BabyBear, _>(BabyBear::from_u32(7)),
            c
        );
        assert_eq!(
            coeffs.iter().copied().horner::<BabyBear, _>(BabyBear::ZERO),
            c
        );
    }

    #[test]
    fn horner_matches_running_power_loop() {
        // Coefficients are ascending: result = Σ c_i · x^i.
        let coeffs: [BabyBear; 5] = [
            BabyBear::from_u32(3),
            BabyBear::from_u32(5),
            BabyBear::from_u32(7),
            BabyBear::from_u32(11),
            BabyBear::from_u32(13),
        ];
        let x = BabyBear::from_u32(4);

        let horner: BabyBear = coeffs.iter().copied().horner(x);

        let mut expected = BabyBear::ZERO;
        let mut power = BabyBear::ONE;
        for &c in &coeffs {
            expected += c * power;
            power *= x;
        }
        assert_eq!(horner, expected);
    }

    #[test]
    fn horner_acc_composes_over_concatenated_coefficients() {
        // For ascending coefficients [low ‖ high] (low covers degrees 0..l,
        // high covers degrees l..l+h):
        //
        //     P(x) = (Σ_i low_i x^i) + x^l · (Σ_j high_j x^j).
        //
        // `horner_acc(seed, x)` on `low` returns `seed · x^l + Σ_i low_i x^i`,
        // so passing the standalone `high` evaluation as the seed reconstructs P.
        let low: [BabyBear; 3] = [
            BabyBear::from_u32(2),
            BabyBear::from_u32(3),
            BabyBear::from_u32(5),
        ];
        let high: [BabyBear; 2] = [BabyBear::from_u32(7), BabyBear::from_u32(11)];
        let x = BabyBear::from_u32(6);

        let mut concat = low.to_vec();
        concat.extend_from_slice(&high);
        let one_shot: BabyBear = concat.iter().copied().horner(x);

        let high_eval: BabyBear = high.iter().copied().horner(x);
        let combined: BabyBear = low.iter().copied().horner_acc(high_eval, x);
        assert_eq!(combined, one_shot);
    }

    #[test]
    fn horner_with_extension_accumulator_over_base_coefficients() {
        // Cross-type Horner: base-field coefficients, extension-field point.
        // The other Horner tests in this file use the same field on both sides.
        type EF = p3_field::extension::BinomialExtensionField<BabyBear, 4>;

        let coeffs: [BabyBear; 4] = [
            BabyBear::from_u32(1),
            BabyBear::from_u32(2),
            BabyBear::from_u32(3),
            BabyBear::from_u32(4),
        ];

        // All coordinates non-zero so the evaluation exercises every extension
        // component, not just the base-field embedding.
        let x = EF::from_basis_coefficients_slice(&[
            BabyBear::from_u32(5),
            BabyBear::from_u32(6),
            BabyBear::from_u32(7),
            BabyBear::from_u32(8),
        ])
        .unwrap();

        let horner: EF = coeffs.iter().copied().horner(x);

        // Reference: explicit sum_{i} c_i * x^i.
        let mut expected = EF::ZERO;
        let mut power = EF::ONE;
        for &c in &coeffs {
            expected += power * c;
            power *= x;
        }
        assert_eq!(horner, expected);
    }
}
