/// Common `packed_mod_add` / `packed_mod_sub` proptests shared across architecture backends.
#[allow(unused_macros)]
macro_rules! packed_mod_tests {
    () => {
        use proptest::prelude::*;
        use proptest::test_runner::TestRunner;

        // KoalaBear prime: 2^31 - 2^24 + 1.
        const P: u32 = 0x7f000001;

        /// Reference scalar modular addition: (a + b) mod P, returns value in [0, P].
        fn ref_add(a: u32, b: u32) -> u32 {
            let sum = a + b;
            if sum >= P { sum - P } else { sum }
        }

        /// Reference scalar modular subtraction: (a - b) mod P, returns value in [0, P].
        fn ref_sub(a: u32, b: u32) -> u32 {
            if a >= b { a - b } else { a + P - b }
        }

        fn val_in_range() -> impl Strategy<Value = u32> {
            0..P
        }

        fn array_strategy<const N: usize>() -> impl Strategy<Value = [u32; N]> {
            proptest::collection::vec(val_in_range(), N).prop_map(|v| v.try_into().unwrap())
        }

        // ------- packed_mod_add / packed_mod_sub helpers -------

        fn check_packed_mod_add<const WIDTH: usize>(a: [u32; WIDTH], b: [u32; WIDTH]) {
            let mut res = [0u32; WIDTH];
            packed_mod_add(&a, &b, &mut res, P, ref_add);
            for i in 0..WIDTH {
                assert_eq!(
                    res[i],
                    ref_add(a[i], b[i]),
                    "add mismatch at index {i} for width {WIDTH}"
                );
            }
        }

        fn check_packed_mod_sub<const WIDTH: usize>(a: [u32; WIDTH], b: [u32; WIDTH]) {
            let mut res = [0u32; WIDTH];
            packed_mod_sub(&a, &b, &mut res, P, ref_sub);
            for i in 0..WIDTH {
                assert_eq!(
                    res[i],
                    ref_sub(a[i], b[i]),
                    "sub mismatch at index {i} for width {WIDTH}"
                );
            }
        }

        fn run_packed_add_test<const WIDTH: usize>() {
            let mut runner = TestRunner::default();
            runner
                .run(
                    &(array_strategy::<WIDTH>(), array_strategy::<WIDTH>()),
                    |(a, b)| {
                        check_packed_mod_add(a, b);
                        Ok(())
                    },
                )
                .unwrap();
        }

        fn run_packed_sub_test<const WIDTH: usize>() {
            let mut runner = TestRunner::default();
            runner
                .run(
                    &(array_strategy::<WIDTH>(), array_strategy::<WIDTH>()),
                    |(a, b)| {
                        check_packed_mod_sub(a, b);
                        Ok(())
                    },
                )
                .unwrap();
        }

        // ------- packed_mod_add proptests for all widths -------

        #[test]
        fn test_packed_mod_add_w1() {
            run_packed_add_test::<1>();
        }
        #[test]
        fn test_packed_mod_add_w2() {
            run_packed_add_test::<2>();
        }
        #[test]
        fn test_packed_mod_add_w3() {
            run_packed_add_test::<3>();
        }
        #[test]
        fn test_packed_mod_add_w4() {
            run_packed_add_test::<4>();
        }
        #[test]
        fn test_packed_mod_add_w5() {
            run_packed_add_test::<5>();
        }
        #[test]
        fn test_packed_mod_add_w6() {
            run_packed_add_test::<6>();
        }
        #[test]
        fn test_packed_mod_add_w7() {
            run_packed_add_test::<7>();
        }
        #[test]
        fn test_packed_mod_add_w8() {
            run_packed_add_test::<8>();
        }

        // ------- packed_mod_sub proptests for all widths -------

        #[test]
        fn test_packed_mod_sub_w1() {
            run_packed_sub_test::<1>();
        }
        #[test]
        fn test_packed_mod_sub_w2() {
            run_packed_sub_test::<2>();
        }
        #[test]
        fn test_packed_mod_sub_w3() {
            run_packed_sub_test::<3>();
        }
        #[test]
        fn test_packed_mod_sub_w4() {
            run_packed_sub_test::<4>();
        }
        #[test]
        fn test_packed_mod_sub_w5() {
            run_packed_sub_test::<5>();
        }
        #[test]
        fn test_packed_mod_sub_w6() {
            run_packed_sub_test::<6>();
        }
        #[test]
        fn test_packed_mod_sub_w7() {
            run_packed_sub_test::<7>();
        }
        #[test]
        fn test_packed_mod_sub_w8() {
            run_packed_sub_test::<8>();
        }

        // ------- Boundary value tests -------

        #[test]
        fn test_add_boundary_values() {
            let mut res = [0u32; 4];
            packed_mod_add(&[0, 0, 0, 0], &[0, 0, 0, 0], &mut res, P, ref_add);
            assert_eq!(res, [0, 0, 0, 0]);

            packed_mod_add(
                &[P - 1, P - 1, 1, 0],
                &[1, 0, P - 1, 0],
                &mut res,
                P,
                ref_add,
            );
            assert_eq!(res, [0, P - 1, 0, 0]);

            packed_mod_add(&[P - 1; 4], &[P - 1; 4], &mut res, P, ref_add);
            assert_eq!(res, [P - 2; 4]);
        }

        #[test]
        fn test_sub_boundary_values() {
            let mut res = [0u32; 4];
            packed_mod_sub(&[0, 0, 0, 0], &[0, 0, 0, 0], &mut res, P, ref_sub);
            assert_eq!(res, [0, 0, 0, 0]);

            packed_mod_sub(&[0; 4], &[1; 4], &mut res, P, ref_sub);
            assert_eq!(res, [P - 1; 4]);

            packed_mod_sub(&[P - 1; 4], &[P - 1; 4], &mut res, P, ref_sub);
            assert_eq!(res, [0; 4]);
        }

        // ------- Add/sub inverse property -------

        fn check_add_sub_inverse<const WIDTH: usize>(a: [u32; WIDTH], b: [u32; WIDTH]) {
            let mut sum = [0u32; WIDTH];
            let mut roundtrip = [0u32; WIDTH];
            packed_mod_add(&a, &b, &mut sum, P, ref_add);
            packed_mod_sub(&sum, &b, &mut roundtrip, P, ref_sub);
            assert_eq!(roundtrip, a, "add/sub inverse failed for width {WIDTH}");
        }

        #[test]
        fn test_add_sub_inverse_w4() {
            let mut runner = TestRunner::default();
            runner
                .run(&(array_strategy::<4>(), array_strategy::<4>()), |(a, b)| {
                    check_add_sub_inverse(a, b);
                    Ok(())
                })
                .unwrap();
        }

        #[test]
        fn test_add_sub_inverse_w8() {
            let mut runner = TestRunner::default();
            runner
                .run(&(array_strategy::<8>(), array_strategy::<8>()), |(a, b)| {
                    check_add_sub_inverse(a, b);
                    Ok(())
                })
                .unwrap();
        }
    };
}
