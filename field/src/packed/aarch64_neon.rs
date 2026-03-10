//! A collection of helper methods when Neon is available

use core::arch::aarch64::{self, uint32x4_t};
use core::mem::transmute;

/// Convert a four element array of u32's into a packed vector.
///
/// This will be a no-op.
#[inline(always)]
fn array_to_uint32x4(input: [u32; 4]) -> uint32x4_t {
    // Safety: `[u32; 4]` has the same size as `uint32x4_t`.
    unsafe { transmute::<[u32; 4], uint32x4_t>(input) }
}

/// Convert a packed vector into a four element array of u32's.
///
/// This will be a no-op.
#[inline(always)]
fn uint32x4_to_array(input: uint32x4_t) -> [u32; 4] {
    // Safety: `[u32; 4]` has the same size and alignment as `uint32x4_t`.
    unsafe { transmute::<uint32x4_t, [u32; 4]>(input) }
}

/// Add the packed vectors `a` and `b` modulo `p`.
///
/// This allows us to add 4 elements at once.
///
/// Assumes that `p` is less than `2^31` and `a + b <= 2P`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a + b) mod p`.
/// It will be equal to `P` if and only if `a + b = 2P` so provided `a + b < 2P`
/// the result is guaranteed to be less than `P`.
#[inline]
#[must_use]
pub fn uint32x4_mod_add(a: uint32x4_t, b: uint32x4_t, p: uint32x4_t) -> uint32x4_t {
    // We want this to compile to:
    //      add   t.4s, a.4s, b.4s
    //      sub   u.4s, t.4s, P.4s
    //      umin  res.4s, t.4s, u.4s
    // throughput: .75 cyc/vec (5.33 els/cyc)
    // latency: 6 cyc

    // See field/src/packed/x86_64_avx.rs for a proof of correctness of this algorithm.

    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let t = aarch64::vaddq_u32(a, b);
        let u = aarch64::vsubq_u32(t, p);
        aarch64::vminq_u32(t, u)
    }
}

/// Subtract the packed vectors `a` and `b` modulo `p`.
///
/// This allows us to subtract 4 elements at once.
///
/// Assumes that `p` is less than `2^31` and `|a - b| <= P`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a - b) mod p`.
/// It will be equal to `P` if and only if `a - b = P` so provided `a - b < P`
/// the result is guaranteed to be less than `P`.
#[inline]
#[must_use]
pub fn uint32x4_mod_sub(a: uint32x4_t, b: uint32x4_t, p: uint32x4_t) -> uint32x4_t {
    // We want this to compile to:
    //      sub   t.4s, a.4s, b.4s
    //      add   u.4s, t.4s, P.4s
    //      umin  res.4s, t.4s, u.4s
    // throughput: .75 cyc/vec (5.33 els/cyc)
    // latency: 6 cyc

    // See field/src/packed/x86_64_avx.rs for a proof of correctness of this algorithm.

    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let t = aarch64::vsubq_u32(a, b);
        let u = aarch64::vaddq_u32(t, p);
        aarch64::vminq_u32(t, u)
    }
}

/// Add two arrays of integers modulo `P` using packings.
///
/// Assumes that `P` is less than `2^31` and `a + b <= 2P` for all array pairs `a, b`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a + b) mod P`.
/// It will be equal to `P` if and only if `a + b = 2P` so provided `a + b < 2P`
/// the result is guaranteed to be less than `P`.
///
/// Scalar add is assumed to be a function which implements `a + b % P` with the
/// same specifications as above.
#[inline(always)]
pub fn packed_mod_add<const WIDTH: usize>(
    a: &[u32; WIDTH],
    b: &[u32; WIDTH],
    res: &mut [u32; WIDTH],
    p: u32,
    scalar_add: fn(u32, u32) -> u32,
) {
    match WIDTH {
        1 => res[0] = scalar_add(a[0], b[0]),
        2 => {
            res[0] = scalar_add(a[0], b[0]);
            res[1] = scalar_add(a[1], b[1]);
        }
        3 => {
            res[0] = scalar_add(a[0], b[0]);
            res[1] = scalar_add(a[1], b[1]);
            res[2] = scalar_add(a[2], b[2]);
        }
        4 => {
            // Perfectly fits into a uint32x4_t vector.
            let out: [u32; 4] = unsafe {
                let a = array_to_uint32x4([a[0], a[1], a[2], a[3]]);
                let b = array_to_uint32x4([b[0], b[1], b[2], b[3]]);
                let p: uint32x4_t = aarch64::vdupq_n_u32(p);
                uint32x4_to_array(uint32x4_mod_add(a, b, p))
            };

            res.copy_from_slice(&out);
        }
        5 => {
            // We fit what we can into a uint32x4_t element.
            // The final add is done using a scalar addition.
            let out: [u32; 4] = unsafe {
                let a = array_to_uint32x4([a[0], a[1], a[2], a[3]]);
                let b = array_to_uint32x4([b[0], b[1], b[2], b[3]]);
                let p: uint32x4_t = aarch64::vdupq_n_u32(p);
                uint32x4_to_array(uint32x4_mod_add(a, b, p))
            };

            res[4] = scalar_add(a[4], b[4]);

            res[..4].copy_from_slice(&out);
        }
        6 => {
            // First 4 elements fit into a uint32x4_t, remaining 2 use scalar.
            let out: [u32; 4] = unsafe {
                let a = array_to_uint32x4([a[0], a[1], a[2], a[3]]);
                let b = array_to_uint32x4([b[0], b[1], b[2], b[3]]);
                let p: uint32x4_t = aarch64::vdupq_n_u32(p);
                uint32x4_to_array(uint32x4_mod_add(a, b, p))
            };

            res[4] = scalar_add(a[4], b[4]);
            res[5] = scalar_add(a[5], b[5]);

            res[..4].copy_from_slice(&out);
        }
        7 => {
            // First 4 elements fit into a uint32x4_t, remaining 3 use scalar.
            let out: [u32; 4] = unsafe {
                let a = array_to_uint32x4([a[0], a[1], a[2], a[3]]);
                let b = array_to_uint32x4([b[0], b[1], b[2], b[3]]);
                let p: uint32x4_t = aarch64::vdupq_n_u32(p);
                uint32x4_to_array(uint32x4_mod_add(a, b, p))
            };

            res[4] = scalar_add(a[4], b[4]);
            res[5] = scalar_add(a[5], b[5]);
            res[6] = scalar_add(a[6], b[6]);

            res[..4].copy_from_slice(&out);
        }
        8 => {
            // This perfectly fits into two uint32x4_t elements.
            let (out_lo, out_hi): ([u32; 4], [u32; 4]) = unsafe {
                let p: uint32x4_t = aarch64::vdupq_n_u32(p);

                let a_lo = array_to_uint32x4([a[0], a[1], a[2], a[3]]);
                let b_lo = array_to_uint32x4([b[0], b[1], b[2], b[3]]);
                let out_lo = uint32x4_to_array(uint32x4_mod_add(a_lo, b_lo, p));

                let a_hi = array_to_uint32x4([a[4], a[5], a[6], a[7]]);
                let b_hi = array_to_uint32x4([b[4], b[5], b[6], b[7]]);
                let out_hi = uint32x4_to_array(uint32x4_mod_add(a_hi, b_hi, p));
                (out_lo, out_hi)
            };

            res[..4].copy_from_slice(&out_lo);
            res[4..].copy_from_slice(&out_hi);
        }
        _ => panic!("Currently unsupported width for packed addition"),
    }
}

/// Subtract two arrays of integers modulo `P` using packings.
///
/// Assumes that `p` is less than `2^31` and `|a - b| <= P`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a - b) mod p`.
/// It will be equal to `P` if and only if `a - b = P` so provided `a - b < P`
/// the result is guaranteed to be less than `P`.
///
/// Scalar sub is assumed to be a function which implements `a - b % P` with the
/// same specifications as above.
///
#[inline(always)]
pub fn packed_mod_sub<const WIDTH: usize>(
    a: &[u32; WIDTH],
    b: &[u32; WIDTH],
    res: &mut [u32; WIDTH],
    p: u32,
    scalar_sub: fn(u32, u32) -> u32,
) {
    match WIDTH {
        1 => res[0] = scalar_sub(a[0], b[0]),
        2 => {
            res[0] = scalar_sub(a[0], b[0]);
            res[1] = scalar_sub(a[1], b[1]);
        }
        3 => {
            res[0] = scalar_sub(a[0], b[0]);
            res[1] = scalar_sub(a[1], b[1]);
            res[2] = scalar_sub(a[2], b[2]);
        }
        4 => {
            // Perfectly fits into a uint32x4_t vector.
            let out: [u32; 4] = unsafe {
                let a = array_to_uint32x4([a[0], a[1], a[2], a[3]]);
                let b = array_to_uint32x4([b[0], b[1], b[2], b[3]]);
                let p: uint32x4_t = aarch64::vdupq_n_u32(p);
                uint32x4_to_array(uint32x4_mod_sub(a, b, p))
            };

            res.copy_from_slice(&out);
        }
        5 => {
            // We fit what we can into a uint32x4_t element.
            // The final sub is done using a scalar subtraction.
            let out: [u32; 4] = unsafe {
                let a = array_to_uint32x4([a[0], a[1], a[2], a[3]]);
                let b = array_to_uint32x4([b[0], b[1], b[2], b[3]]);
                let p: uint32x4_t = aarch64::vdupq_n_u32(p);
                uint32x4_to_array(uint32x4_mod_sub(a, b, p))
            };

            res[4] = scalar_sub(a[4], b[4]);

            res[..4].copy_from_slice(&out);
        }
        6 => {
            // First 4 elements fit into a uint32x4_t, remaining 2 use scalar.
            let out: [u32; 4] = unsafe {
                let a = array_to_uint32x4([a[0], a[1], a[2], a[3]]);
                let b = array_to_uint32x4([b[0], b[1], b[2], b[3]]);
                let p: uint32x4_t = aarch64::vdupq_n_u32(p);
                uint32x4_to_array(uint32x4_mod_sub(a, b, p))
            };

            res[4] = scalar_sub(a[4], b[4]);
            res[5] = scalar_sub(a[5], b[5]);

            res[..4].copy_from_slice(&out);
        }
        7 => {
            // First 4 elements fit into a uint32x4_t, remaining 3 use scalar.
            let out: [u32; 4] = unsafe {
                let a = array_to_uint32x4([a[0], a[1], a[2], a[3]]);
                let b = array_to_uint32x4([b[0], b[1], b[2], b[3]]);
                let p: uint32x4_t = aarch64::vdupq_n_u32(p);
                uint32x4_to_array(uint32x4_mod_sub(a, b, p))
            };

            res[4] = scalar_sub(a[4], b[4]);
            res[5] = scalar_sub(a[5], b[5]);
            res[6] = scalar_sub(a[6], b[6]);

            res[..4].copy_from_slice(&out);
        }
        8 => {
            // This perfectly fits into two uint32x4_t elements.
            let (out_lo, out_hi): ([u32; 4], [u32; 4]) = unsafe {
                let p: uint32x4_t = aarch64::vdupq_n_u32(p);

                let a_lo = array_to_uint32x4([a[0], a[1], a[2], a[3]]);
                let b_lo = array_to_uint32x4([b[0], b[1], b[2], b[3]]);
                let out_lo = uint32x4_to_array(uint32x4_mod_sub(a_lo, b_lo, p));

                let a_hi = array_to_uint32x4([a[4], a[5], a[6], a[7]]);
                let b_hi = array_to_uint32x4([b[4], b[5], b[6], b[7]]);
                let out_hi = uint32x4_to_array(uint32x4_mod_sub(a_hi, b_hi, p));
                (out_lo, out_hi)
            };

            res[..4].copy_from_slice(&out_lo);
            res[4..].copy_from_slice(&out_hi);
        }
        _ => panic!("Currently unsupported width for packed subtraction"),
    }
}

#[cfg(test)]
mod tests {
    use core::arch::aarch64;

    use proptest::prelude::*;
    use proptest::test_runner::TestRunner;

    use super::*;

    // Use a prime < 2^31 for testing. KoalaBear prime: 2^31 - 2^24 + 1
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

    // ------- uint32x4_mod_add / uint32x4_mod_sub -------

    fn check_uint32x4_mod_add(a: [u32; 4], b: [u32; 4]) {
        unsafe {
            let av = array_to_uint32x4(a);
            let bv = array_to_uint32x4(b);
            let p_vec: uint32x4_t = aarch64::vdupq_n_u32(P);
            let res = uint32x4_to_array(uint32x4_mod_add(av, bv, p_vec));
            for i in 0..4 {
                assert_eq!(res[i], ref_add(a[i], b[i]), "add mismatch at index {i}");
            }
        }
    }

    fn check_uint32x4_mod_sub(a: [u32; 4], b: [u32; 4]) {
        unsafe {
            let av = array_to_uint32x4(a);
            let bv = array_to_uint32x4(b);
            let p_vec: uint32x4_t = aarch64::vdupq_n_u32(P);
            let res = uint32x4_to_array(uint32x4_mod_sub(av, bv, p_vec));
            for i in 0..4 {
                assert_eq!(res[i], ref_sub(a[i], b[i]), "sub mismatch at index {i}");
            }
        }
    }

    #[test]
    fn test_uint32x4_mod_add() {
        let mut runner = TestRunner::default();
        runner
            .run(&(array_strategy::<4>(), array_strategy::<4>()), |(a, b)| {
                check_uint32x4_mod_add(a, b);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn test_uint32x4_mod_sub() {
        let mut runner = TestRunner::default();
        runner
            .run(&(array_strategy::<4>(), array_strategy::<4>()), |(a, b)| {
                check_uint32x4_mod_sub(a, b);
                Ok(())
            })
            .unwrap();
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
        // 0 + 0 = 0
        let mut res = [0u32; 4];
        packed_mod_add(&[0, 0, 0, 0], &[0, 0, 0, 0], &mut res, P, ref_add);
        assert_eq!(res, [0, 0, 0, 0]);

        // P-1 + 1 = 0
        packed_mod_add(
            &[P - 1, P - 1, 1, 0],
            &[1, 0, P - 1, 0],
            &mut res,
            P,
            ref_add,
        );
        assert_eq!(res, [0, P - 1, 0, 0]);

        // P-1 + P-1 = P-2
        packed_mod_add(&[P - 1; 4], &[P - 1; 4], &mut res, P, ref_add);
        assert_eq!(res, [P - 2; 4]);
    }

    #[test]
    fn test_sub_boundary_values() {
        // 0 - 0 = 0
        let mut res = [0u32; 4];
        packed_mod_sub(&[0, 0, 0, 0], &[0, 0, 0, 0], &mut res, P, ref_sub);
        assert_eq!(res, [0, 0, 0, 0]);

        // 0 - 1 = P-1
        packed_mod_sub(&[0; 4], &[1; 4], &mut res, P, ref_sub);
        assert_eq!(res, [P - 1; 4]);

        // a - a = 0
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
            .run(
                &(array_strategy::<4>(), array_strategy::<4>()),
                |(a, b)| {
                    check_add_sub_inverse(a, b);
                    Ok(())
                },
            )
            .unwrap();
    }

    #[test]
    fn test_add_sub_inverse_w8() {
        let mut runner = TestRunner::default();
        runner
            .run(
                &(array_strategy::<8>(), array_strategy::<8>()),
                |(a, b)| {
                    check_add_sub_inverse(a, b);
                    Ok(())
                },
            )
            .unwrap();
    }
}
