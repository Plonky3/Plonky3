use core::arch::aarch64::{int32x4_t, uint32x4_t};
use core::mem::transmute;

use p3_monty_31::{MontyParametersNeon, PackedMontyField31Neon};

use crate::KoalaBearParameters;

const WIDTH: usize = 4;

impl MontyParametersNeon for KoalaBearParameters {
    const PACKED_P: uint32x4_t = unsafe { transmute::<[u32; WIDTH], _>([0x7f000001; WIDTH]) };
    // This MU is the same 0x88000001 as elsewhere, just interpreted as an `i32`.
    const PACKED_MU: int32x4_t = unsafe { transmute::<[i32; WIDTH], _>([-0x7effffff; WIDTH]) };
}

pub type PackedKoalaBearNeon = PackedMontyField31Neon<KoalaBearParameters>;

#[cfg(test)]
mod tests {
    use p3_field::PrimeField32;
    use p3_field_testing::{assert_broadcast_dot_product_matches_scalar, test_packed_field};

    use super::WIDTH;
    use crate::KoalaBear;

    const SPECIAL_VALS: [KoalaBear; WIDTH] =
        KoalaBear::new_array([0x00000000, 0x00000001, 0x00000002, 0x7f000000]);

    test_packed_field!(
        crate::PackedKoalaBearNeon,
        &[crate::PackedKoalaBearNeon::ZERO],
        &[crate::PackedKoalaBearNeon::ONE],
        p3_monty_31::PackedMontyField31Neon::<crate::KoalaBearParameters>(super::SPECIAL_VALS)
    );

    fn to_monty(value: u32, prime: u32) -> u32 {
        (((value as u64) << 32) % (prime as u64)) as u32
    }

    fn reduce_once(value: u32, prime: u32) -> u32 {
        let sub = value.wrapping_sub(prime);
        value.min(sub)
    }

    fn assert_dot_product_5_carry_critical(lhs: [u32; 5], rhs: [u32; 5], prime: u32) {
        let mut products = [0_u64; 5];
        for i in 0..5 {
            let lhs_monty = to_monty(lhs[i], prime);
            let rhs_monty = to_monty(rhs[i], prime);
            products[i] = (lhs_monty as u64) * (rhs_monty as u64);
        }

        let sum_a = products[0] + products[1] + products[2];
        let sum_b = products[3] + products[4];

        let c_lo_a = sum_a as u32;
        let c_hi_a = (sum_a >> 32) as u32;
        let c_lo_b = sum_b as u32;
        let c_hi_b = (sum_b >> 32) as u32;

        let c_hi_a_red = reduce_once(c_hi_a, prime);
        let (c_lo, carry) = c_lo_a.overflowing_add(c_lo_b);
        let c_hi_sum = c_hi_a_red + c_hi_b;
        let c_hi_red = reduce_once(c_hi_sum, prime);
        let c_hi_prime = c_hi_red + u32::from(carry);

        assert!(
            carry,
            "dot_product_5 carry-critical case should overflow c_lo"
        );
        assert_eq!(
            c_hi_prime, prime,
            "dot_product_5 carry-critical case should hit c_hi_prime = P before final reduction",
        );
        assert!(
            c_lo < c_lo_a,
            "carry flag and c_lo relationship should match"
        );
    }

    fn assert_dot_product_8_carry_critical(lhs: [u32; 8], rhs: [u32; 8], prime: u32) {
        let mut products = [0_u64; 8];
        for i in 0..8 {
            let lhs_monty = to_monty(lhs[i], prime);
            let rhs_monty = to_monty(rhs[i], prime);
            products[i] = (lhs_monty as u64) * (rhs_monty as u64);
        }

        let sum_a = products[0] + products[1] + products[2] + products[3];
        let sum_b = products[4] + products[5] + products[6] + products[7];

        let c_lo_a = sum_a as u32;
        let c_hi_a = (sum_a >> 32) as u32;
        let c_lo_b = sum_b as u32;
        let c_hi_b = (sum_b >> 32) as u32;

        let c_hi_a_red = reduce_once(c_hi_a, prime);
        let c_hi_b_red = reduce_once(c_hi_b, prime);
        let (_, carry) = c_lo_a.overflowing_add(c_lo_b);
        let c_hi = c_hi_a_red + c_hi_b_red + u32::from(carry);
        let c_hi_prime = reduce_once(c_hi, prime);

        assert!(
            carry,
            "dot_product_8 carry-critical case should overflow c_lo"
        );
        assert_eq!(
            c_hi_prime,
            prime - 1,
            "dot_product_8 carry-critical case should hit c_hi_prime = P-1",
        );
    }

    #[test]
    fn test_dot_product_5_carry_boundary_case() {
        const P: u32 = KoalaBear::ORDER_U32;
        let lhs_raw = [P - 1, 1, 8, P - 3, P - 2];
        let rhs_raw = [P - 4, 9, P - 2, P - 5, 6];
        assert_dot_product_5_carry_critical(lhs_raw, rhs_raw, P);

        assert_broadcast_dot_product_matches_scalar::<crate::PackedKoalaBearNeon, 5>(
            KoalaBear::new_array(lhs_raw),
            KoalaBear::new_array(rhs_raw),
            "KoalaBear NEON dot_product_5 carry-critical",
        );
    }

    #[test]
    fn test_dot_product_8_carry_boundary_case() {
        const P: u32 = KoalaBear::ORDER_U32;
        const HALF_PLUS_ONE: u32 = (P / 2) + 1;
        let lhs_raw = [HALF_PLUS_ONE, P - 2, 6, 0, P - 1, 5, 6, P - 1];
        let rhs_raw = [9, 5, 2, 7, P - 3, P - 1, 9, P - 5];
        assert_dot_product_8_carry_critical(lhs_raw, rhs_raw, P);

        assert_broadcast_dot_product_matches_scalar::<crate::PackedKoalaBearNeon, 8>(
            KoalaBear::new_array(lhs_raw),
            KoalaBear::new_array(rhs_raw),
            "KoalaBear NEON dot_product_8 carry-critical",
        );
    }
}
