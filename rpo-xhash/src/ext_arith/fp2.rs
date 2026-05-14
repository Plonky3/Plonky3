//! F_p^2 arithmetic for the extension S-box.
//!
//! Binomial extension: α² = W. Element [a, b] = a + b·α.
//! Karatsuba mul: 3 base muls, 2 reductions.
//! Square: 2 base muls + 1 double, 2 reductions.
//!
//! Only the variants actually used by the crate are kept (BabyBear x^7 / Fp²
//! for xHash-BabyBear).

// ============================================================
// BabyBear: α² = 11, d=7. Barrett reduction.
// ============================================================

pub mod babybear {
    use crate::reduce::monty31::reduce_bb;
    const W: u64 = 11;

    #[inline(always)]
    pub fn mul(x: [u32; 2], y: [u32; 2]) -> [u32; 2] {
        let ac = x[0] as u64 * y[0] as u64;
        let bd = x[1] as u64 * y[1] as u64;
        let cross = (x[0] as u64 + x[1] as u64) * (y[0] as u64 + y[1] as u64);
        let ad_bc = cross - ac - bd;
        let c0 = ac + reduce_bb(bd) as u64 * W;
        [reduce_bb(c0), reduce_bb(ad_bc)]
    }

    #[inline(always)]
    pub fn sqr(x: [u32; 2]) -> [u32; 2] {
        let a2 = x[0] as u64 * x[0] as u64;
        let b2 = x[1] as u64 * x[1] as u64;
        let ab2 = 2 * (x[0] as u64 * x[1] as u64);
        let c0 = a2 + reduce_bb(b2) as u64 * W;
        [reduce_bb(c0), reduce_bb(ab2)]
    }

    /// x^7 = (x³)² · x = ((x²·x))² · x. Cost: 2 sqr + 2 mul.
    #[inline(always)]
    pub fn pow7(z: [u32; 2]) -> [u32; 2] {
        let z2 = sqr(z);
        let z3 = mul(z2, z);
        let z6 = sqr(z3);
        mul(z6, z)
    }
}

#[cfg(test)]
mod tests {
    use super::babybear;
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, PrimeField32};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn babybear_fp2_matches_naive_field_ops() {
        #[inline(always)]
        fn mul_w(x: BabyBear) -> BabyBear {
            let x2 = x.double();
            let x4 = x2.double();
            let x8 = x4.double();
            x8 + x2 + x
        }

        #[inline(always)]
        fn mul(x: [BabyBear; 2], y: [BabyBear; 2]) -> [BabyBear; 2] {
            let ac = x[0] * y[0];
            let bd = x[1] * y[1];
            let cross = (x[0] + x[1]) * (y[0] + y[1]);
            [ac + mul_w(bd), cross - ac - bd]
        }

        #[inline(always)]
        fn sqr(x: [BabyBear; 2]) -> [BabyBear; 2] {
            let a2 = x[0].square();
            let b2 = x[1].square();
            let ab2 = (x[0] * x[1]).double();
            [a2 + mul_w(b2), ab2]
        }

        let mut rng = StdRng::seed_from_u64(101);
        for _ in 0..100 {
            let z = [
                rng.random::<u32>() % BabyBear::ORDER_U32,
                rng.random::<u32>() % BabyBear::ORDER_U32,
            ];
            let zf = [BabyBear::new(z[0]), BabyBear::new(z[1])];
            let z2 = sqr(zf);
            let z3 = mul(z2, zf);
            let z6 = sqr(z3);
            let got = mul(z6, zf);
            let expected = babybear::pow7(z);
            assert_eq!(got[0].as_canonical_u32(), expected[0]);
            assert_eq!(got[1].as_canonical_u32(), expected[1]);
        }
    }
}
