//! F_p^3 arithmetic for M31, KoalaBear, and Goldilocks.
//!
//! M31:        binomial α³ = 5, d=5.
//! KoalaBear:  trinomial α³ + α + 4 = 0, d=3.
//! Goldilocks: binomial α³ = 2, d=7.

// ============================================================
// M31 F_p^3: binomial α³ = 5. Karatsuba: 6 muls, 3 reductions.
// ============================================================

pub mod m31 {
    use crate::reduce::m31::reduce;
    const W: u64 = 5;

    #[inline(always)]
    pub fn mul(x: [u32; 3], y: [u32; 3]) -> [u32; 3] {
        let a0b0 = x[0] as u64 * y[0] as u64;
        let a1b1 = x[1] as u64 * y[1] as u64;
        let a2b2 = x[2] as u64 * y[2] as u64;

        let cross01 = (x[1] as u64 + x[2] as u64) * (y[1] as u64 + y[2] as u64) - a1b1 - a2b2;
        let c0 = a0b0 + reduce(cross01) as u64 * W;

        let cross1 = (x[0] as u64 + x[1] as u64) * (y[0] as u64 + y[1] as u64) - a0b0 - a1b1;
        let c1 = cross1 + reduce(a2b2) as u64 * W;

        let cross2 = (x[0] as u64 + x[2] as u64) * (y[0] as u64 + y[2] as u64) - a0b0 - a2b2;
        let c2 = cross2 + a1b1;

        [reduce(c0), reduce(c1), reduce(c2)]
    }

    #[inline(always)]
    pub fn sqr(x: [u32; 3]) -> [u32; 3] {
        let a2 = x[2] as u64;
        let w_a2 = reduce((a2 << 2) + a2); // 5 * a2 via shift+add
        let a0sq = x[0] as u64 * x[0] as u64;
        let a1sq = x[1] as u64 * x[1] as u64;
        let a1_wa2 = x[1] as u64 * w_a2 as u64;
        let a0_a1 = x[0] as u64 * x[1] as u64;
        let wa2_a2 = w_a2 as u64 * x[2] as u64;
        let a0_a2 = x[0] as u64 * x[2] as u64;
        [
            reduce(a0sq + 2 * a1_wa2),
            reduce(wa2_a2 + 2 * a0_a1),
            reduce(a1sq + 2 * a0_a2),
        ]
    }
}

// ============================================================
// KoalaBear F_p^3: trinomial α³ + α + 4 = 0. d=3 (cheapest S-box).
// Uses Montgomery-native KoalaBear field ops to stay in Monty form.
// ============================================================

pub mod koalabear_field {
    use p3_field::PrimeCharacteristicRing;
    use p3_koala_bear::KoalaBear;

    #[inline(always)]
    fn mul_w0(x: KoalaBear) -> KoalaBear {
        x.double().double() // 4 * x
    }

    #[inline(always)]
    pub fn mul(x: [KoalaBear; 3], y: [KoalaBear; 3]) -> [KoalaBear; 3] {
        let a0b0 = x[0] * y[0];
        let a1b1 = x[1] * y[1];
        let a2b2 = x[2] * y[2];

        let c1 = (x[0] + x[1]) * (y[0] + y[1]) - a0b0 - a1b1;
        let c2 = (x[0] + x[2]) * (y[0] + y[2]) - a0b0 - a2b2 + a1b1;
        let c3 = (x[1] + x[2]) * (y[1] + y[2]) - a1b1 - a2b2;

        [a0b0 - mul_w0(c3), c1 - c3 - mul_w0(a2b2), c2 - a2b2]
    }

    #[inline(always)]
    pub fn sqr(x: [KoalaBear; 3]) -> [KoalaBear; 3] {
        mul(x, x)
    }
}

// ============================================================
// Goldilocks F_p^3: binomial α³ = 2, d=7.
// ============================================================

pub mod goldilocks {
    use p3_field::PrimeCharacteristicRing;
    use p3_goldilocks::Goldilocks;

    #[inline(always)]
    fn mul_w(x: Goldilocks) -> Goldilocks {
        x.double() // W = 2
    }

    /// (a0 + a1*α + a2*α²) * (b0 + b1*α + b2*α²) mod α³ = 2.
    #[inline(always)]
    pub fn mul(x: [Goldilocks; 3], y: [Goldilocks; 3]) -> [Goldilocks; 3] {
        let [a0, a1, a2] = x;
        let [b0, b1, b2] = y;

        let c0 = a0 * b0 + mul_w(a1 * b2 + a2 * b1);
        let c1 = a0 * b1 + a1 * b0 + mul_w(a2 * b2);
        let c2 = a0 * b2 + a1 * b1 + a2 * b0;

        [c0, c1, c2]
    }

    #[inline(always)]
    pub fn sqr(x: [Goldilocks; 3]) -> [Goldilocks; 3] {
        let [a0, a1, a2] = x;

        let c0 = a0.square() + mul_w((a1 * a2).double());
        let c1 = (a0 * a1).double() + mul_w(a2.square());
        let c2 = a1.square() + (a0 * a2).double();

        [c0, c1, c2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m31_fp3_sqr_basic() {
        // (1 + α)² mod α³=5: 1 + 2α + α²
        let z2 = m31::sqr([1, 1, 0]);
        assert_eq!(z2, [1, 2, 1]);
    }

    #[test]
    fn m31_fp3_mul_matches_sqr() {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};
        let mut rng = StdRng::seed_from_u64(77);
        let p = crate::reduce::m31::P as u32;
        for _ in 0..100 {
            let z = [
                rng.random::<u32>() % p,
                rng.random::<u32>() % p,
                rng.random::<u32>() % p,
            ];
            assert_eq!(m31::mul(z, z), m31::sqr(z));
        }
    }

    #[test]
    fn kb_fp3_field_mul_matches_sqr() {
        use p3_field::PrimeField32;
        use p3_koala_bear::KoalaBear;
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};
        let mut rng = StdRng::seed_from_u64(88);
        for _ in 0..100 {
            let z = [
                KoalaBear::new(rng.random::<u32>() % KoalaBear::ORDER_U32),
                KoalaBear::new(rng.random::<u32>() % KoalaBear::ORDER_U32),
                KoalaBear::new(rng.random::<u32>() % KoalaBear::ORDER_U32),
            ];
            assert_eq!(koalabear_field::mul(z, z), koalabear_field::sqr(z));
        }
    }
}
