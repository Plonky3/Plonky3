use crate::field::Field;
use core::ops::{Add, AddAssign, BitXorAssign, Mul, MulAssign};

/// The prime field `F_p` where `p = 2^31 - 1`.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Mersenne31 {
    /// Not necessarily canonical, but must fit in 31 bits.
    value: u32,
}

impl Mersenne31 {
    pub const ORDER: u32 = (1 << 31) - 1;
    /// Two's complement of `ORDER`, i.e. `2^32 - ORDER`.
    pub const NEG_ORDER: u32 = Self::ORDER.wrapping_neg();
}

impl Field for Mersenne31 {
    const ZERO: Self = Self { value: 0 };
    const ONE: Self = Self { value: 1 };
    const NEG_ONE: Self = Self {
        value: Self::ORDER - 1,
    };
}

impl Add<Self> for Mersenne31 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut sum = self.value + rhs.value;
        // If sum's most significant bit is set, we clear it and add 1, since 2^31 = 1 mod p.
        // This addition of 1 cannot overflow 2^31, since sum has a max of
        // 2 * (2^31 - 1) = 2^32 - 2.
        let msb = sum & (1 << 31);
        sum.bitxor_assign(msb);
        sum.bitxor_assign((msb != 0) as u32);
        Self { value: sum }
    }
}

impl AddAssign<Self> for Mersenne31 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Mul<Self> for Mersenne31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let prod = (self.value as u64) * (rhs.value as u64);
        let prod_lo = prod as u32;
        let prod_hi = (prod >> 32) as u32;

        // Because each value was 31 bits, prod_hi is at most 30 bits.
        // So we can apply its weight of 2 (since 2^32 = 2 mod p) without overflow.
        let prod_hi_weighted = prod_hi << 1;

        let (sum, over) = prod_lo.overflowing_add(prod_hi_weighted);
        let (sum, _) = sum.overflowing_add((over as u32) * Self::NEG_ORDER);
        // TODO: Clear most significant bit.
        Self { value: sum }
    }
}

impl MulAssign<Self> for Mersenne31 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

// #[derive(Copy, Clone, Eq, PartialEq, Debug)]
// pub struct Mersenne31Arr<const N: usize> {
//     values: [Mersenne31; N],
// }
//
// impl<const N: usize> FieldArr<N, Mersenne31> for Mersenne31Arr<N> {}
//
// impl<const N: usize> Add<Self> for Mersenne31Arr<N> {
//     type Output = Self;
//
//     fn add(self, rhs: Self) -> Self {
//         // TODO: Naive for now.
//         let values = std::array::from_fn(|i| self.values[i] + rhs.values[i]);
//         Self { values }
//     }
// }
//
// impl<const N: usize> Mul<Mersenne31> for Mersenne31Arr<N> {
//     type Output = Self;
//
//     fn mul(self, rhs: Mersenne31) -> Self {
//         // TODO: Naive for now.
//         let values = self.values.map(|x| x * rhs);
//         Self { values }
//     }
// }
//
// impl<const N: usize> Mul<Self> for Mersenne31Arr<N> {
//     type Output = Self;
//
//     fn mul(self, rhs: Self) -> Self {
//         // TODO: Naive for now.
//         let values = std::array::from_fn(|i| self.values[i] * rhs.values[i]);
//         Self { values }
//     }
// }
