use core::fmt;
use core::fmt::{Debug, Display, Formatter};
use core::hash::Hash;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{AbstractField, Field, TwoAdicField};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

#[derive(Copy, Clone, Default, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Fp17 {
    v: u32,
}

impl Fp17 {
    const ORDER: u32 = 17;
    pub fn new(v: u32) -> Self {
        Fp17 { v: v % 17 }
    }
}

impl Display for Fp17 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.v, f)
    }
}

impl Distribution<Fp17> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Fp17 {
        Fp17::new(rng.gen_range(0..17))
    }
}

impl Product for Fp17 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::new(1), |a, b| a * b)
    }
}

impl Sum for Fp17 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::new(0), |a, b| a + b)
    }
}

impl TwoAdicField for Fp17 {
    const TWO_ADICITY: usize = 4;
    fn power_of_two_generator() -> Self {
        Self::new(3)
    }
}

impl Add for Fp17 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.v + rhs.v)
    }
}
impl AddAssign for Fp17 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl Sub for Fp17 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(Self::ORDER + self.v - rhs.v)
    }
}
impl SubAssign for Fp17 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl Neg for Fp17 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(Self::ORDER - self.v)
    }
}
impl Mul for Fp17 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.v * rhs.v)
    }
}
impl MulAssign for Fp17 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl Div for Fp17 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::new(self.v.mul(rhs.inverse().v))
    }
}
impl DivAssign for Fp17 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
impl Field for Fp17 {
    type Packing = Self;
    fn try_inverse(&self) -> Option<Self> {
        let mut t: i32 = 0;
        let mut newt: i32 = 1;
        let mut r: i32 = Self::ORDER as i32;
        let mut newr: i32 = self.v as i32;
        while newr != 0 {
            let quotient = r / newr;
            let oldt = t;
            t = newt;
            newt = oldt - quotient * newt;
            let oldr = r;
            r = newr;
            newr = oldr - quotient * newr;
        }
        if r > 1 {
            return None;
        }
        if t < 0 {
            t += Self::ORDER as i32;
        }
        Some(Self::new(t as u32))
    }
}

impl AbstractField for Fp17 {
    const ZERO: Self = Self { v: 0 };
    const ONE: Self = Self { v: 1 };
    const TWO: Self = Self { v: 2 };
    const NEG_ONE: Self = Self { v: 16 };
    fn multiplicative_group_generator() -> Self {
        Self::new(3)
    }
    fn from_bool(b: bool) -> Self {
        if b {
            Self::ONE
        } else {
            Self::ZERO
        }
    }
    fn from_canonical_u8(n: u8) -> Self {
        Self::new(u32::from(n))
    }
    fn from_canonical_u16(n: u16) -> Self {
        Self::new(u32::from(n))
    }
    fn from_canonical_u32(n: u32) -> Self {
        Self::new(n)
    }
    fn from_canonical_u64(n: u64) -> Self {
        Self::new((n % (Self::ORDER as u64)) as u32)
    }
    fn from_canonical_usize(n: usize) -> Self {
        Self::new((n % (Self::ORDER as usize)) as u32)
    }
    fn from_wrapped_u32(n: u32) -> Self {
        Self::from_canonical_u32(n)
    }
    fn from_wrapped_u64(n: u64) -> Self {
        Self::from_canonical_u64(n)
    }
}
