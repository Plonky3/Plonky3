//! The prime field `F_p` where `p = 2^64 - 2^32 + 1`.

#![no_std]

use core::fmt;
use core::fmt::{Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, BitXorAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use p3_field::field::{Field, PrimeField};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

/// The prime field `F_p` where `p = 2^64 - 2^32 + 1`.
#[derive(Copy, Clone, Debug, Default)]
pub struct Goldilocks {
    /// Not necessarily canonical, but must fit in 31 bits.
    value: u64,
}

impl Goldilocks {
    pub const ORDER: u64 = 0xFFFFFFFF00000001;

    /// Two's complement of `ORDER`, i.e. `2^64 - ORDER = 2^32 - 1`.
    const NEG_ORDER: u64 = Self::ORDER.wrapping_neg();

    fn as_canonical_u64(&self) -> u64 {
        let mut c = self.value;
        // We only need one condition subtraction, since 2 * ORDER would not fit in a u64.
        if c >= Self::ORDER {
            c -= Self::ORDER;
        }
        c
    }
}

impl PartialEq for Goldilocks {
    fn eq(&self, other: &Self) -> bool {
        self.as_canonical_u64() == other.as_canonical_u64()
    }
}

impl Eq for Goldilocks {}
