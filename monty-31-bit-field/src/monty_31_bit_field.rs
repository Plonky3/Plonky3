//! An abstraction of 31-bit fields which use a MONTY approach for addition and multiplication with a MONTY constant = 2^32.

use core::fmt::{Debug};

pub trait MONTY31BitAbstractField: 
    Sized
    + Default
    + Clone
    + Debug
{

    const P: u32;
    const MONTY_BITS: u32;
    const MONTY_MU: u32;
    const MONTY_MASK: u32 = ((1u64 << Self::MONTY_BITS) - 1) as u32;

    const MONTY_ZERO: u32 = 0; // The monty form of 0 is always 0.
    const MONTY_ONE: u32;
    const MONTY_TWO: u32;
    const MONTY_NEG_ONE: u32 = Self::P - Self::MONTY_ONE; // As MONTY_ONE =/= 0, MONTY_NEG_ONE = P - MONTY_ONE.

    const MONTY_GEN: u32; // A generator of the fields multiplicative group.

    /// Compute lhs + rhs mod P.
    /// Assumes both inputs lie in 0..P
    /// Returns an output in 0..P
    #[inline]
    fn add_u32(lhs: u32, rhs: u32) -> u32 {
        let mut sum = lhs + rhs;
        let (corr_sum, over) = sum.overflowing_sub(Self::P);
        if !over {
            sum = corr_sum;
        }
        sum
    }

    /// Compute lhs - rhs mod P.
    /// Assumes both inputs lie in 0..P
    /// Returns an output in 0..P
    #[inline]
    fn sub_u32(lhs: u32, rhs: u32) -> u32 {
        let (mut diff, over) = lhs.overflowing_sub(rhs);
        let corr = if over { Self::P } else { 0 };
        diff = diff.wrapping_add(corr);
        diff
    }

    /// Compute lhs*rhs mod P.
    /// Assumes both inputs lie in 0..P and are saved in MONTY form
    /// Returns an output in 0..P also in MONTY form
    #[inline]
    fn mul_u32_monty(lhs: u32, rhs: u32) -> u32 {
        let long_prod = lhs as u64 * rhs as u64;
        Self::monty_reduce(long_prod)
    }

    #[inline]
    fn to_monty(x: u32) -> u32 {
        (((x as u64) << Self::MONTY_BITS) % Self::P as u64) as u32
    }

    #[inline]
    fn to_monty_64(x: u64) -> u32 {
        (((x as u128) << Self::MONTY_BITS) % Self::P as u128) as u32
    }

    /// Montgomery reduction of a value in `0..P << MONTY_BITS`.
    #[inline]
    #[must_use]
    fn monty_reduce(x: u64) -> u32 {
        let t = x.wrapping_mul(Self::MONTY_MU as u64) & (Self::MONTY_MASK as u64);
        let u = t * (Self::P as u64);

        let (x_sub_u, over) = x.overflowing_sub(u);
        let x_sub_u_hi = (x_sub_u >> Self::MONTY_BITS) as u32;
        let corr = if over { Self::P } else { 0 };
        x_sub_u_hi.wrapping_add(corr)
    }
}