use core::fmt::{Debug, Display};
use core::hash::Hash;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use core::slice;

use p3_util::log2_ceil_u64;

use crate::packed::PackedField;

/// A generalization of `Field` which permits things like
/// - an actual field element
/// - a symbolic expression which would evaluate to a field element
/// - a vector of field elements
pub trait AbstractField:
    Sized
    + Default
    + Clone
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Neg<Output = Self>
    + Mul<Output = Self>
    + MulAssign
    + Sum
    + Product
    + Debug
{
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const NEG_ONE: Self;

    fn from_bool(b: bool) -> Self;
    fn from_canonical_u8(n: u8) -> Self;
    fn from_canonical_u16(n: u16) -> Self;
    fn from_canonical_u32(n: u32) -> Self;
    fn from_canonical_u64(n: u64) -> Self;
    fn from_canonical_usize(n: usize) -> Self;

    fn from_wrapped_u32(n: u32) -> Self;
    fn from_wrapped_u64(n: u64) -> Self;

    fn multiplicative_group_generator() -> Self;

    #[must_use]
    fn double(&self) -> Self {
        self.clone() + self.clone()
    }

    #[must_use]
    fn square(&self) -> Self {
        self.clone() * self.clone()
    }

    #[must_use]
    fn cube(&self) -> Self {
        self.square() * self.clone()
    }

    #[must_use]
    fn powers(&self) -> Powers<Self> {
        Powers {
            base: self.clone(),
            current: Self::ONE,
        }
    }

    fn dot_product<const N: usize>(u: &[Self; N], v: &[Self; N]) -> Self {
        u.iter().zip(v).map(|(x, y)| x.clone() * y.clone()).sum()
    }
}

/// An `AbstractField` which abstracts the given field `F`.
pub trait AbstractionOf<F: Field>:
    AbstractField
    + From<F>
    + Add<F, Output = Self>
    + AddAssign<F>
    + Sub<F, Output = Self>
    + SubAssign<F>
    + Mul<F, Output = Self>
    + MulAssign<F>
    + Sum<F>
    + Product<F>
{
}

impl<F: Field> AbstractionOf<F> for F {}

/// An element of a finite field.
pub trait Field:
    AbstractField + 'static + Copy + Div<Self, Output = Self> + Eq + Hash + Send + Sync + Display
{
    type Packing: PackedField<Scalar = Self>;

    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    fn is_one(&self) -> bool {
        *self == Self::ONE
    }

    /// self * 2^exp
    #[must_use]
    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        *self * Self::TWO.exp_u64(exp)
    }

    /// self / 2^exp
    #[must_use]
    #[inline]
    fn div_2exp_u64(&self, exp: u64) -> Self {
        *self / Self::TWO.exp_u64(exp)
    }

    /// The multiplicative inverse of this field element, if it exists.
    ///
    /// NOTE: The inverse of `0` is undefined and will return `None`.
    #[must_use]
    fn try_inverse(&self) -> Option<Self>;

    #[must_use]
    fn inverse(&self) -> Self {
        self.try_inverse().expect("Tried to invert zero")
    }

    #[must_use]
    #[inline(always)]
    fn exp_const_u64<const POWER: u64>(&self) -> Self {
        match POWER {
            0 => Self::ONE,
            1 => *self,
            2 => self.square(),
            3 => self.cube(),
            4 => self.square().square(),
            5 => self.square().square() * *self,
            6 => self.square().cube(),
            7 => {
                let x2 = self.square();
                let x3 = x2 * *self;
                let x4 = x2.square();
                x3 * x4
            }
            _ => self.exp_u64(POWER),
        }
    }

    // Default naive square and multiply implementation for powers.
    #[must_use]
    #[inline]
    fn exp_u64_default(&self, power: u64) -> Self {
        let mut current = *self;
        let mut product = Self::ONE;

        for j in 0..bits_u64(power) {
            if (power >> j & 1) != 0 {
                product *= current;
            }
            current = current.square();
        }
        product
    }

    // For specific fields we hard code some addition chains whilst defaulting back to the naive implementation in other cases.
    #[must_use]
    fn exp_u64(&self, power: u64) -> Self {
        self.exp_u64_default(power)
    }

    #[must_use]
    fn exp_power_of_2(&self, power_log: usize) -> Self {
        let mut res = *self;
        for _ in 0..power_log {
            res = res.square();
        }
        res
    }
}

pub trait PrimeField: Field + Ord {}

/// A prime field of order less than `2^64`.
pub trait PrimeField64: PrimeField {
    const ORDER_U64: u64;

    fn bits() -> usize {
        log2_ceil_u64(Self::ORDER_U64) as usize
    }

    /// Return the representative of `value` that is less than `ORDER_U64`.
    fn as_canonical_u64(&self) -> u64;

    /// Return the value \sum_{i=0}^N u[i] * v[i].
    ///
    /// NB: Assumes that sum(u) <= 2^32 to allow implementations to avoid
    /// overflow handling.
    ///
    /// TODO: Mark unsafe because of the assumption?
    fn linear_combination_u64<const N: usize>(u: [u64; N], v: &[Self; N]) -> Self;
}

/// A prime field of order less than `2^32`.
pub trait PrimeField32: PrimeField64 {
    const ORDER_U32: u32;

    /// Return the representative of `value` that is less than `ORDER_U32`.
    fn as_canonical_u32(&self) -> u32;
}

pub trait AbstractExtensionField<Base>:
    AbstractField
    + Add<Base, Output = Self>
    + AddAssign<Base>
    + Sub<Base, Output = Self>
    + SubAssign<Base>
    + Mul<Base, Output = Self>
    + MulAssign<Base>
{
    const D: usize;

    fn from_base(b: Base) -> Self;

    /// Suppose this field extension is represented by the quotient
    /// ring B[X]/(f(X)) where B is `Base` and f is an irreducible
    /// polynomial of degree `D`. This function takes a slice `bs` of
    /// length at most D, and constructs the field element
    /// \sum_i bs[i] * X^i.
    ///
    /// NB: The value produced by this function fundamentally depends
    /// on the choice of irreducible polynomial f. Care must be taken
    /// to ensure portability if these values might ever be passed to
    /// (or rederived within) another compilation environment where a
    /// different f might have been used.
    fn from_base_slice(bs: &[Base]) -> Self;

    /// Suppose this field extension is represented by the quotient
    /// ring B[X]/(f(X)) where B is `Base` and f is an irreducible
    /// polynomial of degree `D`. This function takes a field element
    /// \sum_i bs[i] * X^i and returns the coefficients as a slice
    /// `bs` of length at most D containing, from lowest degree to
    /// highest.
    ///
    /// NB: The value produced by this function fundamentally depends
    /// on the choice of irreducible polynomial f. Care must be taken
    /// to ensure portability if these values might ever be passed to
    /// (or rederived within) another compilation environment where a
    /// different f might have been used.
    fn as_base_slice(&self) -> &[Base];
}

pub trait ExtensionField<Base: Field>: Field + AbstractExtensionField<Base> {
    fn is_in_basefield(&self) -> bool {
        self.as_base_slice()[1..].iter().all(|x| x.is_zero())
    }
}

impl<F: Field> ExtensionField<F> for F {}

impl<F: AbstractField> AbstractExtensionField<F> for F {
    const D: usize = 1;

    fn from_base(b: F) -> Self {
        b
    }

    fn from_base_slice(bs: &[F]) -> Self {
        assert_eq!(bs.len(), 1);
        bs[0].clone()
    }

    fn as_base_slice(&self) -> &[F] {
        slice::from_ref(self)
    }
}

/// A field which supplies information like the two-adicity of its multiplicative group, and methods
/// for obtaining two-adic roots of unity.
pub trait TwoAdicField: Field {
    /// The number of factors of two in this field's multiplicative group.
    const TWO_ADICITY: usize;

    /// Generator of a multiplicative subgroup of order `2^TWO_ADICITY`.
    fn power_of_two_generator() -> Self;

    /// Returns a primitive root of order `2^bits`.
    #[must_use]
    fn primitive_root_of_unity(bits: usize) -> Self {
        assert!(bits <= Self::TWO_ADICITY);
        let base = Self::power_of_two_generator();
        base.exp_power_of_2(Self::TWO_ADICITY - bits)
    }
}

/// An iterator over the powers of a certain base element `b`: `b^0, b^1, b^2, ...`.
#[derive(Clone)]
pub struct Powers<F> {
    pub base: F,
    pub current: F,
}

impl<F: AbstractField> Iterator for Powers<F> {
    type Item = F;

    fn next(&mut self) -> Option<F> {
        let result = self.current.clone();
        self.current *= self.base.clone();
        Some(result)
    }
}

const fn bits_u64(n: u64) -> usize {
    (64 - n.leading_zeros()) as usize
}
