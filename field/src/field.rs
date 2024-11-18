use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{Debug, Display};
use core::hash::Hash;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use core::slice;

use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::One;
use nums::{Factorizer, FactorizerFromSplitter, MillerRabin, PollardRho};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::exponentiation::exp_u64_by_squaring;
use crate::packed::{PackedField, PackedValue};
use crate::Packable;

/// A commutative ring.
pub trait CommutativeRing:
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
    /// The additive identity of the ring.
    ///
    /// For every element `a` in the ring we require the following properties:
    ///
    /// `a + ZERO = ZERO + a = a,`
    ///
    /// `a + (-a) = (-a) + a = ZERO.`
    const ZERO: Self;

    /// The multiplicative identity of the ring
    ///
    /// For every element `a` in the ring we require the following property:
    ///
    /// `a*ONE = ONE*a = a.`
    const ONE: Self;

    /// The elementary function `double(a) = 2*a`.
    ///
    /// This function should be preferred over calling `a + a` or `TWO * a` as a faster implementation may be available for some algebras.
    /// If the field has characteristic 2 then this returns 0.
    #[must_use]
    fn double(&self) -> Self {
        self.clone() + self.clone()
    }

    /// The elementary function `square(a) = a^2`.
    ///
    /// This function should be preferred over calling `a * a`, as a faster implementation may be available for some algebras.
    #[must_use]
    fn square(&self) -> Self {
        self.clone() * self.clone()
    }

    /// The elementary function `cube(a) = a^3`.
    ///
    /// This function should be preferred over calling `a * a * a`, as a faster implementation may be available for some algebras.
    #[must_use]
    fn cube(&self) -> Self {
        self.square() * self.clone()
    }

    /// Compute self^{2^power_log} by repeated squaring.
    #[must_use]
    fn exp_power_of_2(&self, power_log: usize) -> Self {
        let mut res = self.clone();
        for _ in 0..power_log {
            res = res.square();
        }
        res
    }

    /// Compute the dot product of two vectors.
    fn dot_product<const N: usize>(u: &[Self; N], v: &[Self; N]) -> Self {
        u.iter().zip(v).map(|(x, y)| x.clone() * y.clone()).sum()
    }

    /// Allocates a vector of zero elements of length `len`. Many operating systems zero pages
    /// before assigning them to a userspace process. In that case, our process should not need to
    /// write zeros, which would be redundant. However, the compiler may not always recognize this.
    ///
    /// In particular, `vec![Self::ZERO; len]` appears to result in redundant userspace zeroing.
    /// This is the default implementation, but implementors may wish to provide their own
    /// implementation which transmutes something like `vec![0u32; len]`.
    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        vec![Self::ZERO; len]
    }
}

/// A commutative ring `(R)` with prime characteristic `(p)`.
///
/// The characteristic is the unique smallest integer `r > 0` such that `0 = r . 1 = 1 + 1 + ... + 1 (r times)`.
/// When the characteristic is prime, the ring `R` becomes an algebra over the field `ℤ/p` (Integers mod p).
pub trait PrimeCharacteristicRing: CommutativeRing {
    /// The field `ℤ/p`.
    type Char: PrimeField;

    /// Embed an element of the prime field `ℤ/p` into the ring `R`.
    ///
    /// Given any integer `r ∈ ℤ`, `from_char(r mod p)` will be equal to:
    ///
    /// `Self::ONE + ... + Self::ONE (r mod p times)`
    fn from_char(f: Self::Char) -> Self;

    /// Return `Self::ONE` if `b` is `true` and `Self::ZERO` if `b` is `false`.
    fn from_bool(b: bool) -> Self {
        // Some rings might reimplement this to avoid the branch.
        if b {
            Self::ONE
        } else {
            Self::ZERO
        }
    }

    /// Given an integer `r`, return the sum of `r` copies of `ONE`:
    ///
    /// `r.Self::ONE =  Self::ONE + ... + Self::ONE (r times)`.
    ///
    /// Note that the output only depends on `r mod p`.
    fn from_u8(int: u8) -> Self {
        Self::from_char(Self::Char::from_int(int))
    }

    /// Given an integer `r`, return the sum of `r` copies of `ONE`:
    ///
    /// `r.Self::ONE =  Self::ONE + ... + Self::ONE (r times)`.
    ///
    /// Note that the output only depends on `r mod p`.
    fn from_u16(int: u8) -> Self {
        Self::from_char(Self::Char::from_int(int))
    }

    /// Given an integer `r`, return the sum of `r` copies of `ONE`:
    ///
    /// `r.Self::ONE =  Self::ONE + ... + Self::ONE (r times)`.
    ///
    /// Note that the output only depends on `r mod p`.
    fn from_u32(int: u8) -> Self {
        Self::from_char(Self::Char::from_int(int))
    }

    /// Given an integer `r`, return the sum of `r` copies of `ONE`:
    ///
    /// `r.Self::ONE =  Self::ONE + ... + Self::ONE (r times)`.
    ///
    /// Note that the output only depends on `r mod p`.
    fn from_u64(int: u8) -> Self {
        Self::from_char(Self::Char::from_int(int))
    }

    /// Given an integer `r`, return the sum of `r` copies of `ONE`:
    ///
    /// `r.Self::ONE =  Self::ONE + ... + Self::ONE (r times)`.
    ///
    /// Note that the output only depends on `r mod p`.
    fn from_usize(int: u8) -> Self {
        Self::from_char(Self::Char::from_int(int))
    }

    /// Exponentiation by a `u64` power.
    ///
    /// The default implementation calls `exp_u64_generic`, which by default performs exponentiation
    /// by squaring. Rather than override this method, it is generally recommended to have the
    /// concrete field type override `exp_u64_generic`, so that any optimizations will apply to all
    /// abstract fields.
    #[must_use]
    #[inline]
    fn exp_u64(&self, power: u64) -> Self {
        Self::Char::exp_u64_generic(self.clone(), power)
    }

    /// Exponentiation by a constant power.
    ///
    /// For a collection of small values we implement custom multiplication chain circuits which can be faster than the
    /// simpler square and multiply approach.
    #[must_use]
    #[inline(always)]
    fn exp_const_u64<const POWER: u64>(&self) -> Self {
        match POWER {
            0 => Self::ONE,
            1 => self.clone(),
            2 => self.square(),
            3 => self.cube(),
            4 => self.square().square(),
            5 => self.square().square() * self.clone(),
            6 => self.square().cube(),
            7 => {
                let x2 = self.square();
                let x3 = x2.clone() * self.clone();
                let x4 = x2.square();
                x3 * x4
            }
            _ => self.exp_u64(POWER),
        }
    }

    /// The elementary function `halve(a) = a/2`.
    ///
    /// Will error if the field characteristic is 2.
    #[must_use]
    fn halve(&self) -> Self {
        // This should be overwritten by most field implementations.
        self.clone() * Self::from_char(Self::Char::TWO.inverse())
    }

    /// Multiply by a given power of two. `mul_2exp_u64(a, exp) = 2^exp * a`
    #[must_use]
    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        // This should be overwritten by most field implementations.
        self.clone() * Self::from_char(Self::Char::TWO.exp_u64(exp))
    }

    /// Divide by a given power of two. `div_2exp_u64(a, exp) = a/2^exp`
    #[must_use]
    #[inline]
    fn div_2exp_u64(&self, exp: u64) -> Self {
        // This should be overwritten by most field implementations.
        self.clone() * Self::from_char(Self::Char::TWO.inverse().exp_u64(exp))
    }
}

pub trait PermutationMonomial<const N: usize> {
    // TODO!!
}

pub trait PermutaitonMonomialInverse {
    // TODO!!
}

/// A commutative algebra over a finite field.
///
/// This permits elements like:
/// - an actual field element
/// - a symbolic expression which would evaluate to a field element
/// - an array of field elements
///
/// Mathematically speaking, this is an algebraic structure with addition,
/// multiplication and scalar multiplication. The addition and multiplication
/// maps must be both commutative and associative, and there must
/// exist identity elements for both (named `ZERO` and `ONE`
/// respectively). Furthermore, multiplication must distribute over
/// addition. Finally, the scalar multiplication must be realized by
/// a ring homomorphism from the field to the algebra.
pub trait FieldAlgebra: PrimeCharacteristicRing {
    type F: Field;

    /// Interpret a field element as a commutative algebra element.
    ///
    /// Mathematically speaking, this map is a ring homomorphism from the base field
    /// to the commutative algebra. The existence of this map makes this structure
    /// an algebra and not simply a commutative ring.
    fn from_f(f: Self::F) -> Self;

    /// Construct an iterator which returns powers of `self: self^0, self^1, self^2, ...`.
    #[must_use]
    fn powers(&self) -> Powers<Self> {
        self.shifted_powers(Self::ONE)
    }

    /// Construct an iterator which returns powers of `self` multiplied by `start: start, start*self^1, start*self^2, ...`.
    fn shifted_powers(&self, start: Self) -> Powers<Self> {
        Powers {
            base: self.clone(),
            current: start,
        }
    }

    fn powers_packed<P: PackedField<Scalar = Self>>(&self) -> PackedPowers<Self, P> {
        self.shifted_powers_packed(Self::ONE)
    }

    fn shifted_powers_packed<P: PackedField<Scalar = Self>>(
        &self,
        start: Self,
    ) -> PackedPowers<Self, P> {
        let mut current = P::from_f(start);
        let slice = current.as_slice_mut();
        for i in 1..P::WIDTH {
            slice[i] = slice[i - 1].clone() * self.clone();
        }

        PackedPowers {
            multiplier: P::from_f(self.clone()).exp_u64(P::WIDTH as u64),
            current,
        }
    }
}

/// An element of a finite field.
pub trait Field:
    FieldAlgebra<F = Self>
    + Packable
    + 'static
    + Copy
    + Div<Self, Output = Self>
    + Eq
    + Hash
    + Send
    + Sync
    + Display
    + Serialize
    + DeserializeOwned
{
    type Packing: PackedField<Scalar = Self>;

    /// A generator of this field's entire multiplicative group.
    const GENERATOR: Self;

    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    fn is_one(&self) -> bool {
        *self == Self::ONE
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

    fn order() -> BigUint;

    /// A list of (factor, exponent) pairs.
    fn multiplicative_group_factors() -> Vec<(BigUint, usize)> {
        let primality_test = MillerRabin { error_bits: 128 };
        let composite_splitter = PollardRho;
        let factorizer = FactorizerFromSplitter {
            primality_test,
            composite_splitter,
        };
        let n = Self::order() - BigUint::one();
        factorizer.factor_counts(&n)
    }

    #[inline]
    fn bits() -> usize {
        Self::order().bits() as usize
    }
}

/// Implementation of the quotient map `r → r mod p`.
pub trait QuotientMap<Int>: Sized {
    /// Convert a given integer into an element of the field `ℤ/p`.
    ///   
    /// This is the most generic method which makes no assumptions on the size of the input.
    /// Where possible, this method should be used with the smallest possible integer type.
    /// For example, if a 32-bit integer `x` is known to be less than `2^16`, then
    /// `from_int(x as u16)` will often be faster than `from_int(x)`. This is particularly true
    /// for boolean data.
    ///
    /// This method is also strongly preferred over `from_canonical_checked/from_canonical_unchecked`.
    /// It will usually be identical when `Int` is a small type, e.g. `bool/u8/u16` and is safer for
    /// larger types.
    fn from_int(int: Int) -> Self;

    // Q: We could also make from_canonical_checked/from_canonical_unchecked assume that the input lies in
    // 0 to p - 1. Would this be better? The downside of this is that this might lead to the methods
    // being a little slower if that doesn't align with the underlying representation. On the other hand
    // it would let us make a guarantee that the output won't suddenly become invalid.

    // A:   When dealing with unsigned types, from_canonical assumes that the input lies in [0, P).
    //      When dealing with signed types, from_canonical assumes that the input lies in [-(P - 1)/2, (P + 1)/2).
    //      TODO: Add this into assumptions.

    /// Convert a given integer into an element of the field `ℤ/p`. The input is guaranteed
    /// to lie within some specific range.
    ///
    /// The exact range depends on the specific field and is checked by assert statements at run time. Where possible
    /// it is safer to use `from_int` as, if the internal representation of the field changes, the allowed
    /// range will also change.
    fn from_canonical_checked(int: Int) -> Option<Self>;

    /// Convert a given integer into an element of the field `ℤ/p`. The input is guaranteed
    /// to lie within some specific range.
    ///
    /// # Safety
    ///
    /// The exact range depends on the specific field and is not checked. Using this function is not recommended.
    /// If the internal representation of the field changes, the expected range may also change which might lead
    /// to undefined behaviour. However this will be faster than `from_int/from_canonical_checked` in some
    /// circumstances and so we provide it here for careful use in performance critical applications.
    unsafe fn from_canonical_unchecked(int: Int) -> Self;
}

/// A field isomorphic to `ℤ/p` for some prime `p`.
///
/// There is a natural map from `ℤ` to `ℤ/p` given by `r → r mod p`.
pub trait PrimeField:
    Field
    + Ord
    + QuotientMap<bool>
    + QuotientMap<u8>
    + QuotientMap<u16>
    + QuotientMap<u32>
    + QuotientMap<u64>
    + QuotientMap<u128>
    + QuotientMap<usize>
    + QuotientMap<i8>
    + QuotientMap<i16>
    + QuotientMap<i32>
    + QuotientMap<i64>
    + QuotientMap<i128>
    + QuotientMap<isize>
{
    /// The field element 2 mod p.
    ///
    /// This is provided as a convenience as `TWO` occurs regularly in
    /// the proving system. This also is slightly faster than computing
    /// it via addition. Note that multiplication by `TWO` is discouraged.
    /// Instead of `a * TWO` use `a.double()` which will be faster.
    ///
    /// When p = 2, this is equal to ZERO.
    const TWO: Self;

    /// The field element (-1) mod p.
    ///
    /// This is provided as a convenience as `NEG_ONE` occurs regularly in
    /// the proving system. This also is slightly faster than computing
    /// it via negation. Note that where possible `NEG_ONE` should be absorbed
    /// into mathematical operations. For example `a - b` will be faster
    /// than `a + NEG_ONE * b` and similarly `(-b)` is faster than `NEG_ONE * b`.
    ///
    /// When p = 2, this is equal to ONE.
    const NEG_ONE: Self;

    fn as_canonical_biguint(&self) -> BigUint;

    /// Exponentiation by a `u64` power. This is similar to `exp_u64`, but more general in that it
    /// can be used with `CommutativeRing` with prime characteristic, not just this concrete field.
    ///
    /// The default implementation uses naive square and multiply. Implementations may want to
    /// override this and handle certain powers with more optimal addition chains.
    #[must_use]
    #[inline]
    fn exp_u64_generic<PCR: PrimeCharacteristicRing<Char = Self>>(val: PCR, power: u64) -> PCR {
        exp_u64_by_squaring(val, power)
    }
}

/// A prime field `ℤ/p` with order `p < 2^64`.
pub trait PrimeField64: PrimeField {
    const ORDER_U64: u64;

    /// Return the representative of `value` which lies in the range `0 <= x < ORDER_U64`.
    fn as_canonical_u64(&self) -> u64;
}

/// A prime field `ℤ/p` with order `p < 2^32`.
pub trait PrimeField32: PrimeField64 {
    const ORDER_U32: u32;

    /// Return the representative of `value` which lies in the range `0 <= x < ORDER_U32`.
    fn as_canonical_u32(&self) -> u32;
}

/// A commutative algebra over an extension field.
///
/// Mathematically, this trait captures a slightly more interesting structure than the above one liner.
/// As implemented here, A FieldExtensionAlgebra `FEA` over and extension field `EF` is
/// really the result of applying extension of scalars to a FieldAlgebra `FA` to lift `FA`
/// from an algebra over `F` to an algebra over `EF` and so `FEA = EF ⊗ FA` where the tensor
/// product is over `F`.
pub trait FieldExtensionAlgebra<Base: FieldAlgebra>:
    FieldAlgebra
    + From<Base>
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
    /// length at exactly D, and constructs the field element
    /// \sum_i bs[i] * X^i.
    ///
    /// NB: The value produced by this function fundamentally depends
    /// on the choice of irreducible polynomial f. Care must be taken
    /// to ensure portability if these values might ever be passed to
    /// (or rederived within) another compilation environment where a
    /// different f might have been used.
    fn from_base_slice(bs: &[Base]) -> Self;

    /// Similar to `core:array::from_fn`, with the same caveats as
    /// `from_base_slice`.
    fn from_base_fn<F: FnMut(usize) -> Base>(f: F) -> Self;
    fn from_base_iter<I: Iterator<Item = Base>>(iter: I) -> Self;

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

    /// Suppose this field extension is represented by the quotient
    /// ring B[X]/(f(X)) where B is `Base` and f is an irreducible
    /// polynomial of degree `D`. This function returns the field
    /// element `X^exponent` if `exponent < D` and panics otherwise.
    /// (The fact that f is not known at the point that this function
    /// is defined prevents implementing exponentiation of higher
    /// powers since the reduction cannot be performed.)
    ///
    /// NB: The value produced by this function fundamentally depends
    /// on the choice of irreducible polynomial f. Care must be taken
    /// to ensure portability if these values might ever be passed to
    /// (or rederived within) another compilation environment where a
    /// different f might have been used.
    fn monomial(exponent: usize) -> Self {
        assert!(exponent < Self::D, "requested monomial of too high degree");
        let mut vec = vec![Base::ZERO; Self::D];
        vec[exponent] = Base::ONE;
        Self::from_base_slice(&vec)
    }
}

pub trait ExtensionField<Base: Field>: Field + FieldExtensionAlgebra<Base> {
    type ExtensionPacking: FieldExtensionAlgebra<Base::Packing, F = Self>
        + 'static
        + Copy
        + Send
        + Sync;

    #[inline(always)]
    fn is_in_basefield(&self) -> bool {
        self.as_base_slice()[1..].iter().all(Field::is_zero)
    }

    fn as_base(&self) -> Option<Base> {
        if self.is_in_basefield() {
            Some(self.as_base_slice()[0])
        } else {
            None
        }
    }

    fn ext_powers_packed(&self) -> impl Iterator<Item = Self::ExtensionPacking> {
        let powers = self.powers().take(Base::Packing::WIDTH + 1).collect_vec();
        // Transpose first WIDTH powers
        let current = Self::ExtensionPacking::from_base_fn(|i| {
            Base::Packing::from_fn(|j| powers[j].as_base_slice()[i])
        });
        // Broadcast self^WIDTH
        let multiplier = Self::ExtensionPacking::from_base_fn(|i| {
            Base::Packing::from(powers[Base::Packing::WIDTH].as_base_slice()[i])
        });

        core::iter::successors(Some(current), move |&current| Some(current * multiplier))
    }
}

impl<F: Field> ExtensionField<F> for F {
    type ExtensionPacking = F::Packing;
}

impl<FA: FieldAlgebra> FieldExtensionAlgebra<FA> for FA {
    const D: usize = 1;

    fn from_base(b: FA) -> Self {
        b
    }

    fn from_base_slice(bs: &[FA]) -> Self {
        assert_eq!(bs.len(), 1);
        bs[0].clone()
    }

    fn from_base_iter<I: Iterator<Item = FA>>(mut iter: I) -> Self {
        iter.next().unwrap()
    }

    fn from_base_fn<F: FnMut(usize) -> FA>(mut f: F) -> Self {
        f(0)
    }

    #[inline(always)]
    fn as_base_slice(&self) -> &[FA] {
        slice::from_ref(self)
    }
}

/// A field which supplies information like the two-adicity of its multiplicative group, and methods
/// for obtaining two-adic generators.
pub trait TwoAdicField: Field {
    /// The number of factors of two in this field's multiplicative group.
    const TWO_ADICITY: usize;

    /// Returns a generator of the multiplicative group of order `2^bits`.
    /// Assumes `bits <= TWO_ADICITY`, otherwise the result is undefined.
    #[must_use]
    fn two_adic_generator(bits: usize) -> Self;
}

/// An iterator over the powers of a certain base element `b`: `b^0, b^1, b^2, ...`.
#[derive(Clone, Debug)]
pub struct Powers<F> {
    pub base: F,
    pub current: F,
}

impl<FA: FieldAlgebra> Iterator for Powers<FA> {
    type Item = FA;

    fn next(&mut self) -> Option<FA> {
        let result = self.current.clone();
        self.current *= self.base.clone();
        Some(result)
    }
}

/// like `Powers`, but packed into `PackedField` elements
#[derive(Clone, Debug)]
pub struct PackedPowers<F, P: PackedField<Scalar = F>> {
    // base ** P::WIDTH
    pub multiplier: P,
    pub current: P,
}

impl<FA: FieldAlgebra, P: PackedField<Scalar = FA>> Iterator for PackedPowers<FA, P> {
    type Item = P;

    fn next(&mut self) -> Option<P> {
        let result = self.current;
        self.current *= self.multiplier;
        Some(result)
    }
}
