use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{Debug, Display};
use core::hash::Hash;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use core::slice;

use num_bigint::BigUint;
use num_traits::One;
use nums::{Factorizer, FactorizerFromSplitter, MillerRabin, PollardRho};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::exponentiation::bits_u64;
use crate::integers::{from_integer_types, QuotientMap};
use crate::packed::PackedField;
use crate::Packable;

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
pub trait FieldAlgebra:
    Sized
    + Default
    + Clone
    + From<Self::F>
    + Add<Self::F, Output = Self>
    + AddAssign<Self::F>
    + Sub<Self::F, Output = Self>
    + SubAssign<Self::F>
    + Mul<Self::F, Output = Self>
    + MulAssign<Self::F>
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
    type F: Field;

    /// The field `ℤ/p` where the characteristic of this ring is p.
    type PrimeSubfield: PrimeField;

    /// The additive identity of the algebra.
    ///
    /// For every element `a` in the algebra we require the following properties:
    ///
    /// `a + ZERO = ZERO + a = a,`
    ///
    /// `a + (-a) = (-a) + a = ZERO.`
    const ZERO: Self;

    /// The multiplicative identity of the Algebra
    ///
    /// For every element `a` in the algebra we require the following property:
    ///
    /// `a*ONE = ONE*a = a.`
    const ONE: Self;

    /// The element in the algebra given by `ONE + ONE`.
    ///
    /// This is provided as a convenience as `TWO` occurs regularly in
    /// the proving system. This also is slightly faster than computing
    /// it via addition. Note that multiplication by `TWO` is discouraged.
    /// Instead of `a * TWO` use `a.double()` which will be faster.
    ///
    /// If the field has characteristic 2 this is equal to ZERO.
    const TWO: Self;

    /// The element in the algebra given by `-ONE`.
    ///
    /// This is provided as a convenience as `NEG_ONE` occurs regularly in
    /// the proving system. This also is slightly faster than computing
    /// it via negation. Note that where possible `NEG_ONE` should be absorbed
    /// into mathematical operations. For example `a - b` will be faster
    /// than `a + NEG_ONE * b` and similarly `(-b)` is faster than `NEG_ONE * b`.
    ///
    /// If the field has characteristic 2 this is equal to ONE.
    const NEG_ONE: Self;

    /// Embed an element of the prime field `ℤ/p` into the ring `R`.
    ///
    /// Given any element `r ∈ ℤ/p`, represented as an integer between `0` and `p - 1`
    /// `from_prime_subfield(r)` will be equal to:
    ///
    /// `Self::ONE + ... + Self::ONE (r times)`
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self;

    /// Return `Self::ONE` if `b` is `true` and `Self::ZERO` if `b` is `false`.
    fn from_bool(b: bool) -> Self {
        // Some rings might reimplement this to avoid the branch.
        if b {
            Self::ONE
        } else {
            Self::ZERO
        }
    }

    from_integer_types!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

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

    /// Exponentiation by a `u64` power.
    #[must_use]
    #[inline]
    fn exp_u64(&self, power: u64) -> Self {
        let mut current = self.clone();
        let mut product = Self::ONE;

        for j in 0..bits_u64(power) {
            if (power >> j & 1) != 0 {
                product *= current.clone();
            }
            current = current.square();
        }
        product
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

    /// Compute self^{2^power_log} by repeated squaring.
    #[must_use]
    fn exp_power_of_2(&self, power_log: usize) -> Self {
        let mut res = self.clone();
        for _ in 0..power_log {
            res = res.square();
        }
        res
    }

    /// self * 2^exp
    #[must_use]
    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        self.clone() * Self::TWO.exp_u64(exp)
    }

    /// Construct an iterator which returns powers of `self: self^0, self^1, self^2, ...`.
    #[must_use]
    fn powers(&self) -> Powers<Self> {
        self.shifted_powers(Self::ONE)
    }

    /// Construct an iterator which returns powers of `self` shifted by `start: start, start*self^1, start*self^2, ...`.
    fn shifted_powers(&self, start: Self) -> Powers<Self> {
        Powers {
            base: self.clone(),
            current: start,
        }
    }

    /// Construct an iterator which returns powers of `self` packed into `PackedField` elements.
    ///
    /// E.g. if `PACKING::WIDTH = 4` this returns the elements:
    /// `[self^0, self^1, self^2, self^3], [self^4, self^5, self^6, self^7], ...`.
    fn powers_packed<P: PackedField<Scalar = Self>>(&self) -> Powers<P> {
        self.shifted_powers_packed(Self::ONE)
    }

    /// Construct an iterator which returns powers of `self` shifted by start
    /// and packed into `PackedField` elements.
    ///
    /// E.g. if `PACKING::WIDTH = 4` this returns the elements:
    /// `[start, start*self, start*self^2, start*self^3], [start*self^4, start*self^5, start*self^6, start*self^7], ...`.
    fn shifted_powers_packed<P: PackedField<Scalar = Self>>(&self, start: Self) -> Powers<P> {
        let mut current: P = start.into();
        let slice = current.as_slice_mut();
        for i in 1..P::WIDTH {
            slice[i] = slice[i - 1].clone() * self.clone();
        }

        Powers {
            base: self.clone().exp_u64(P::WIDTH as u64).into(),
            current,
        }
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

/// A field algebra which can be serialized into and out of a
/// collection of field elements.
///
/// We make no guarantees about consistency of this Serialization/Deserialization
/// across different versions of Plonky3.
///
/// ### Mathematical Description
///
/// Mathematically a more accurate name for this trait would be BasedFreeVectorSpace.
///
/// As `F` is a field, every field algebra `A`, over `F` is an `F`-vector space.
/// This means we can pick a basis of elements `B = {b_0, ..., b_{n-1}}` in `A`
/// such that, given any element `a`, we can find a unique set of `n` elements of `F`,
/// `f_0, ..., f_{n - 1}` satisfying `a = f_0 b_0 + ... + f_{n - 1} b_{n - 1}`.
///
/// Thus choosing this basis `B` allows us to map between elements of `A` and
/// arrays of `n` elements of `F`. Clearly this map depends entirely on the
/// choice of basis `B` which may change across versions of Plonky3.
pub trait Serializable<F: FieldAlgebra>: Sized {
    // We could alternatively call this BasedAlgebra?
    // The name is currently trying to indicate what this is meant to be
    // used for as opposed to being mathematically accurate.

    const DIMENSION: usize;

    /// Fixes a basis for the algebra `A` and uses this to
    /// map an element of `A` to a vector of `n` `F` elements.
    ///
    /// # Safety
    ///
    /// The value produced by this function fundamentally depends
    /// on the choice of basis. Care must be taken
    /// to ensure portability if these values might ever be passed to
    /// (or rederived within) another compilation environment where a
    /// different basis might have been used.
    fn serialize_as_slice(&self) -> &[F];

    /// Fixes a basis for the algebra `A` and uses this to
    /// map `n` `F` elements to an element of `A`.
    ///
    /// # Safety
    ///
    /// The value produced by this function fundamentally depends
    /// on the choice of basis. Care must be taken
    /// to ensure portability if these values might ever be passed to
    /// (or rederived within) another compilation environment where a
    /// different basis might have been used.
    #[inline]
    fn deserialize_slice(slice: &[F]) -> Self {
        Self::deserialize_fn(|i| slice[i].clone())
    }

    /// Fixes a basis for the algebra `A` and uses this to
    /// map `n` `F` elements to an element of `A`. Similar
    /// to `core:array::from_fn`, the `n` `F` elements are
    /// given by `Fn(0), ..., Fn(n - 1)`.
    ///
    /// # Safety
    ///
    /// The value produced by this function fundamentally depends
    /// on the choice of basis. Care must be taken
    /// to ensure portability if these values might ever be passed to
    /// (or rederived within) another compilation environment where a
    /// different basis might have been used.
    fn deserialize_fn<Fn: FnMut(usize) -> F>(f: Fn) -> Self;

    fn ith_basis_element(i: usize) -> Self {
        Self::deserialize_fn(|j| {
            if i == j {
                F::ONE.clone()
            } else {
                F::ZERO.clone()
            }
        })
    }

    /// Fixes a basis for the algebra `A` and uses this to
    /// map `n` `F` elements to an element of `A`.
    ///
    /// # Safety
    ///
    /// The value produced by this function fundamentally depends
    /// on the choice of basis. Care must be taken
    /// to ensure portability if these values might ever be passed to
    /// (or rederived within) another compilation environment where a
    /// different basis might have been used.
    fn deserialize_iter<I: Iterator<Item = F>>(iter: I) -> Self;
}

impl<F: FieldAlgebra> Serializable<F> for F {
    const DIMENSION: usize = 1;

    fn serialize_as_slice(&self) -> &[F] {
        slice::from_ref(self)
    }

    fn deserialize_slice(slice: &[F]) -> Self {
        slice[0].clone()
    }

    fn deserialize_fn<Fn: FnMut(usize) -> F>(mut f: Fn) -> Self {
        f(0)
    }

    fn deserialize_iter<I: Iterator<Item = F>>(mut iter: I) -> Self {
        iter.next().unwrap()
    }
}

/// A ring implements `InjectiveMonomial<N>` if the algebraic function
/// `f(x) = x^N` is an injective map on elements of the ring.
///
/// We do not enforce that this map be invertible as there are useful
/// cases such as polynomials or symbolic expressions where no inverse exists.
///
/// However, if the ring is a field with order `q` or an array of such field elements,
/// then `f(x) = x^N` will be injective if and only if it is invertible and so in
/// such cases this monomial acts as a permutation. Moreover, this will occur
/// exactly when `N` and `q - 1` are relatively prime i.e. `gcd(N, q - 1) = 1`.
pub trait InjectiveMonomial<const N: u64>: FieldAlgebra {
    /// Compute `x -> x^n` for a given `n > 1` such that this
    /// map is injective.
    fn injective_exp_n(&self) -> Self {
        self.exp_const_u64::<N>()
    }
}

/// A ring implements PermutationMonomial<N> if the algebraic function
/// `f(x) = x^N` is invertible and thus acts as a permutation on elements of the ring.
///
/// In all cases we care about, this means that we can find another integer `K` such
/// that `x = x^{NK}` for all elements of our ring.
pub trait PermutationMonomial<const N: u64>: InjectiveMonomial<N> {
    /// Compute `x -> x^K` for a given `K > 1` such that
    /// `x^{NK} = x` for all elements `x`.
    fn injective_exp_root_n(&self) -> Self;
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

    /// Computes input/2.
    /// Should be overwritten by most field implementations to use bitshifts.
    /// Will error if the field characteristic is 2.
    #[must_use]
    fn halve(&self) -> Self {
        let half = Self::TWO
            .try_inverse()
            .expect("Cannot divide by 2 in fields with characteristic 2");
        *self * half
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

pub trait PrimeField:
    Field
    + Ord
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
    fn as_canonical_biguint(&self) -> BigUint;
}

/// A prime field of order less than `2^64`.
pub trait PrimeField64: PrimeField {
    const ORDER_U64: u64;

    /// Return the representative of `value` that is less than `ORDER_U64`.
    fn as_canonical_u64(&self) -> u64;

    /// Convert a field element to a `u64` such that any two field elements
    /// are converted to the same `u64` if and only if they represent the same value.
    ///
    /// This will be the fastest way to convert a field element to a `u64` and
    /// is intended for use in hashing. It will also be consistent across different targets.
    fn to_unique_u64(&self) -> u64 {
        // A simple default which is optimal for some fields.
        self.as_canonical_u64()
    }
}

/// A prime field of order less than `2^32`.
pub trait PrimeField32: PrimeField64 {
    const ORDER_U32: u32;

    /// Return the representative of `value` that is less than `ORDER_U32`.
    fn as_canonical_u32(&self) -> u32;

    /// Convert a field element to a `u32` such that any two field elements
    /// are converted to the same `u32` if and only if they represent the same value.
    ///
    /// This will be the fastest way to convert a field element to a `u32` and
    /// is intended for use in hashing. It will also be consistent across different targets.
    fn to_unique_u32(&self) -> u32 {
        // A simple default which is optimal for some fields.
        self.as_canonical_u32()
    }
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
    + Serializable<Base>
{
    const D: usize;
}

pub trait ExtensionField<Base: Field>: Field + FieldExtensionAlgebra<Base> {
    type ExtensionPacking: FieldExtensionAlgebra<Base::Packing, F = Self>
        + 'static
        + Copy
        + Send
        + Sync;

    fn is_in_basefield(&self) -> bool;

    fn as_base(&self) -> Option<Base>;

    fn from_base(val: Base) -> Self;

    /// Construct an iterator which returns powers of `self` packed into `ExtensionPacking` elements.
    ///
    /// E.g. if `PACKING::WIDTH = 4` this returns the elements:
    /// `[self^0, self^1, self^2, self^3], [self^4, self^5, self^6, self^7], ...`.
    fn ext_powers_packed(&self) -> Powers<Self::ExtensionPacking>;
}

impl<F: Field> ExtensionField<F> for F {
    type ExtensionPacking = F::Packing;

    fn is_in_basefield(&self) -> bool {
        true
    }

    fn as_base(&self) -> Option<F> {
        Some(*self)
    }

    fn from_base(val: F) -> Self {
        val
    }

    fn ext_powers_packed(&self) -> Powers<Self::ExtensionPacking> {
        self.powers_packed()
    }
}

impl<FA: FieldAlgebra> FieldExtensionAlgebra<FA> for FA {
    const D: usize = 1;
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

/// An iterator which returns the powers of a base element `b` shifted by current `c`: `c, c * b, c * b^2, ...`.
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
