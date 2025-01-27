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

/// A commutative ring `(R)` with prime characteristic `(p)`.
///
/// This permits elements like:
/// - A single finite field element
/// - a symbolic expression which would evaluate to a field element
/// - an array of finite field elements
///
/// ### Mathematical Description
///
/// Mathematically, a commutative ring is a set of objects which supports an addition-like
/// like operation, `+`, and a multiplication-like operation `*`.
///  Let `x, y, z` denote arbitrary elements of the set.
///
/// Then, an operation is addition-like if it satisfies the following properties:
/// - Commutativity => `x + y = y + x`
/// - Associativity => `x + (y + z) = (x + y) + z`
/// - Unit => There exists an identity element `ZERO` satisfying `x + ZERO = x`.
/// - Inverses => For every `x` there exists a unique inverse `(-x)` satisfying `x + (-x) = ZERO`
///
/// Similarly, an operation is multiplication-like if it satisfies the following properties:
/// - Commutativity => `x * y = y * x`
/// - Associativity => `x * (y * z) = (x * y) * z`
/// - Unit => There exists an identity element `ONE` satisfying `x * ONE = x`.
/// - Distributivity => The two operations `+` and `*` must together satisfy `x * (y + z) = (x * y) + (x * z)`
///
/// Unlike in the addition case, we do not require inverses to exist with respect to `*`.
///
/// The simplest examples of commutative rings are the integers (`ℤ`), and the integers mod `N` (`ℤ/N`).
///
/// The characteristic of a ring is the smallest positive integer `r` such that `0 = r . 1 = 1 + 1 + ... + 1 (r times)`.
/// For example, the characteristic of the modulo ring `ℤ/N` is `N`.
///
/// Rings with prime characteristic are particularly special due to their close relationship with finite fields.
pub trait PrimeCharacteristicRing:
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
    /// The field `ℤ/p` where the characteristic of this ring is p.
    type PrimeSubfield: PrimeField;

    /// The additive identity of the ring.
    ///
    /// For every element `a` in the ring we require the following properties:
    ///
    /// `a + ZERO = ZERO + a = a,`
    ///
    /// `a + (-a) = (-a) + a = ZERO.`
    const ZERO: Self;

    /// The multiplicative identity of the ring.
    ///
    /// For every element `a` in the ring we require the following property:
    ///
    /// `a*ONE = ONE*a = a.`
    const ONE: Self;

    /// The element in the ring given by `ONE + ONE`.
    ///
    /// This is provided as a convenience as `TWO` occurs regularly in
    /// the proving system. This also is slightly faster than computing
    /// it via addition. Note that multiplication by `TWO` is discouraged.
    /// Instead of `a * TWO` use `a.double()` which will be faster.
    ///
    /// If the field has characteristic 2 this is equal to ZERO.
    const TWO: Self;

    /// The element in the ring given by `-ONE`.
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
    /// This function should be preferred over calling `a + a` or `TWO * a` as a faster implementation may be available for some rings.
    /// If the field has characteristic 2 then this returns 0.
    #[must_use]
    fn double(&self) -> Self {
        self.clone() + self.clone()
    }

    /// The elementary function `square(a) = a^2`.
    ///
    /// This function should be preferred over calling `a * a`, as a faster implementation may be available for some rings.
    #[must_use]
    fn square(&self) -> Self {
        self.clone() * self.clone()
    }

    /// The elementary function `cube(a) = a^3`.
    ///
    /// This function should be preferred over calling `a * a * a`, as a faster implementation may be available for some rings.
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

/// A vector space over `F` which can be serialized into and out of a collection of `F` elements.
///
/// For the most part, we will usually expect `F` to be a field but there
/// are a few cases where it is handy to allow it to just be a ring. In
/// particular, every ring implements `Serializable<Self>`.
///
/// We make no guarantees about consistency of this Serialization/Deserialization
/// across different versions of Plonky3.
///
/// ### Mathematical Description
///
/// Mathematically a more accurate name for this trait would be `BasedVectorSpace` or
/// even more generally `BasedFreeModule` if you want to account for cases where `F` is
/// not a field.
///
/// Given a vector space, `A` over `F`, we can pick a basis of elements `B = {b_0, ..., b_{n-1}}`
/// in `A` such that, given any element `a`, we can find a unique set of `n` elements of `F`,
/// `f_0, ..., f_{n - 1}` satisfying `a = f_0 b_0 + ... + f_{n - 1} b_{n - 1}`.
///
/// Thus choosing this basis `B` allows us to map between elements of `A` and
/// arrays of `n` elements of `F`. Clearly this map depends entirely on the
/// choice of basis `B` which may change across versions of Plonky3.
pub trait Serializable<F: PrimeCharacteristicRing>: Sized {
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
    ///
    /// The user should ensure that the slice has length `DIMENSION`. If
    /// it is shorter than this, the function will panic, if it is longer the
    /// extra elements will be ignored.
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
    ///
    /// If the iterator contains more than `DIMENSION` many elements,
    /// the rest will be ignored.
    fn deserialize_iter<I: Iterator<Item = F>>(iter: I) -> Self;

    /// Given a basis for the Algebra `A`, return the i'th basis element.
    ///
    /// # Safety
    ///
    /// The value produced by this function fundamentally depends
    /// on the choice of basis. Care must be taken
    /// to ensure portability if these values might ever be passed to
    /// (or rederived within) another compilation environment where a
    /// different basis might have been used.
    fn ith_basis_element(i: usize) -> Self {
        Self::deserialize_fn(|j| F::from_bool(i == j))
    }
}

impl<F: PrimeCharacteristicRing> Serializable<F> for F {
    const DIMENSION: usize = 1;

    #[inline]
    fn serialize_as_slice(&self) -> &[F] {
        slice::from_ref(self)
    }

    #[inline]
    fn deserialize_fn<Fn: FnMut(usize) -> F>(mut f: Fn) -> Self {
        f(0)
    }

    #[inline]
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
pub trait InjectiveMonomial<const N: u64>: PrimeCharacteristicRing {
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

/// A ring `R` implements `Algebra<F>` if there is an injective homomorphism
///  from `F` into `R`; in particular only `F::ZERO` maps to `R::ZERO`.
///
/// For the most part, we will usually expect `F` to be a field but there
/// are a few cases where it is handy to allow it to just be a ring. In
/// particular, every ring naturally implements `Algebra<Self>`.
///
/// ### Mathematical Description
///
/// Let `x` and `y` denote arbitrary elements of `F`. Then
/// we require that our map `from` has the properties:
/// - Preserves Identity: `from(F::ONE) = R::ONE`
/// - Commutes with Addition: `from(x + y) = from(x) + from(y)`
/// - Commutes with Multiplication: `from(x * y) = from(x) * from(y)`
///
/// Such maps are known as ring homomorphisms and are injective if the
/// only element which maps to `R::ZERO` is `F::ZERO`.
///
/// The existence of this map makes `R` into an `F`-module and hence an `F`-algebra.
/// If, additionally, `R` is a field, then this makes `R` a field extension of `F`.
pub trait Algebra<F>:
    PrimeCharacteristicRing
    + From<F>
    + Add<F, Output = Self>
    + AddAssign<F>
    + Sub<F, Output = Self>
    + SubAssign<F>
    + Mul<F, Output = Self>
    + MulAssign<F>
{
}

// Every ring is an algebra over itself.
impl<R: PrimeCharacteristicRing> Algebra<R> for R {}

/// An element of a finite field.
pub trait Field:
    Algebra<Self>
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

/// A field isomorphic to `ℤ/p` for some prime `p`.
///
/// There is a natural map from `ℤ` to `ℤ/p` which sends an integer `r` to its conjugacy class `[r]`.
/// Canonically, each conjugacy class `[r]` can be represented by the unique integer `s` in `[0, p - 1)`
/// satisfying `s = r mod p`. This however is often not the most convenient computational representation
/// and so internal representations of field elements might differ from this and may change over time.
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

/// A prime field `ℤ/p` with order, `p < 2^64`.
pub trait PrimeField64: PrimeField {
    const ORDER_U64: u64;

    /// Return the representative of `value` in canonical form
    /// which lies in the range `0 <= x < ORDER_U64`.
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

/// A prime field `ℤ/p` with order `p < 2^32`.
pub trait PrimeField32: PrimeField64 {
    const ORDER_U32: u32;

    /// Return the representative of `value` in canonical form
    /// which lies in the range `0 <= x < ORDER_U64`.
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
/// really the result of applying extension of scalars to a algebra `FA` to lift `FA`
/// from an algebra over `F` to an algebra over `EF` and so `FEA = EF ⊗ FA` where the tensor
/// product is over `F`.
///
/// This will be deleted in a future PR. It's currently only needed to give some traits for
/// ExtensionPacking and so we will soon replace it by a new packed extension field trait.
pub trait FieldExtensionAlgebra<Base: PrimeCharacteristicRing>:
    Algebra<Base> + Serializable<Base>
{
    const D: usize;
}

pub trait ExtensionField<Base: Field>: Field + FieldExtensionAlgebra<Base> {
    type ExtensionPacking: FieldExtensionAlgebra<Base::Packing>
        + Algebra<Self>
        + 'static
        + Copy
        + Send
        + Sync;

    fn is_in_basefield(&self) -> bool;

    fn as_base(&self) -> Option<Base>;

    /// Construct an iterator which returns powers of `self` packed into `ExtensionPacking` elements.
    ///
    /// E.g. if `PACKING::WIDTH = 4` this returns the elements:
    /// `[self^0, self^1, self^2, self^3], [self^4, self^5, self^6, self^7], ...`.
    ///
    /// Once we create the new packed extension field trait, this function will be moved there.
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

    fn ext_powers_packed(&self) -> Powers<Self::ExtensionPacking> {
        self.powers_packed()
    }
}

impl<R: PrimeCharacteristicRing> FieldExtensionAlgebra<R> for R {
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

impl<R: PrimeCharacteristicRing> Iterator for Powers<R> {
    type Item = R;

    fn next(&mut self) -> Option<R> {
        let result = self.current.clone();
        self.current *= self.base.clone();
        Some(result)
    }
}
