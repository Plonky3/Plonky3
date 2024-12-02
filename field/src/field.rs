use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{Debug, Display};
use core::hash::Hash;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use num_bigint::BigUint;
use num_traits::One;
use nums::{Factorizer, FactorizerFromSplitter, MillerRabin, PollardRho};
use paste::paste;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::packed::PackedField;
use crate::{bits_u64, Packable, PackedFieldExtension};

/// This trait encompasses a very wide collection of disparate things which support some
/// operation with similar properties to addition. Some examples are
/// - A Field
/// - The Unit Circle in a complex extension
/// - An Elliptic Curve
///
/// ### Mathematical Description
///
/// Mathematically an abelian group is an algebraic structure which supports an addition-like
/// like operation `+`. Let `x, y, z` denote arbitrary elements of the struct. Then, an
/// operation is addition-like if it satisfies the following properties:
/// - Commutativity => `x + y = y + x`
/// - Associativity => `x + (y + z) = (x + y) + z`
/// - Unit => There exists an identity element `ZERO` satisfying `x + ZERO = x`.
/// - Inverses => For every `x` there exists a unique inverse `(-x)` satisfying `x + (-x) = ZERO`
pub trait AbelianGroup:
    Sized
    + Default
    + Clone
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Neg<Output = Self>
    + Sum
{
    /// The additive identity element.
    ///
    /// For every other element `x` we require the following properties:
    ///
    /// `x + ZERO = ZERO + x = x,`
    ///
    /// `x + (-x) = (-x) + x = ZERO.`
    const ZERO: Self;

    /// The elementary function `double(a) = a + a`.
    ///
    /// This function should be preferred over calling `a + a` or `TWO * a`
    /// as a faster implementation may be available.
    #[must_use]
    fn double(&self) -> Self {
        self.clone() + self.clone()
    }

    /// The function which adds `x` to itself `r` times.
    ///
    /// E.g. `r * x = x + x + ... + x (r times)`
    ///
    /// The slight abuse of notation `*` is justified by this being
    /// precisely the `ℤ`-module structure that exists for all Abelian groups.
    #[must_use]
    fn mul_u64(&self, r: u64) -> Self {
        let mut current = self.clone();
        let mut res = Self::ZERO;

        for j in 0..bits_u64(r) {
            if (r >> j & 1) != 0 {
                res += current.clone();
            }
            current = current.double();
        }
        res
    }

    /// Double `x` an `exp` number of times to compute `2^{exp} * x`.
    ///
    /// For exp < 64, this will be the same as `x.mul_u64(1 << exp)`
    /// but may be a little faster to compute.
    ///
    /// This will be slow for large inputs and should be avoided for
    /// exp bigger than 32 or so.
    #[must_use]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        // TODO: Should there be a default implementation here?
        // This is all we can do here, by assumption, we don't have multiplication.
        // So fields will need to reimplement this. But tbh this should never be
        // called with exp large anyway so it might not be an issue?

        // This should be reimplemented by most rings as faster methods are available.
        let mut current = self.clone();

        for _ in 0..exp {
            current = current.double();
        }
        current
    }
}

/// An abelian group which additionally supports multiplication. Examples of
/// structs which should implement this trait are structs containing
/// - A single finite field element.
/// - A symbolic expression which may be evaluated to a finite field element.
/// - an array of finite field elements.
///
/// In practice every struct which implements this is expected to also implement PrimeCharacteristicRing.
///
/// ### Mathematical Description
///
/// Mathematically a commutative ring is an Abelian group with a multiplication-like operation `*`.
/// Let `x, y, z` denote arbitrary elements of the struct. Then, an operation is multiplication-like
/// if it satisfies the following properties:
/// - Commutativity => `x * y = y * x`
/// - Associativity => `x * (y * z) = (x * y) * z`
/// - Unit => There exists an identity element `ONE` satisfying `x * ONE = x`.
/// - Distributivity => The two operations `+` and `*` must together satisfy `x * (y + z) = (x * y) + (x * z)`
///
/// Unlike in the Abelian group case, we do not require inverses to exist with respect to `*`.
///
/// The simplest examples of commutative rings are the integers (`ℤ`), and the integers mod `N` (`ℤ/N`).
pub trait CommutativeRing: AbelianGroup + Mul<Output = Self> + MulAssign + Product + Debug {
    /// The multiplicative identity of the ring
    ///
    /// For every element `x` in the ring we require the following property:
    ///
    /// `x*ONE = ONE*x = x.`
    const ONE: Self;

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

    /// Exponentiation by a `u64` power using the square and multiply algorithm.
    ///
    /// This uses log(power) squares and log(power) multiplications.
    ///
    /// For a specific power, this can usually be improved upon by using
    /// an optimized addition chain but these need to be computed on a
    /// case by case basis.
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

    /// Compute the dot product of two vectors.
    fn dot_product<const N: usize>(u: &[Self; N], v: &[Self; N]) -> Self {
        u.iter().zip(v).map(|(x, y)| x.clone() * y.clone()).sum()
    }

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

/// This is a simple macro which lets us cleanly define the function `from_Int`
/// with `Int` can be replaced by any integer type.
///
/// Running, `from_integer_types(Int)` adds the following code to a trait:
///
/// ```rust,ignore
/// /// Given an integer `r`, return the sum of `r` copies of `ONE`:
/// ///
/// /// `r.Self::ONE =  Self::ONE + ... + Self::ONE (r times)`.
/// ///
/// /// Note that the output only depends on `r mod p`.
/// ///
/// /// This should be avoided in performance critical locations.
/// fn from_Int(int: Int) -> Self {
///     Self::from_char(Self::Char::from_int(int))
/// }
/// ```
///
/// This macro can be run for any `Int` where `Self::Char` implements `QuotientMap<Int>`.
/// It considerably cuts down on the amount of copy/pasted code.
macro_rules! from_integer_types {
    ($($type:ty),* $(,)? ) => {
        $( paste!{
        /// Given an integer `r`, return the sum of `r` copies of `ONE`:
        ///
        /// `r.Self::ONE =  Self::ONE + ... + Self::ONE (r times)`.
        ///
        /// Note that the output only depends on `r mod p`.
        ///
        /// This should be avoided in performance critical locations.
        fn [<from_ $type>](int: $type) -> Self {
            Self::from_char(Self::Char::from_int(int))
        }
    }
        )*
    };
}

/// A commutative ring `(R)` with prime characteristic `(p)`.
///
/// Whilst many rings with other characteristics exist, we expect every struct here
/// which implements CommutativeRing to also implement PrimeCharacteristicRing.
///
/// This trait provides a collection of convenience methods allowing elements of
/// simple integer classes `bool, u8, ...` to be converted into ring elements. These
/// should generally be used in non performance critical locations as converting elements
/// into the internal representation for the ring is often slow.
///
/// ### Mathematical Description
///
/// The characteristic is the unique smallest integer `r > 0` such that `0 = r . 1 = 1 + 1 + ... + 1 (r times)`.
/// For example, the characteristic of the modulo ring `ℤ/N` is `N`.
///
/// When the characteristic is prime, the ring `R` becomes an algebra over the field `ℤ/p` (Integers mod p).
pub trait PrimeCharacteristicRing: CommutativeRing {
    /// The field `ℤ/p`.
    type Char: PrimeField;

    /// Embed an element of the prime field `ℤ/p` into the ring `R`.
    ///
    /// Given any element `r ∈ ℤ/p`, represented as an integer between `0` and `p - 1`
    /// `from_char(r)` will be equal to:
    ///
    /// `Self::ONE + ... + Self::ONE (r times)`
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

    from_integer_types!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

    /// The elementary function `halve(a) = a/2`.
    ///
    /// Will error if the field characteristic is 2.
    #[must_use]
    fn halve(&self) -> Self {
        // This should be overwritten by most field implementations.
        self.clone() * Self::from_char(Self::Char::TWO.inverse())
    }

    /// Divide by a given power of two. `div_2exp_u64(a, exp) = a/2^exp`
    ///
    /// Will error if the field characteristic is 2.
    #[must_use]
    #[inline]
    fn div_2exp_u64(&self, exp: u64) -> Self {
        // This should be overwritten by most field implementations.
        self.clone() * Self::from_char(Self::Char::TWO.inverse().exp_u64(exp))
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
pub trait InjectiveMonomial<const N: u64> {
    fn injective_monomial(&self) -> Self;
}

/// A ring implements PermutationMonomial<N> if the algebraic function
/// `f(x) = x^N` is invertible and thus acts as a permutation on elements of the ring.
pub trait PermutationMonomial<const N: u64>: InjectiveMonomial<N> {
    fn monomial_inverse(&self) -> Self;
}

/// A ring `R` implements `FieldAlgebra<F>` if there is a natural
/// map from `F` into `R` such that the only element which maps
/// to `R::ZERO` is `F::ZERO`.
///
/// For the most part, we will usually expect `F` to be a field but there
/// are a few cases where it is handy to allow it to just be a ring.
///
/// ### Mathematical Description
///
/// Let `x` and `y` denote arbitrary elements of the `S`. Then
/// by "natural" map we require that our map `from`
/// has the following properties:
/// - Preserves Identity: `from(F::ONE) = R::ONE`
/// - Commutes with Addition: `from(x + y) = from(x) + from(y)`
/// - Commutes with Multiplication: `from(x * y) = from(x) * from(y)`
///
/// Such maps are known as ring homomorphisms and are injective if the
/// only element which maps to `R::ZERO` is `F::ZERO`.
///
/// The existence of this map makes `R` into an `F`-module. If `F` is a field
/// then this makes `R` into an `F`-Algebra and if `R` is also a field then
/// this means that `R` is a field extension of `F`.
pub trait FieldAlgebra<F>:
    From<F>
    + Add<F, Output = Self>
    + AddAssign<F>
    + Sub<F, Output = Self>
    + SubAssign<F>
    + Mul<F, Output = Self>
    + MulAssign<F>
    + PrimeCharacteristicRing
{
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
pub trait Serializable<F> {
    // We could alternatively call this BasedAlgebra?
    // The name is currently trying to indicate what this is meant to be
    // used for as opposed to being mathematically accurate.

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
    fn serialize(&self) -> Vec<F>;

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
    fn deserialize_slice(slice: &[F]) -> Self;

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
    fn deserialize_iter<I: Iterator<Item = F>>(iter: I) -> Self;
}

/// An element of a finite field.
///
/// A ring is a field if every element `x` has a unique multiplicative inverse `x^{-1}`
/// which satisfies `x * x^{-1} = F::ONE`.
pub trait Field:
    PrimeCharacteristicRing
    + FieldAlgebra<Self>
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

    /// Check if the given field element is equal to the unique additive identity (ZERO).
    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    /// Check if the given field element is equal to the unique multiplicative identity (ONE).
    fn is_one(&self) -> bool {
        *self == Self::ONE
    }

    /// The multiplicative inverse of this field element, if it exists.
    ///
    /// NOTE: The inverse of `0` is undefined and will return `None`.
    #[must_use]
    fn try_inverse(&self) -> Option<Self>;

    /// The multiplicative inverse of this field element, if it exists.
    ///
    /// NOTE: The inverse of `0` is undefined and will error.
    #[must_use]
    fn inverse(&self) -> Self {
        self.try_inverse().expect("Tried to invert zero")
    }

    /// The number of elements in the field.
    ///
    /// This will either be prime if the field is a PrimeField or a power of a
    /// prime if the field is an extension field.
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

    /// The number of bits required to define an element of this field.
    ///
    /// Usually due to storage and practical reasons the memory size of
    /// a field element will be a little larger than bits().
    #[inline]
    fn bits() -> usize {
        Self::order().bits() as usize
    }
}

/// Implementation of the quotient map `ℤ -> ℤ/p` which sends an integer `r` to its conjugacy class `[r]`.
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
    fn from_canonical_checked(int: Int) -> Option<Self> {
        // Q: Should from_canonical_checked error if it is outside the canonical range or just if
        // the safety bounds for from_canonical_unchecked are not satisfied?

        // For some fields, there is no benefit to the knowledge that the integer is in the canonical range so
        // we can always use from_int.
        Some(Self::from_int(int))
    }

    /// Convert a given integer into an element of the field `ℤ/p`. The input is guaranteed
    /// to lie within a specific range depending on `p`. If the input lies outside of this
    /// range, the output is undefined.
    ///
    /// In general `from_canonical_unchecked` will be faster for either `signed` or `unsigned`
    /// types but the specifics will depend on the field.
    ///
    /// # Safety
    /// - If `Int` is an unsigned integer type then the allowed range will include `[0, p - 1]`.
    /// - If `Int` is a signed integer type then the allowed range will include `[-(p - 1)/2, (p - 1)/2]`.
    ///
    /// In general
    ///
    /// to undefined behaviour. However this will be faster than `from_int/from_canonical_checked` in some
    /// circumstances and so we provide it here for careful use in performance critical applications.
    unsafe fn from_canonical_unchecked(int: Int) -> Self {
        // For some fields, there is no benefit to the knowledge that the integer is in the canonical range so
        // we default to from_int which is always safe and correct.
        Self::from_int(int)
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
    // TODO: Decide if this should be put into Commutative Ring.

    /// The field element 2 mod p.
    ///
    /// This is provided as a convenience as `TWO` occurs regularly in
    /// the proving system. This also is slightly faster than computing
    /// it via addition. Note that multiplication by `TWO` is discouraged.
    /// Instead of `a * TWO` use `a.double()` which will be faster.
    ///
    /// When p = 2, this is equal to ZERO.
    const TWO: Self;

    fn as_canonical_biguint(&self) -> BigUint;
}

/// A prime field `ℤ/p` with order `p < 2^64`.
pub trait PrimeField64: PrimeField {
    const ORDER_U64: u64;

    /// Return the representative of `value` in canonical form
    /// which lies in the range `0 <= x < ORDER_U64`.
    fn as_canonical_u64(&self) -> u64;

    /// Convert the field element to a u64 such that any two field elements
    /// representing the same value are converted to the same u64.
    ///
    /// This will be the fastest way to get a unique u64 representative
    /// from the field element and is intended for use in Hashing. In general,
    /// `val.hash_to_u64()` and `val.as_canonical_u64()` may be different.
    fn hash_to_u64(&self) -> u64;
}

/// A prime field `ℤ/p` with order `p < 2^32`.
pub trait PrimeField32: PrimeField64 {
    const ORDER_U32: u32;

    /// Return the representative of `value` in the canonical form
    /// which lies in the range `0 <= x < ORDER_U32`.
    fn as_canonical_u32(&self) -> u32;

    /// Convert the field element to a u32 such that any two field elements
    /// representing the same value are converted to the same u32.
    ///
    /// This will be the fastest way to get a unique u32 representative
    /// from the field element and is intended for use in Hashing. In general,
    /// `val.as_unique_u32()` and `val.as_canonical_u32()` may be different.
    fn hash_to_u32(&self) -> u32;
}

pub trait ExtensionField<Base: Field>: Field + FieldAlgebra<Base> {
    type ExtensionPacking: PackedFieldExtension<BaseField = Base, ExtField = Self>
        + 'static
        + Copy
        + Send
        + Sync;

    const D: usize;

    /// Determine if the given element lies in the base field.
    fn is_in_basefield(&self) -> bool;

    /// If the element lies in the base field project it down.
    /// Otherwise return None.
    fn as_base(&self) -> Option<Base>;
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
pub struct Powers<CR> {
    pub base: CR,
    pub current: CR,
}

impl<CR: CommutativeRing> Iterator for Powers<CR> {
    type Item = CR;

    fn next(&mut self) -> Option<CR> {
        let result = self.current.clone();
        self.current *= self.base.clone();
        Some(result)
    }
}
