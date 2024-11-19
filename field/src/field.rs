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
///
/// The is the a basic building block trait which implements addition and multiplication.
/// Examples of structs which should implement this trait are structs containing
///
/// - A single finite field element.
/// - A symbolic expression which may be evaluated to a finite field element.
/// - an array of finite field elements.
///
/// In practice every struct which implements this is expected to also implement PrimeCharacteristicRing.
///
/// ### Mathematical Description
///
/// Mathematically a commutative ring is an algebraic structure with two operations Addition `+`
/// and Multiplication `*` which satisfy a collection of properties. For ease of writing, in what follows
/// let `x, y, z` denote arbitrary elements of the ring.
///
/// Both operations must be:
///
/// Commutative => `x + y = y + x` and `x*y = y*x`
///
/// Associative => `x + (y + z) = (x + y) + z` and `x*(y*z) = (x*y)*z`
///
/// Unital      => There exists identity elements `ZERO` and `ONE` respectively meaning
///                `x + ZERO = x` and `x * ONE = x`.
///
/// In addition to the above, Addition must be invertible. Meaning for any `x` there exists
/// a unique inverse `(-x)` satisfying `x + (-x) = ZERO`.
///
/// Finally, the operations must satisfy the distributive property:
/// `x * (y + z) = (x*y) + (x*z)`.
///
/// The simplest examples of commutative rings are the integers (`ℤ`), and the integers mod `N` (`ℤ/N`).
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
    /// For every element `x` in the ring we require the following properties:
    ///
    /// `x + ZERO = ZERO + x = x,`
    ///
    /// `x + (-x) = (-x) + x = ZERO.`
    const ZERO: Self;

    /// The multiplicative identity of the ring
    ///
    /// For every element `x` in the ring we require the following property:
    ///
    /// `x*ONE = ONE*x = x.`
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

/// A commutative ring `(R)` with prime characteristic `(p)`.
///
/// Whilst many rings with other characteristics exist, we expect every struct here
/// which implements CommutativeRing to also implement PrimeCharacteristicRing.
///
/// This struct provides a collection of convenience methods allowing elements of
/// simple integer classes `bool, u8, ...` to be converted into ring elements. These
/// should generally be used in non performance critical locations. In particular,
/// any computations which can be performed in the field `ℤ/p` should be as this
/// will be faster than computing them in the ring.
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

    /// Given an integer `r`, return the sum of `r` copies of `ONE`:
    ///
    /// `r.Self::ONE =  Self::ONE + ... + Self::ONE (r times)`.
    ///
    /// Note that the output only depends on `r mod p`.
    ///
    /// This should be avoided in performance critical locations.
    fn from_u8(int: u8) -> Self {
        Self::from_char(Self::Char::from_int(int))
    }

    /// Given an integer `r`, return the sum of `r` copies of `ONE`:
    ///
    /// `r.Self::ONE =  Self::ONE + ... + Self::ONE (r times)`.
    ///
    ///
    /// This should be avoided in performance critical locations.
    fn from_u16(int: u16) -> Self {
        Self::from_char(Self::Char::from_int(int))
    }

    /// Given an integer `r`, return the sum of `r` copies of `ONE`:
    ///
    /// `r.Self::ONE =  Self::ONE + ... + Self::ONE (r times)`.
    ///
    /// Note that the output only depends on `r mod p`.
    ///
    /// This should be avoided in performance critical locations.
    fn from_u32(int: u32) -> Self {
        Self::from_char(Self::Char::from_int(int))
    }

    /// Given an integer `r`, return the sum of `r` copies of `ONE`:
    ///
    /// `r.Self::ONE =  Self::ONE + ... + Self::ONE (r times)`.
    ///
    /// Note that the output only depends on `r mod p`.
    ///
    /// This should be avoided in performance critical locations.
    fn from_u64(int: u64) -> Self {
        Self::from_char(Self::Char::from_int(int))
    }

    /// Given an integer `r`, return the sum of `r` copies of `ONE`:
    ///
    /// `r.Self::ONE =  Self::ONE + ... + Self::ONE (r times)`.
    ///
    /// Note that the output only depends on `r mod p`.
    ///
    /// This should be avoided in performance critical locations.
    fn from_usize(int: usize) -> Self {
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

    // Q: Can these methods be changed to accept u8 or smaller?
    // To what extent do we actually need to support multiplication
    // by 2^exp for large exp?

    // These are also basically unused though they are probably worth keeping around
    // as they could be helpful in a couple of places.

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

/// A ring should implement InjectiveMonomial<N> if the algebraic function
/// `f(x) = x^N` is an injective map on elements of the ring.
///
/// We do not enforce that this map be invertible as there are useful
/// cases such as polynomials or symbolic expressions where no inverse exists.
///
/// However if the ring is a field with order `q` or an array of such field elements,
/// then `f(x) = x^N` will be injective if and only if it is invertible and so in
/// such cases this monomial acts as a permutation. Moreover, this will occur
/// exactly when `N` and `q - 1` are relatively prime i.e. `gcd(N, q - 1) = 1`.
pub trait InjectiveMonomial<const N: u64> {
    fn injective_monomial(&self) -> Self;
}

/// A ring should implement PermutationMonomial<N> if the algebraic function
/// `f(x) = x^N` is invertible and thus acts as a permutation on elements of the ring.
pub trait PermutationMonomial<const N: u64>: InjectiveMonomial<N> {
    fn monomial_inverse(&self) -> Self;
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
pub trait FieldAlgebra<F: Field>: PrimeCharacteristicRing {
    /// Interpret a field element as a commutative algebra element.
    ///
    /// Mathematically speaking, this map is a ring homomorphism from the base field
    /// to the commutative algebra. The existence of this map makes this structure
    /// an algebra and not simply a commutative ring.
    fn from_f(f: F) -> Self;
}

/// An element of a finite field.
pub trait Field:
    PrimeCharacteristicRing
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

/// Every field is trivially a field algebra over itself.
impl<F: Field> FieldAlgebra<F> for F {
    fn from_f(f: F) -> Self {
        f
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
pub trait FieldExtensionAlgebra<
    BaseField: Field,
    BaseRing: FieldAlgebra<BaseField>,
    ExtField: ExtensionField<BaseField>,
>:
    FieldAlgebra<BaseField>
    + FieldAlgebra<ExtField>
    + From<BaseRing>
    + Add<BaseRing, Output = Self>
    + AddAssign<BaseRing>
    + Sub<BaseRing, Output = Self>
    + SubAssign<BaseRing>
    + Mul<BaseRing, Output = Self>
    + MulAssign<BaseRing>
{
    const D: usize;

    fn from_base(b: BaseRing) -> Self;

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
    fn from_base_slice(bs: &[BaseRing]) -> Self;

    /// Similar to `core:array::from_fn`, with the same caveats as
    /// `from_base_slice`.
    fn from_base_fn<F: FnMut(usize) -> BaseRing>(f: F) -> Self;
    fn from_base_iter<I: Iterator<Item = BaseRing>>(iter: I) -> Self;

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
    fn as_base_slice(&self) -> &[BaseRing];

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
        let mut vec = vec![BaseRing::ZERO; Self::D];
        vec[exponent] = BaseRing::ONE;
        Self::from_base_slice(&vec)
    }
}

pub trait ExtensionField<Base: Field>: Field + FieldExtensionAlgebra<Base, Base, Self> {
    type ExtensionPacking: FieldExtensionAlgebra<Base, Base::Packing, Self>
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

impl<F: Field, FA: FieldAlgebra<F>> FieldExtensionAlgebra<F, FA, F> for FA {
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

    fn from_base_fn<Fn: FnMut(usize) -> FA>(mut f: Fn) -> Self {
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

impl<CR: CommutativeRing> Iterator for Powers<CR> {
    type Item = CR;

    fn next(&mut self) -> Option<CR> {
        let result = self.current.clone();
        self.current *= self.base.clone();
        Some(result)
    }
}
