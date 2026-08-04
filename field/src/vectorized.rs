//! Lockstep evaluation over multiple packed vectors, trading register pressure
//! for instruction-level parallelism in latency-bound field arithmetic.
//!
//! Inspired by stwo's `Vectorized` type:
//! <https://github.com/starkware-libs/stwo/blob/cca98119f/crates/stwo/src/prover/backend/simd/very_packed_m31.rs>

use core::array;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue,
    PrimeCharacteristicRing,
};

/// `N` packed base-field vectors operated on in lockstep.
///
/// A single packed vector often cannot saturate the CPU's multiplier pipes: a
/// dependency chain of packed multiplications leaves most issue slots idle
/// (e.g. NEON's modular multiply has ~10 cycles of latency at ~1.25 cycles of
/// throughput per vector). Widening the *data type* rather than the loop turns
/// every ring operation into `N` independent instructions back to back, giving
/// the out-of-order core `N` interleaved dependency chains without changing the
/// shape of the expression being evaluated.
///
/// Lane `i` of the logical vector lives in `self.0[i / W].as_slice()[i % W]`
/// where `W = F::Packing::WIDTH`, i.e. components hold consecutive blocks of
/// lanes.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
#[must_use]
pub struct Vectorized<F: Field, const N: usize>(pub [F::Packing; N]);

/// `N` packed extension-field vectors operated on in lockstep.
///
/// The extension-field counterpart of [`Vectorized`]: lane `i` corresponds to
/// lane `i` of a `Vectorized<F, N>` operand, so the two types can be mixed in
/// the same expression (`VectorizedExt * Vectorized`, etc.).
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
#[must_use]
pub struct VectorizedExt<F: Field, EF: ExtensionField<F>, const N: usize>(
    pub [EF::ExtensionPacking; N],
);

impl<F: Field, const N: usize> Vectorized<F, N> {
    /// Map a function over the `N` packed components.
    #[inline]
    fn map(self, f: impl FnMut(F::Packing) -> F::Packing) -> Self {
        Self(self.0.map(f))
    }

    /// Combine two values component-wise.
    #[inline]
    fn zip_with<T: Copy>(
        self,
        rhs: &[T; N],
        mut f: impl FnMut(F::Packing, T) -> F::Packing,
    ) -> Self {
        Self(array::from_fn(|i| f(self.0[i], rhs[i])))
    }
}

impl<F: Field, EF: ExtensionField<F>, const N: usize> VectorizedExt<F, EF, N> {
    /// Map a function over the `N` packed components.
    #[inline]
    fn map(self, f: impl FnMut(EF::ExtensionPacking) -> EF::ExtensionPacking) -> Self {
        Self(self.0.map(f))
    }

    /// Combine two values component-wise.
    #[inline]
    fn zip_with<T: Copy>(
        self,
        rhs: &[T; N],
        mut f: impl FnMut(EF::ExtensionPacking, T) -> EF::ExtensionPacking,
    ) -> Self {
        Self(array::from_fn(|i| f(self.0[i], rhs[i])))
    }

    /// Build from basis coefficients, where coefficient `d` is the vectorized
    /// base-field value `coefficients[d]`.
    ///
    /// This is the vectorized analogue of
    /// [`BasedVectorSpace::from_basis_coefficients_fn`]; it takes a slice
    /// rather than a closure so each coefficient is computed once and shared
    /// by all `N` components.
    #[inline]
    pub fn from_vectorized_basis_coefficients(coefficients: &[Vectorized<F, N>]) -> Self {
        debug_assert_eq!(coefficients.len(), EF::DIMENSION);
        Self(array::from_fn(|i| {
            EF::ExtensionPacking::from_basis_coefficients_fn(|d| coefficients[d].0[i])
        }))
    }

    /// Extract the extension-field element at logical lane `i`.
    ///
    /// Lanes `0..F::Packing::WIDTH` come from component `0`, the next
    /// `F::Packing::WIDTH` from component `1`, and so on.
    #[inline]
    pub fn extract(&self, i: usize) -> EF {
        let width = F::Packing::WIDTH;
        self.0[i / width].extract(i % width)
    }
}

impl<F: Field, const N: usize> Default for Vectorized<F, N> {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<F: Field, EF: ExtensionField<F>, const N: usize> Default for VectorizedExt<F, EF, N> {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<F: Field, const N: usize> From<F> for Vectorized<F, N> {
    #[inline]
    fn from(value: F) -> Self {
        Self([F::Packing::from(value); N])
    }
}

impl<F: Field, EF: ExtensionField<F>, const N: usize> From<EF> for VectorizedExt<F, EF, N> {
    #[inline]
    fn from(value: EF) -> Self {
        Self([EF::ExtensionPacking::from(value); N])
    }
}

impl<F: Field, EF: ExtensionField<F>, const N: usize> From<Vectorized<F, N>>
    for VectorizedExt<F, EF, N>
{
    #[inline]
    fn from(value: Vectorized<F, N>) -> Self {
        Self(value.0.map(EF::ExtensionPacking::from))
    }
}

macro_rules! impl_binary_ops {
    ($ty:ty, $($bound:tt)*) => {
        impl<$($bound)*> Add for $ty {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                self.zip_with(&rhs.0, |x, y| x + y)
            }
        }

        impl<$($bound)*> Sub for $ty {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                self.zip_with(&rhs.0, |x, y| x - y)
            }
        }

        impl<$($bound)*> Mul for $ty {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                self.zip_with(&rhs.0, |x, y| x * y)
            }
        }

        impl<$($bound)*> Neg for $ty {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                self.map(|x| -x)
            }
        }

        impl<$($bound)*> AddAssign for $ty {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl<$($bound)*> SubAssign for $ty {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl<$($bound)*> MulAssign for $ty {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl<$($bound)*> Sum for $ty {
            #[inline]
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::ZERO, |acc, x| acc + x)
            }
        }

        impl<$($bound)*> Product for $ty {
            #[inline]
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::ONE, |acc, x| acc * x)
            }
        }
    };
}

macro_rules! impl_scalar_ops {
    ($ty:ty, $scalar:ty, $($bound:tt)*) => {
        impl<$($bound)*> Add<$scalar> for $ty {
            type Output = Self;
            #[inline]
            fn add(self, rhs: $scalar) -> Self {
                self.map(|x| x + rhs)
            }
        }

        impl<$($bound)*> Sub<$scalar> for $ty {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: $scalar) -> Self {
                self.map(|x| x - rhs)
            }
        }

        impl<$($bound)*> Mul<$scalar> for $ty {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: $scalar) -> Self {
                self.map(|x| x * rhs)
            }
        }

        impl<$($bound)*> AddAssign<$scalar> for $ty {
            #[inline]
            fn add_assign(&mut self, rhs: $scalar) {
                *self = *self + rhs;
            }
        }

        impl<$($bound)*> SubAssign<$scalar> for $ty {
            #[inline]
            fn sub_assign(&mut self, rhs: $scalar) {
                *self = *self - rhs;
            }
        }

        impl<$($bound)*> MulAssign<$scalar> for $ty {
            #[inline]
            fn mul_assign(&mut self, rhs: $scalar) {
                *self = *self * rhs;
            }
        }
    };
}

impl_binary_ops!(Vectorized<F, N>, F: Field, const N: usize);
impl_scalar_ops!(Vectorized<F, N>, F, F: Field, const N: usize);
impl_binary_ops!(VectorizedExt<F, EF, N>, F: Field, EF: ExtensionField<F>, const N: usize);
impl_scalar_ops!(VectorizedExt<F, EF, N>, EF, F: Field, EF: ExtensionField<F>, const N: usize);

impl<F: Field, EF: ExtensionField<F>, const N: usize> Add<Vectorized<F, N>>
    for VectorizedExt<F, EF, N>
{
    type Output = Self;
    #[inline]
    fn add(self, rhs: Vectorized<F, N>) -> Self {
        self.zip_with(&rhs.0, |x, y| x + y)
    }
}

impl<F: Field, EF: ExtensionField<F>, const N: usize> Sub<Vectorized<F, N>>
    for VectorizedExt<F, EF, N>
{
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Vectorized<F, N>) -> Self {
        self.zip_with(&rhs.0, |x, y| x - y)
    }
}

impl<F: Field, EF: ExtensionField<F>, const N: usize> Mul<Vectorized<F, N>>
    for VectorizedExt<F, EF, N>
{
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Vectorized<F, N>) -> Self {
        self.zip_with(&rhs.0, |x, y| x * y)
    }
}

impl<F: Field, EF: ExtensionField<F>, const N: usize> AddAssign<Vectorized<F, N>>
    for VectorizedExt<F, EF, N>
{
    #[inline]
    fn add_assign(&mut self, rhs: Vectorized<F, N>) {
        *self = *self + rhs;
    }
}

impl<F: Field, EF: ExtensionField<F>, const N: usize> SubAssign<Vectorized<F, N>>
    for VectorizedExt<F, EF, N>
{
    #[inline]
    fn sub_assign(&mut self, rhs: Vectorized<F, N>) {
        *self = *self - rhs;
    }
}

impl<F: Field, EF: ExtensionField<F>, const N: usize> MulAssign<Vectorized<F, N>>
    for VectorizedExt<F, EF, N>
{
    #[inline]
    fn mul_assign(&mut self, rhs: Vectorized<F, N>) {
        *self = *self * rhs;
    }
}

impl<F: Field, const N: usize> PrimeCharacteristicRing for Vectorized<F, N> {
    type PrimeSubfield = F::PrimeSubfield;

    const ZERO: Self = Self([F::Packing::ZERO; N]);
    const ONE: Self = Self([F::Packing::ONE; N]);
    const TWO: Self = Self([F::Packing::TWO; N]);
    const NEG_ONE: Self = Self([F::Packing::NEG_ONE; N]);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        F::from_prime_subfield(f).into()
    }

    #[inline]
    fn double(&self) -> Self {
        self.map(|x| x.double())
    }

    #[inline]
    fn halve(&self) -> Self {
        self.map(|x| x.halve())
    }

    #[inline]
    fn square(&self) -> Self {
        self.map(|x| x.square())
    }

    #[inline]
    fn cube(&self) -> Self {
        self.map(|x| x.cube())
    }

    #[inline]
    fn exp_const_u64<const POWER: u64>(&self) -> Self {
        self.map(|x| x.exp_const_u64::<POWER>())
    }

    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        self.map(|x| x.mul_2exp_u64(exp))
    }

    #[inline]
    fn dot_product<const M: usize>(u: &[Self; M], v: &[Self; M]) -> Self {
        Self(array::from_fn(|i| {
            F::Packing::dot_product::<M>(
                &array::from_fn::<_, M, _>(|j| u[j].0[i]),
                &array::from_fn::<_, M, _>(|j| v[j].0[i]),
            )
        }))
    }

    #[inline]
    fn sum_array<const M: usize>(input: &[Self]) -> Self {
        assert_eq!(input.len(), M);
        Self(array::from_fn(|i| {
            F::Packing::sum_array::<M>(&array::from_fn::<_, M, _>(|j| input[j].0[i]))
        }))
    }
}

impl<F: Field, EF: ExtensionField<F>, const N: usize> PrimeCharacteristicRing
    for VectorizedExt<F, EF, N>
{
    type PrimeSubfield = <EF::ExtensionPacking as PrimeCharacteristicRing>::PrimeSubfield;

    const ZERO: Self = Self([EF::ExtensionPacking::ZERO; N]);
    const ONE: Self = Self([EF::ExtensionPacking::ONE; N]);
    const TWO: Self = Self([EF::ExtensionPacking::TWO; N]);
    const NEG_ONE: Self = Self([EF::ExtensionPacking::NEG_ONE; N]);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self([EF::ExtensionPacking::from_prime_subfield(f); N])
    }

    #[inline]
    fn double(&self) -> Self {
        self.map(|x| x.double())
    }

    #[inline]
    fn halve(&self) -> Self {
        self.map(|x| x.halve())
    }

    #[inline]
    fn square(&self) -> Self {
        self.map(|x| x.square())
    }
}

impl<F: Field, const N: usize> Algebra<F> for Vectorized<F, N> {
    const BATCHED_LC_CHUNK: usize = <F::Packing as Algebra<F>>::BATCHED_LC_CHUNK;

    #[inline]
    fn mixed_dot_product<const M: usize>(a: &[Self; M], f: &[F; M]) -> Self {
        Self(array::from_fn(|i| {
            F::Packing::mixed_dot_product(&array::from_fn(|j| a[j].0[i]), f)
        }))
    }
}

impl<F: Field, EF: ExtensionField<F>, const N: usize> Algebra<EF> for VectorizedExt<F, EF, N> {
    const BATCHED_LC_CHUNK: usize = <EF::ExtensionPacking as Algebra<EF>>::BATCHED_LC_CHUNK;

    #[inline]
    fn mixed_dot_product<const M: usize>(a: &[Self; M], f: &[EF; M]) -> Self {
        Self(array::from_fn(|i| {
            EF::ExtensionPacking::mixed_dot_product(&array::from_fn(|j| a[j].0[i]), f)
        }))
    }
}

impl<F: Field, EF: ExtensionField<F>, const N: usize> Algebra<Vectorized<F, N>>
    for VectorizedExt<F, EF, N>
{
}

// SAFETY: `Vectorized<F, N>` is `repr(transparent)` over `[F::Packing; N]` and
// `F::Packing: PackedField` guarantees that `F::Packing` can be cast to/from
// `[F; F::Packing::WIDTH]` without UB. Hence `Vectorized<F, N>` can be cast
// to/from `[F; F::Packing::WIDTH * N]`.
unsafe impl<F: Field, const N: usize> PackedValue for Vectorized<F, N> {
    type Value = F;

    const WIDTH: usize = F::Packing::WIDTH * N;

    #[inline]
    fn from_fn<Fn>(mut f: Fn) -> Self
    where
        Fn: FnMut(usize) -> Self::Value,
    {
        Self(array::from_fn(|i| {
            F::Packing::from_fn(|j| f(i * F::Packing::WIDTH + j))
        }))
    }

    #[inline]
    fn from_slice(slice: &[Self::Value]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        let (_, values, _) = unsafe { slice.align_to::<Self>() };
        assert_eq!(values.len(), 1, "slice is not aligned to Self");
        &values[0]
    }

    #[inline]
    fn from_slice_mut(slice: &mut [Self::Value]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        let (_, values, _) = unsafe { slice.align_to_mut::<Self>() };
        assert_eq!(values.len(), 1, "slice is not aligned to Self");
        &mut values[0]
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Value] {
        unsafe { core::slice::from_raw_parts(core::ptr::from_ref(self).cast(), Self::WIDTH) }
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [Self::Value] {
        unsafe { core::slice::from_raw_parts_mut(core::ptr::from_mut(self).cast(), Self::WIDTH) }
    }
}
