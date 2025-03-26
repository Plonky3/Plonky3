use alloc::vec::Vec;
use core::iter::Sum;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ops::Mul;

use num_bigint::BigUint;
use p3_maybe_rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::field::Field;
use crate::{PackedValue, PrimeCharacteristicRing, PrimeField, PrimeField32, TwoAdicField};

/// Computes `Z_H(x)`, where `Z_H` is the vanishing polynomial of a multiplicative subgroup of order `2^log_n`.
pub fn two_adic_subgroup_vanishing_polynomial<F: TwoAdicField>(log_n: usize, x: F) -> F {
    x.exp_power_of_2(log_n) - F::ONE
}

/// Computes `Z_{sH}(x)`, where `Z_{sH}` is the vanishing polynomial of the given coset of a multiplicative
/// subgroup of order `2^log_n`.
pub fn two_adic_coset_vanishing_polynomial<F: TwoAdicField>(log_n: usize, shift: F, x: F) -> F {
    x.exp_power_of_2(log_n) - shift.exp_power_of_2(log_n)
}

/// Computes a multiplicative subgroup whose order is known in advance.
pub fn cyclic_subgroup_known_order<F: Field>(
    generator: F,
    order: usize,
) -> impl Iterator<Item = F> + Clone {
    generator.powers().take(order)
}

/// Computes a coset of a multiplicative subgroup whose order is known in advance.
pub fn cyclic_subgroup_coset_known_order<F: Field>(
    generator: F,
    shift: F,
    order: usize,
) -> impl Iterator<Item = F> + Clone {
    generator.shifted_powers(shift).take(order)
}

pub fn scale_vec<F: Field>(s: F, vec: Vec<F>) -> Vec<F> {
    vec.into_iter().map(|x| s * x).collect()
}

pub fn scale_slice_in_place<F: Field>(s: F, slice: &mut [F]) {
    let (packed, sfx) = F::Packing::pack_slice_with_suffix_mut(slice);
    let packed_s: F::Packing = s.into();
    packed.par_iter_mut().for_each(|x| *x *= packed_s);
    sfx.iter_mut().for_each(|x| *x *= s);
}

/// `x += y * s`, where `s` is a scalar.
pub fn add_scaled_slice_in_place<F, Y>(x: &mut [F], y: Y, s: F)
where
    F: Field,
    Y: Iterator<Item = F>,
{
    if x.len() == 0 {
        return;
    }

    // Convert iterator to a Vec for now, while still using direct iteration
    let y_vec: Vec<F> = y.take(x.len()).collect();
    
    // Check if we got enough elements
    if y_vec.len() < x.len() {
        return;
    }
    
    let (packed_x, sfx_x) = F::Packing::pack_slice_with_suffix_mut(x);
    let (packed_y, sfx_y) = F::Packing::pack_slice_with_suffix(&y_vec);
    
    // Process packed elements in parallel
    packed_x.par_iter_mut().zip(packed_y).for_each(|(x_i, y_i)| {
        *x_i += y_i * s;
    });
    
    // Process remaining elements
    sfx_x.iter_mut().zip(sfx_y).for_each(|(x_i, y_i)| {
        *x_i += y_i * s;
    });
}

// The ideas for the following work around come from the construe crate along with
// the playground example linked in the following comment:
// https://github.com/rust-lang/rust/issues/115403#issuecomment-1701000117

// The goal is to want to make field_to_array a const function in order
// to allow us to convert R constants to BinomialExtensionField<R, D> constants.
//
// The natural approach would be:
// fn field_to_array<R: PrimeCharacteristicRing, const D: usize>(x: R) -> [R; D]
//      let mut arr: [R; D] = [R::ZERO; D];
//      arr[0] = x
//      arr
//
// Unfortunately this doesn't compile as R does not implement Copy and so instead
// implements Drop which cannot be run in constant contexts. Clearly nothing should
// actually be dropped by the above function but the compiler is unable to determine this.
// There is a rust issue for this: https://github.com/rust-lang/rust/issues/73255
// but it seems unlikely to be stabilized anytime soon.
//
// The natural workaround for this is to use MaybeUninit and set each element of the list
// separately. This mostly works but we end up with an array of the form [MaybeUninit<T>; N]
// and there is not currently a way in the standard library to convert this to [T; N].
// There is a method on nightly: array_assume_init so this function should be reworked after
// that has stabilized (More details in Rust issue: https://github.com/rust-lang/rust/issues/96097).
//
// Annoyingly, both transmute and transmute_copy fail here. The first because it cannot handle
// const generics and the second due to interior mutability and the inability to use &mut in const
// functions.
//
// The solution is to implement the map [MaybeUninit<T>; D]) -> MaybeUninit<[T; D]>
// using Union types and ManuallyDrop to essentially do a manual transmute.

union HackyWorkAround<T, const D: usize> {
    complete: ManuallyDrop<MaybeUninit<[T; D]>>,
    elements: ManuallyDrop<[MaybeUninit<T>; D]>,
}

impl<T, const D: usize> HackyWorkAround<T, D> {
    const fn transpose(arr: [MaybeUninit<T>; D]) -> MaybeUninit<[T; D]> {
        // This is safe as [MaybeUninit<T>; D]> and MaybeUninit<[T; D]> are
        // the same type regardless of T. Both are an array or size equal to [T; D]
        // with some data potentially not initialized.
        let transpose = Self {
            elements: ManuallyDrop::new(arr),
        };
        unsafe { ManuallyDrop::into_inner(transpose.complete) }
    }
}

/// Extend a ring `R` element `x` to an array of length `D`
/// by filling zeros.
#[inline]
pub const fn field_to_array<R: PrimeCharacteristicRing, const D: usize>(x: R) -> [R; D] {
    let mut arr: [MaybeUninit<R>; D] = unsafe { MaybeUninit::uninit().assume_init() };

    arr[0] = MaybeUninit::new(x);
    let mut acc = 1;
    loop {
        if acc == D {
            break;
        }
        arr[acc] = MaybeUninit::new(R::ZERO);
        acc += 1;
    }
    // If the code has reached this point every element of arr is correctly initialized.
    // Hence we are safe to reinterpret the array as [R; D].

    unsafe { HackyWorkAround::transpose(arr).assume_init() }
}

/// Given an element x from a 32 bit field F_P compute x/2.
#[inline]
pub const fn halve_u32<const P: u32>(input: u32) -> u32 {
    let shift = (P + 1) >> 1;
    let shr = input >> 1;
    let lo_bit = input & 1;
    let shr_corr = shr + shift;
    if lo_bit == 0 { shr } else { shr_corr }
}

/// Given an element x from a 64 bit field F_P compute x/2.
#[inline]
pub const fn halve_u64<const P: u64>(input: u64) -> u64 {
    let shift = (P + 1) >> 1;
    let shr = input >> 1;
    let lo_bit = input & 1;
    let shr_corr = shr + shift;
    if lo_bit == 0 { shr } else { shr_corr }
}

/// Given a slice of SF elements, reduce them to a TF element using a 2^32-base decomposition.
///
/// This is optimised assuming that the characteristic of TF is greater than 2^64.
pub fn reduce_32<SF: PrimeField32, TF: PrimeField>(vals: &[SF]) -> TF {
    // If the characteristic of TF is > 2^64, from_int and from_canonical_unchecked act identically
    // on u64 and u32 inputs so we use the safer option.
    let po2 = TF::from_int(1u64 << 32);
    let mut result = TF::ZERO;
    for val in vals.iter().rev() {
        result = result * po2 + TF::from_int(val.as_canonical_u32());
    }
    result
}

/// Given an SF element, split it to a vector of TF elements using a 2^64-base decomposition.
///
/// We use a 2^64-base decomposition for a field of size ~2^32 because then the bias will be
/// at most ~1/2^32 for each element after the reduction.
pub fn split_32<SF: PrimeField, TF: PrimeField32>(val: SF, n: usize) -> Vec<TF> {
    let po2 = BigUint::from(1u128 << 64);
    let mut val = val.as_canonical_biguint();
    let mut result = Vec::new();
    for _ in 0..n {
        let mask: BigUint = po2.clone() - BigUint::from(1u128);
        let digit: BigUint = val.clone() & mask;
        let digit_u64s = digit.to_u64_digits();
        if digit_u64s.is_empty() {
            result.push(TF::ZERO)
        } else {
            result.push(TF::from_int(digit_u64s[0]));
        }
        val /= po2.clone();
    }
    result
}

/// Maximally generic dot product.
pub fn dot_product<S, LI, RI>(li: LI, ri: RI) -> S
where
    LI: Iterator,
    RI: Iterator,
    LI::Item: Mul<RI::Item>,
    S: Sum<<LI::Item as Mul<RI::Item>>::Output>,
{
    li.zip(ri).map(|(l, r)| l * r).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use crate::field::Field;
    use crate::PackedValue;
    use rand;

    /// A naive implementation of add_scaled_slice_in_place for testing
    fn add_scaled_slice_in_place_naive<F: Field>(x: &mut [F], y: impl Iterator<Item = F>, s: F) {
        for (x_i, y_i) in x.iter_mut().zip(y) {
            *x_i += y_i * s;
        }
    }

    // Define a simple test field implementation for our tests
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TestField(u32);
    
    impl TestField {
        fn new(val: u32) -> Self {
            Self(val % 7) // Simple modular field with modulus 7
        }
    }
    
    // Implement basic field operations for TestField
    impl Field for TestField {
        type Packing = TestFieldPacking;
        
        const ZERO: Self = Self(0);
        const ONE: Self = Self(1);
        const TWO: Self = Self(2);
        const NEG_ONE: Self = Self(6); // -1 mod 7 = 6
        
        fn random(_rng: impl rand::Rng) -> Self {
            unimplemented!("Not needed for test")
        }
        
        fn is_zero(&self) -> bool {
            self.0 == 0
        }
        
        fn square(&self) -> Self {
            Self::new(self.0 * self.0)
        }
        
        fn cube(&self) -> Self {
            Self::new(self.0 * self.0 * self.0)
        }
        
        fn double(&self) -> Self {
            Self::new(self.0 * 2)
        }
        
        fn exp_power_of_2(&self, _power: usize) -> Self {
            unimplemented!("Not needed for test")
        }
        
        fn powers(&self) -> impl Iterator<Item = Self> {
            unimplemented!("Not needed for test")
        }
    }
    
    // Implementation of standard operators for TestField
    impl core::ops::Add for TestField {
        type Output = Self;
        
        fn add(self, rhs: Self) -> Self::Output {
            Self::new(self.0 + rhs.0)
        }
    }
    
    impl core::ops::AddAssign for TestField {
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }
    
    impl core::ops::Mul for TestField {
        type Output = Self;
        
        fn mul(self, rhs: Self) -> Self::Output {
            Self::new(self.0 * rhs.0)
        }
    }
    
    impl core::ops::MulAssign for TestField {
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }
    
    impl core::ops::Sub for TestField {
        type Output = Self;
        
        fn sub(self, rhs: Self) -> Self::Output {
            Self::new(self.0 + 7 - rhs.0)
        }
    }
    
    impl core::ops::SubAssign for TestField {
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }
    
    impl core::ops::Neg for TestField {
        type Output = Self;
        
        fn neg(self) -> Self::Output {
            Self::new(7 - self.0)
        }
    }
    
    // Define a mock packing implementation to test the full code path
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TestFieldPacking(TestField);
    
    impl From<TestField> for TestFieldPacking {
        fn from(value: TestField) -> Self {
            Self(value)
        }
    }
    
    impl core::ops::Add for TestFieldPacking {
        type Output = Self;
        
        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0)
        }
    }
    
    impl core::ops::AddAssign for TestFieldPacking {
        fn add_assign(&mut self, rhs: Self) {
            self.0 += rhs.0;
        }
    }
    
    impl core::ops::Mul for TestFieldPacking {
        type Output = Self;
        
        fn mul(self, rhs: Self) -> Self::Output {
            Self(self.0 * rhs.0)
        }
    }
    
    impl core::ops::MulAssign for TestFieldPacking {
        fn mul_assign(&mut self, rhs: Self) {
            self.0 *= rhs.0;
        }
    }
    
    impl PackedValue for TestFieldPacking {
        type Scalar = TestField;
        const STRIDE: usize = 1;
        
        // For testing purposes, these methods create vectors with just one element
        fn pack_slice(slice: &[Self::Scalar]) -> (alloc::vec::Vec<Self>, &[Self::Scalar]) {
            let mut result = Vec::new();
            if !slice.is_empty() {
                result.push(Self(slice[0]));
                (result, &slice[1..])
            } else {
                (result, slice)
            }
        }
        
        fn pack_slice_with_suffix(slice: &[Self::Scalar]) -> (alloc::vec::Vec<Self>, &[Self::Scalar]) {
            Self::pack_slice(slice)
        }
        
        fn pack_slice_mut(slice: &mut [Self::Scalar]) -> (alloc::vec::Vec<Self>, &mut [Self::Scalar]) {
            let mut result = Vec::new();
            if !slice.is_empty() {
                result.push(Self(slice[0]));
                (result, &mut slice[1..])
            } else {
                (result, slice)
            }
        }
        
        fn pack_slice_with_suffix_mut(slice: &mut [Self::Scalar]) -> (alloc::vec::Vec<Self>, &mut [Self::Scalar]) {
            Self::pack_slice_mut(slice)
        }
    }

    #[test]
    fn test_add_scaled_slice_in_place() {
        // Test with various vector sizes
        let sizes = [0, 1, 2, 4, 7, 8];
        
        for &size in &sizes {
            // Create test vectors
            let mut vec1: Vec<TestField> = (0..size).map(|i| TestField::new(i as u32)).collect();
            let mut vec2 = vec1.clone();
            
            // Create y vector and scalar
            let y_vec: Vec<TestField> = (0..size).map(|i| TestField::new((i * 2) as u32)).collect();
            let s = TestField::new(3);
            
            // Apply the optimized version to vec1
            add_scaled_slice_in_place(&mut vec1, y_vec.iter().cloned(), s);
            
            // Apply the naive version to vec2
            add_scaled_slice_in_place_naive(&mut vec2, y_vec.iter().cloned(), s);
            
            // Verify that both vectors are identical
            assert_eq!(vec1, vec2, "Optimized and naive implementations should produce the same result for size {}", size);
        }
    }
}
