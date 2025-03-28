use alloc::vec::Vec;
use core::iter::Sum;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ops::Mul;

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
    // TODO: Use PackedField
    x.iter_mut().zip(y).for_each(|(x_i, y_i)| *x_i += y_i * s);
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
    let mut arr: [_; D] = unsafe { MaybeUninit::uninit().assume_init() };

    arr[0] = MaybeUninit::new(x);
    let mut acc = 1;
    while acc < D {
        arr[acc] = MaybeUninit::new(R::ZERO);
        acc += 1;
    }
    // If the code has reached this point every element of arr is correctly initialized.
    // Hence we are safe to reinterpret the array as [R; D].

    unsafe { HackyWorkAround::transpose(arr).assume_init() }
}

/// Given an element x from a 32 bit field F_P compute x/2.
#[inline]
pub const fn halve_u32<const P: u32>(x: u32) -> u32 {
    let shift = (P + 1) >> 1;
    let half = x >> 1;
    if x & 1 == 0 { half } else { half + shift }
}

/// Given an element x from a 64 bit field F_P compute x/2.
#[inline]
pub const fn halve_u64<const P: u64>(x: u64) -> u64 {
    let shift = (P + 1) >> 1;
    let half = x >> 1;
    if x & 1 == 0 { half } else { half + shift }
}

/// Reduce a slice of 32-bit field elements into a single element of a larger field.
///
/// Uses base-$2^{32}$ decomposition:
///
/// ```math
/// \begin{equation}
/// \text{reduce\_32}(vals) = \sum_{i=0}^{n-1} a_i \cdot 2^{32i}
/// \end{equation}
/// ```
///
/// Assumes `TF` has characteristic > $2^{64}$ to avoid overflow.
pub fn reduce_32<SF: PrimeField32, TF: PrimeField>(vals: &[SF]) -> TF {
    let base = TF::from_int(1u64 << 32);
    vals.iter().rev().fold(TF::ZERO, |acc, val| {
        acc * base + TF::from_int(val.as_canonical_u32())
    })
}

/// Split a large field element into `n` base-$2^{64}$ chunks in a 32-bit field.
///
/// Converts:
/// ```math
/// \begin{equation}
/// x = \sum_{i=0}^{n-1} d_i \cdot 2^{64i}
/// \end{equation}
/// ```
///
/// Pads with zeros if needed.
pub fn split_32<SF: PrimeField, TF: PrimeField32>(val: SF, n: usize) -> Vec<TF> {
    let mut result: Vec<TF> = val
        .as_canonical_biguint()
        .to_u64_digits()
        .iter()
        .take(n)
        .map(|d| TF::from_u64(*d))
        .collect();

    // Pad with zeros if needed
    result.resize_with(n, || TF::ZERO);
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
