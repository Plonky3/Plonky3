use alloc::vec::Vec;
use core::iter::Sum;
use core::mem::MaybeUninit;
use core::ops::Mul;

use p3_maybe_rayon::prelude::*;

use crate::field::Field;
use crate::{PackedValue, PrimeCharacteristicRing, PrimeField, PrimeField32};

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

/// Scales each element of the slice by `s` using packing.
///
/// # Performance
/// For large slices, use [`par_scale_slice_in_place`].
pub fn scale_slice_in_place<F: Field>(slice: &mut [F], s: F) {
    let (packed, sfx) = F::Packing::pack_slice_with_suffix_mut(slice);
    let packed_s: F::Packing = s.into();
    packed.iter_mut().for_each(|x| *x *= packed_s);
    sfx.iter_mut().for_each(|x| *x *= s);
}

/// Scales each element of the slice by `s` using packing and parallelization.
///
/// # Performance
/// For small slices, use [`scale_slice_in_place`].
/// Requires the `parallel` feature.
pub fn par_scale_slice_in_place<F: Field>(slice: &mut [F], s: F) {
    let (packed, sfx) = F::Packing::pack_slice_with_suffix_mut(slice);
    let packed_s: F::Packing = s.into();
    packed.par_iter_mut().for_each(|x| *x *= packed_s);
    sfx.iter_mut().for_each(|x| *x *= s);
}

/// Adds `other`, scaled by `s`, to the mutable `slice` using packing, or `slice += other * s`.
///
/// # Performance
/// For large slices, use [`par_add_scaled_slice_in_place`].
pub fn add_scaled_slice_in_place<F: Field>(slice: &mut [F], other: &[F], s: F) {
    debug_assert_eq!(slice.len(), other.len(), "slices must have equal length");
    let (slice_packed, slice_sfx) = F::Packing::pack_slice_with_suffix_mut(slice);
    let (other_packed, other_sfx) = F::Packing::pack_slice_with_suffix(other);
    let packed_s: F::Packing = s.into();
    slice_packed
        .iter_mut()
        .zip(other_packed)
        .for_each(|(x, y)| *x += *y * packed_s);
    slice_sfx
        .iter_mut()
        .zip(other_sfx)
        .for_each(|(x, y)| *x += *y * s);
}

/// Adds `other`, scaled by `s`, to the mutable `slice` using packing, or `slice += other * s`.
///
/// # Performance
/// For small slices, use [`add_scaled_slice_in_place`].
/// Requires the `parallel` feature.
pub fn par_add_scaled_slice_in_place<F: Field>(slice: &mut [F], other: &[F], s: F) {
    debug_assert_eq!(slice.len(), other.len(), "slices must have equal length");
    let (slice_packed, slice_sfx) = F::Packing::pack_slice_with_suffix_mut(slice);
    let (other_packed, other_sfx) = F::Packing::pack_slice_with_suffix(other);
    let packed_s: F::Packing = s.into();
    slice_packed
        .par_iter_mut()
        .zip(other_packed.par_iter())
        .for_each(|(x, y)| *x += *y * packed_s);
    slice_sfx
        .iter_mut()
        .zip(other_sfx)
        .for_each(|(x, y)| *x += *y * s);
}

/// Extend a ring `R` element `x` to an array of length `D`
/// by filling zeros.
#[inline]
pub const fn field_to_array<R: PrimeCharacteristicRing, const D: usize>(x: R) -> [R; D] {
    let mut arr = [const { MaybeUninit::uninit() }; D];
    arr[0] = MaybeUninit::new(x);
    let mut i = 1;
    while i < D {
        arr[i] = MaybeUninit::new(R::ZERO);
        i += 1;
    }
    unsafe { core::mem::transmute_copy::<_, [R; D]>(&arr) }
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
///     \text{reduce\_32}(vals) = \sum_{i=0}^{n-1} a_i \cdot 2^{32i}
/// \end{equation}
/// ```
pub fn reduce_32<SF: PrimeField32, TF: PrimeField>(vals: &[SF]) -> TF {
    // If the characteristic of TF is > 2^64, from_int and from_canonical_unchecked act identically
    let base = TF::from_int(1u64 << 32);
    vals.iter().rev().fold(TF::ZERO, |acc, val| {
        acc * base + TF::from_int(val.as_canonical_u32())
    })
}

/// Split a large field element into `n` base-$2^{64}$ chunks and map each into a 32-bit field.
///
/// Converts:
/// ```math
/// \begin{equation}
///     x = \sum_{i=0}^{n-1} d_i \cdot 2^{64i}
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
