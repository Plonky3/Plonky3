use alloc::vec;
use alloc::vec::Vec;
use core::iter::Sum;
use core::mem::MaybeUninit;
use core::ops::{Add, Mul};

use num_bigint::BigUint;
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
pub fn scale_slice_in_place_single_core<F: Field>(slice: &mut [F], s: F) {
    let (packed, sfx) = F::Packing::pack_slice_with_suffix_mut(slice);
    let packed_s: F::Packing = s.into();
    packed.iter_mut().for_each(|x| *x *= packed_s);
    sfx.iter_mut().for_each(|x| *x *= s);
}

/// Scales each element of the slice by `s` using packing and parallelization.
///
/// # Performance
/// For small slices, use [`scale_slice_in_place_single_core`].
/// Requires the `parallel` feature.
#[inline]
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
#[must_use]
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
#[must_use]
pub const fn halve_u32<const P: u32>(x: u32) -> u32 {
    let shift = (P + 1) >> 1;
    let half = x >> 1;
    if x & 1 == 0 { half } else { half + shift }
}

/// Given an element x from a 64 bit field F_P compute x/2.
#[inline]
#[must_use]
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
///
/// Equivalent to [`reduce_packed`] with `radix_bits = 32`.
#[must_use]
pub fn reduce_32<SF: PrimeField32, TF: PrimeField>(vals: &[SF]) -> TF {
    reduce_packed(vals, 32)
}

/// Horner-evaluate `vals` as base-$2^{radix\_bits}$ digits into `TF`, shifting each digit by `+1`.
///
/// This reserves zero as an out-of-band "no digit" value, so sequences of different lengths remain
/// distinct when packed into a fixed-width slot.
#[must_use]
pub fn reduce_packed_shifted<SF: PrimeField32, TF: PrimeField>(vals: &[SF], radix_bits: u32) -> TF {
    debug_assert!((radix_bits < 64) && ((SF::ORDER_U32 as u64) < (1u64 << radix_bits)));
    let base = TF::from_int(1u64 << radix_bits);
    vals.iter()
        .map(|val| TF::from_int(val.as_canonical_u32() as u64 + 1))
        .horner(base)
}

/// Bit length of `F::ORDER_U32 - 1`, i.e. the smallest `b` with `F::ORDER_U32 - 1 < 2^b`.
///
/// Used for tight base-$2^b$ absorb packing so canonical [`PrimeField32`] digits are always
/// valid base-$2^b$ digits (more limbs per [`PrimeField`] slot than radix $2^{32}$ when
/// `ORDER_U32 < 2^{32}`).
#[inline]
#[must_use]
pub const fn absorb_radix_bits<F: PrimeField32>() -> u32 {
    u32::BITS - (F::ORDER_U32 - 1).leading_zeros()
}

/// Horner-evaluate `vals` as base-$2^{radix\_bits}$ digits into `TF`.
///
/// Requires every canonical `SF` digit to be strictly less than `2^{radix\_bits}` (true when
/// `radix_bits ≥ absorb_radix_bits::<SF>()`).
#[must_use]
pub fn reduce_packed<SF: PrimeField32, TF: PrimeField>(vals: &[SF], radix_bits: u32) -> TF {
    debug_assert!((absorb_radix_bits::<SF>() <= radix_bits) && (radix_bits < 64));
    let base = TF::from_int(1u64 << radix_bits);
    vals.iter()
        .map(|val| TF::from_int(val.as_canonical_u32()))
        .horner(base)
}

/// Largest `b` such that every integer in `[0, 2^b)` maps injectively into `F` via `PrimeField32::from_int`.
///
/// Equivalently `b = floor(log2(p-1))` for prime `p = F::ORDER_U32`.
#[inline]
#[must_use]
pub const fn injective_pack_bits<F: PrimeField32>() -> u32 {
    (F::ORDER_U32 - 1).ilog2()
}

/// Maximum number of [`PrimeField32`] elements packable into [`PrimeField`] injectively via
/// [`reduce_packed`] with the given `radix_bits` (base-$2^{radix\_bits}$ digits bounded by
/// `F::ORDER_U32 - 1`).
///
/// Returns the largest `k` such that
/// `(F::ORDER_U32 - 1) · ∑_{i=0}^{k-1} (2^{radix\_bits})^i < PF::order()`.
#[must_use]
pub fn max_packed_injective_limbs<F: PrimeField32, PF: PrimeField>(radix_bits: u32) -> usize {
    max_packed_injective_limbs_with_max_digit::<PF>(radix_bits, F::ORDER_U32 - 1)
}

fn max_packed_injective_limbs_with_max_digit<PF: PrimeField>(
    radix_bits: u32,
    max_digit: u32,
) -> usize {
    debug_assert!((0 < radix_bits) && (radix_bits < 64));
    let max_digit = BigUint::from(max_digit);
    let base = BigUint::from(1u32) << (radix_bits as usize);
    let pf_order = PF::order();
    let mut k = 0usize;
    let mut max_val = BigUint::ZERO;
    let mut power = BigUint::from(1u32);
    loop {
        let new_max = &max_val + &max_digit * &power;
        if new_max >= pf_order {
            break k;
        }
        max_val = new_max;
        power *= &base;
        k += 1;
    }
}

/// Maximum number of shifted [`PrimeField32`] elements packable into [`PrimeField`] injectively
/// via [`reduce_packed_shifted`] with the given `radix_bits`.
///
/// Returns the largest `k` such that
/// `F::ORDER_U32 · ∑_{i=0}^{k-1} (2^{radix\_bits})^i < PF::order()`.
#[must_use]
pub fn max_shifted_packed_injective_limbs<F: PrimeField32, PF: PrimeField>(
    radix_bits: u32,
) -> usize {
    max_packed_injective_limbs_with_max_digit::<PF>(radix_bits, F::ORDER_U32)
}

/// Maximum limbs per [`PrimeField`] rate slot when absorbing with radix
/// $2^{\texttt{absorb\\_radix\\_bits::\<F\>()}}$ (see [`reduce_packed`]).
#[must_use]
pub fn max_absorb_injective_limbs<F: PrimeField32, PF: PrimeField>() -> usize {
    max_packed_injective_limbs::<F, PF>(absorb_radix_bits::<F>())
}

/// Maximum shifted limbs per [`PrimeField`] rate slot when absorbing with radix
/// $2^{\texttt{absorb\\_radix\\_bits::\<F\>()}}$ (see [`reduce_packed_shifted`]).
#[must_use]
pub fn max_shifted_absorb_injective_limbs<F: PrimeField32, PF: PrimeField>() -> usize {
    max_shifted_packed_injective_limbs::<F, PF>(absorb_radix_bits::<F>())
}

/// Returns true iff every integer in `[0, SF::order())` fits in `num_limbs` little-endian
/// base-`2^radix_bits` digits without truncation, i.e. `2^{num_limbs · radix_bits} ≥ SF::order()`.
#[must_use]
pub fn pf_packed_limbs_cover_order<SF: PrimeField>(num_limbs: usize, radix_bits: u32) -> bool {
    let Some(total_bits) = num_limbs.checked_mul(radix_bits as usize) else {
        return false;
    };
    (BigUint::from(1u32) << total_bits) >= SF::order()
}

/// Split `val` into `num_limbs` little-endian base-`2^radix_bits` limbs, each mapped into `TF`.
///
/// Each output limb is in `[0, 2^radix_bits)`. Pads with zero limbs if the value has fewer
/// non-zero digits than `num_limbs`.
///
/// **Parameter requirements**
///
/// - `radix_bits ≤ injective_pack_bits::<TF>()` so each limb maps injectively into `TF` via
///   `PrimeField32::from_int`. If `radix_bits` is too large, distinct limbs can collide after
///   reduction modulo `TF::ORDER`.
/// - For a **lossless** transcript binding of arbitrary `SF` values, also require
///   `pf_packed_limbs_cover_order::<SF>(num_limbs, radix_bits)`. Deliberately truncated
///   splits (e.g. challengers that use `floor` limb counts for squeeze) omit high bits by design
///   and do not satisfy that coverage check.
#[must_use]
pub fn split_pf_to_packed_limbs<SF: PrimeField, TF: PrimeField32>(
    val: SF,
    num_limbs: usize,
    radix_bits: u32,
) -> Vec<TF> {
    debug_assert!((0 < radix_bits) && (radix_bits < 64));
    debug_assert!(
        radix_bits <= injective_pack_bits::<TF>(),
        "radix_bits must be ≤ injective_pack_bits::<TF>() for injective limb embedding"
    );

    // Use a primitive u32 mask!
    let mask_u32: u32 = (1u32 << radix_bits) - 1;
    let mut rem = val.as_canonical_biguint();
    let mut out = vec![TF::ZERO; num_limbs];

    for item in out.iter_mut() {
        // Look at the lowest limb directly, no allocations
        let limb = rem.iter_u32_digits().next().unwrap_or(0) & mask_u32;
        *item = TF::from_int(limb);

        // In-place bitshift modifies the BigUint without allocating
        rem >>= radix_bits;
    }

    out
}

/// Number of `TF` limbs with statistical bias `< 1/|TF|` when decomposing a uniformly random
/// `PF` element in base `|TF|` (see [`split_pf_to_field_order_limbs`]).
///
/// Returns the largest `k` such that `TF::ORDER^{k+1} < PF::ORDER`. Each retained limb `c_i`
/// (`i < k`) has bias `≈ 1/⌊PF::ORDER / TF::ORDER^{i+2}⌋ < 1/TF::ORDER`.
///
/// Unlike the power-of-two radix variant ([`split_pf_to_packed_limbs`] with
/// `radix_bits = injective_pack_bits::<TF>()`), which confines each challenge to
/// `[0, 2^{radix_bits})` (≈ 50% of `TF`'s domain for BabyBear), this gives limbs that are
/// near-uniform over the **entire** `TF` domain.
///
/// # BabyBear concrete values
/// | PF | Good limbs |
/// |---|---|
/// | Goldilocks (64-bit) | 1 |
/// | BN254 (254-bit) | 7 |
#[must_use]
pub fn squeeze_field_order_num_limbs<PF: PrimeField, TF: PrimeField32>() -> usize {
    let p = BigUint::from(TF::ORDER_U32);
    let n = PF::order();
    let mut count = 0usize;
    let mut power = BigUint::from(1u32);
    while &power * &p < n {
        power *= &p;
        count += 1;
    }
    count.saturating_sub(1)
}

/// Split `val` into `num_limbs` little-endian base-`|TF|` limbs, each mapped into `TF`.
///
/// Decomposes `val` as `c0 + c1·p + c2·p² + …` (p = `TF::ORDER_U32`), returning
/// `[c0, c1, …, c_{num_limbs-1}]` with `0 ≤ ci < p`. Pads with `TF::ZERO` if `val` has
/// fewer significant digits than `num_limbs`.
///
/// Use [`squeeze_field_order_num_limbs`] to choose `num_limbs` such that each retained limb
/// is near-uniform over all of `TF` when `val` is uniformly random.
#[must_use]
pub fn split_pf_to_field_order_limbs<SF: PrimeField, TF: PrimeField32>(
    val: SF,
    num_limbs: usize,
) -> Vec<TF> {
    let p_u32 = TF::ORDER_U32;
    let mut rem = val.as_canonical_biguint();
    let mut out = Vec::with_capacity(num_limbs);

    for _ in 0..num_limbs {
        // Fast, primitive 32-bit modulo (no heap allocation!)
        let limb = (&rem % p_u32).to_u32_digits().first().copied().unwrap_or(0);
        out.push(TF::from_int(limb));

        // Fast, primitive in-place 32-bit division
        rem /= p_u32;
    }
    out
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
#[must_use]
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
#[must_use]
pub fn dot_product<S, LI, RI>(li: LI, ri: RI) -> S
where
    LI: Iterator,
    RI: Iterator,
    LI::Item: Mul<RI::Item>,
    S: Sum<<LI::Item as Mul<RI::Item>>::Output>,
{
    li.zip(ri).map(|(l, r)| l * r).sum()
}

/// Horner-style polynomial evaluation over a [`DoubleEndedIterator`].
///
/// The iterator yields coefficients in **ascending degree order**
/// `[c_0, c_1, …, c_{n-1}]`. Both methods walk the iterator back-to-front
/// via [`DoubleEndedIterator::rfold`], avoiding any allocation.
///
/// # Convention
///
/// Given an evaluation point `x` and accumulator `acc`,
/// [`HornerIter::horner_acc`] computes
///
/// ```text
/// acc · xⁿ + Σ_{i=0..n} c_i · xⁱ
///   = c_0 + x · (c_1 + x · (… + x · (c_{n-1} + x · acc)))
/// ```
///
/// [`HornerIter::horner`] is the same with `acc = Acc::default()`, i.e. the
/// polynomial evaluation `Σ_i c_i · xⁱ`.
///
/// For inputs in *descending* degree order, call `.rev()` on the iterator
/// first.
pub trait HornerIter: DoubleEndedIterator + Sized {
    /// Horner fold with an explicit accumulator. See the trait docs for the
    /// evaluation convention.
    #[inline]
    fn horner_acc<Acc, X>(self, acc: Acc, x: X) -> Acc
    where
        Acc: Mul<X, Output = Acc> + Add<Self::Item, Output = Acc>,
        X: Clone,
    {
        self.rfold(acc, |a, v| a * x.clone() + v)
    }

    /// Horner fold starting from `Acc::default()`. See the trait docs for the
    /// evaluation convention.
    #[inline]
    fn horner<Acc, X>(self, x: X) -> Acc
    where
        Acc: Default + Mul<X, Output = Acc> + Add<Self::Item, Output = Acc>,
        X: Clone,
    {
        self.horner_acc(Acc::default(), x)
    }
}

impl<I: DoubleEndedIterator> HornerIter for I {}
