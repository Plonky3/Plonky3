//! Equality tables built as one tensor product.
//!
//! The table of a point splits along its coordinates into two factors:
//!
//! ```text
//!     point = (high, low)
//!     eq(point, h * 2^l + j) = eq(high, h) * eq(low, j)
//! ```
//!
//! - The low factor is small, packed, and stays in L1.
//! - The high factor holds one scalar seed per output row.
//! - Every output entry is then one multiplication and one store, written exactly once.
//!
//! Doubling the table one variable at a time costs the same multiplications.
//! It rewrites the table once per variable, though, and the last passes stream from memory.

use alloc::vec::Vec;

use p3_field::{Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::split_eq::MUL_ACC_BYTES;

/// Variables held by the low factor of a tensor-built table.
///
/// Every output row rereads the whole low factor, so it must stay in L1:
///
/// ```text
///     2^10 extension elements * 16 to 24 bytes = 16 to 24 KiB
/// ```
pub(crate) const LOW_VARS: usize = 10;

/// Returns `scale * eq(point, x)` for every `x`, doubling one coordinate per pass.
///
/// Each coordinate `c` doubles the filled prefix, as a new high bit:
///
/// ```text
///     [ t ]  ->  [ t * (1 - c) | t * c ]
/// ```
///
/// The last coordinate enters first, so the first coordinate binds the most significant bit.
///
/// One multiplication per entry: the `1 - c` share is recovered by a subtraction.
///
/// The table grows straight into fresh capacity, so no slot is zeroed before it is written.
///
/// # Panics
///
/// Panics if `2^point.len()` overflows `usize`.
pub(crate) fn eq_doubled<F: Field, A: Algebra<F> + Copy>(point: &[F], scale: A) -> Vec<A> {
    let len = 1usize
        .checked_shl(point.len() as u32)
        .expect("Point length too large: 2^n overflows usize.");
    let mut table = Vec::with_capacity(len);
    let slots = &mut table.spare_capacity_mut()[..len];

    // Invariant: before pass i, exactly the first 2^i slots are written.
    slots[0].write(scale);
    for (i, &var) in point.iter().rev().enumerate() {
        let (lo, hi) = slots[..2 << i].split_at_mut(1 << i);
        for (lo, hi) in lo.iter_mut().zip(hi) {
            // SAFETY: `lo` lies in the first 2^i slots, which the invariant says are written.
            let lo = unsafe { lo.assume_init_mut() };
            let share = *lo * var;
            hi.write(share);
            *lo -= share;
        }
    }

    // SAFETY: after the last pass the invariant covers all 2^n slots.
    unsafe { table.set_len(len) };
    table
}

/// Builds the packed table `scale * eq(point, .)` on one core, by doubling.
///
/// The last `log2(W)` coordinates fill the lanes of one seed.
/// The rest double that seed across the packed entries.
///
/// # Panics
///
/// Panics if the point has fewer than `log2(W)` coordinates.
pub(crate) fn packed_eq_serial<F, EF>(point: &[EF], scale: EF) -> Vec<EF::ExtensionPacking>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let log_width = log2_strict_usize(F::Packing::WIDTH);
    assert!(point.len() >= log_width);
    let (outer, lanes) = point.split_at(point.len() - log_width);

    // One packed seed holds the table of the lane coordinates.
    let seed = EF::ExtensionPacking::from_ext_slice(&eq_doubled(lanes, scale));

    // The outer coordinates double that seed across the packed entries.
    eq_doubled(outer, seed)
}

/// Splits a point into its high and low factors.
///
/// The low factor keeps at least `min_low` coordinates, and at most `max(LOW_VARS, min_low)`.
#[inline]
pub(crate) fn split_low<EF>(point: &[EF], min_low: usize) -> (&[EF], &[EF]) {
    let low_vars = point.len().min(LOW_VARS.max(min_low));
    point.split_at(point.len() - low_vars)
}

/// Bytes one tensor row is charged, for a row of `packed_len` packed products.
///
/// A packed product multiplies all `W` lanes in about the time one extension
/// multiply-accumulate is charged, measured at 10 ns for a degree-4 prime extension on
/// AVX-512 and 4.6 ns for `GF(2^192)` on AVX2.
///
/// Unpacking the lanes as they are stored costs about as much again, so it doubles the charge.
///
/// Pricing each product by its packed width instead overcharges by `W`.
/// That splits tables worth a few microseconds, and the dispatch then costs more than the work.
const fn row_bytes<EF>(packed_len: usize, passes: usize) -> usize {
    passes * MUL_ACC_BYTES * size_of::<EF>() * packed_len
}

/// Builds the packed tensor product, one row per high seed:
///
/// ```text
///     table[h * |low| + j] = low[j] * high[h]
/// ```
///
/// Rows are independent, so the loop splits across threads through the cost model.
///
/// Every slot is written exactly once, straight into fresh capacity.
/// A zeroed buffer would cost a memset first whenever the allocator recycles memory.
///
/// # Panics
///
/// Panics if `low` is empty.
pub(crate) fn tensor_packed<F, EF>(
    high: &[EF],
    low: &[EF::ExtensionPacking],
) -> Vec<EF::ExtensionPacking>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let len = high.len() * low.len();
    let mut table = Vec::with_capacity(len);
    // One item multiplies and stores one whole row.
    table.spare_capacity_mut()[..len]
        .par_chunks_exact_mut(low.len())
        .zip(high.par_iter())
        .with_min_task_bytes(row_bytes::<EF>(low.len(), 1))
        .for_each(|(row, &seed)| {
            for (slot, &weight) in row.iter_mut().zip(low) {
                slot.write(weight * seed);
            }
        });
    // SAFETY: the rows tile the first `len` slots, and each row writes all `|low|` of its own.
    unsafe { table.set_len(len) };
    table
}

/// Builds the scalar tensor product, unpacking the lanes of each packed product as it stores:
///
/// ```text
///     table[(h * |low| + j) * W + lane] = (low[j] * high[h]).lane
/// ```
///
/// The multiplication runs packed, and only the store transposes.
///
/// # Panics
///
/// Panics if `low` is empty.
pub(crate) fn tensor_unpacked<F, EF>(high: &[EF], low: &[EF::ExtensionPacking]) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let width = F::Packing::WIDTH;
    let len = high.len() * low.len() * width;
    let mut table = Vec::with_capacity(len);
    // One item multiplies and stores one whole row.
    table.spare_capacity_mut()[..len]
        .par_chunks_exact_mut(low.len() * width)
        .zip(high.par_iter())
        .with_min_task_bytes(row_bytes::<EF>(low.len(), 2))
        .for_each(|(row, &seed)| {
            for (lanes, &weight) in row.chunks_exact_mut(width).zip(low) {
                let product = weight * seed;
                for (lane, slot) in lanes.iter_mut().enumerate() {
                    slot.write(product.extract(lane));
                }
            }
        });
    // SAFETY: the rows tile the first `len` slots, and each row writes all `|low| * W` of its own.
    unsafe { table.set_len(len) };
    table
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::point::Point;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type EP = <EF as ExtensionField<F>>::ExtensionPacking;

    proptest! {
        #[test]
        fn tensor_rows_match_the_doubled_table(
            high_vars in 0usize..=4,
            low_vars in 0usize..=6,
            seed in any::<u64>(),
        ) {
            // Invariant: both tensor kernels reproduce the doubled table of the joined point.
            let log_width = log2_strict_usize(<F as Field>::Packing::WIDTH);
            let low_vars = low_vars + log_width;
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, high_vars + low_vars);
            let scale: EF = rand::RngExt::random(&mut rng);

            // Reference: the whole table, doubled on one core.
            let expected = eq_doubled(point.as_slice(), scale);

            // Factors: scaled high seeds, and an unscaled packed low table.
            let (high_point, low_point) = point.as_slice().split_at(high_vars);
            let high = eq_doubled(high_point, scale);
            let low = packed_eq_serial::<F, EF>(low_point, EF::ONE);

            // Scalar rows must equal the reference entry for entry.
            let unpacked = tensor_unpacked::<F, EF>(&high, &low);
            prop_assert_eq!(&unpacked, &expected);

            // Packed rows must equal the reference once their lanes are read back.
            let packed = tensor_packed::<F, EF>(&high, &low);
            let lanes: Vec<EF> = <EP as PackedFieldExtension<F, EF>>::to_ext_iter(packed).collect();
            prop_assert_eq!(&lanes, &expected);
        }
    }
}
