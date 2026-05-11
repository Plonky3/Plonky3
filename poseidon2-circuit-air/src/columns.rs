//! Column definitions for the Poseidon2 circuit AIR.

use alloc::vec::Vec;
use core::borrow::{Borrow, BorrowMut};
use core::mem::size_of;

use p3_circuit::ops::{KoalaBearD1Width16, Poseidon2Params};

/// Number of extension-field limbs for Poseidon2 input and output.
///
/// Each limb is one extension-field element.
///
/// It is stored as a group of base-field columns whose count equals the
/// extension degree.
///
/// The Poseidon2 state has this many limbs on both the input and output
/// sides.
pub const POSEIDON2_LIMBS: usize = 4;

/// Number of output limbs exposed publicly via cross-table lookup.
///
/// Only the first two output limbs are sent to the Witness table.
///
/// The remaining output limbs are consumed internally by the chaining
/// constraints.
pub const POSEIDON2_PUBLIC_OUTPUT_LIMBS: usize = 2;

/// Value columns for one row of the Poseidon2 circuit table.
///
/// The type parameter carries the inner permutation columns.
///
/// It holds the full input/output state plus all intermediate round
/// registers.
///
/// Two extra circuit-specific columns follow the permutation block.
///
/// # Memory Layout
///
/// ```text
///     [ ── permutation columns ── | mmcs_bit | mmcs_index_sum ]
/// ```
#[repr(C)]
pub struct Poseidon2CircuitCols<T, P> {
    /// Inner Poseidon2 permutation columns.
    ///
    /// Holds input limbs, output limbs, and all intermediate round state.
    ///
    /// The exact width depends on the permutation parameters.
    pub poseidon2: P,

    /// Merkle direction bit.
    ///
    /// Zero means the current digest is the left child.
    ///
    /// One means the current digest is the right child.
    ///
    /// Only meaningful on rows where the Merkle-path flag is set.
    ///
    /// Constrained to be boolean on every row regardless.
    ///
    /// This is a value column, not preprocessed, because the prover
    /// chooses it at runtime based on the Merkle proof path.
    pub mmcs_bit: T,

    /// Running MMCS query-index accumulator.
    ///
    /// Across a chain of Merkle rows this accumulates the binary
    /// decomposition of the leaf index.
    ///
    /// The recurrence is:
    ///
    /// ```text
    ///     next_sum = current_sum × 2 + next_bit
    /// ```
    ///
    /// The constraint is only active when the row is not a chain start
    /// and the Merkle-path flag is set.
    ///
    /// On chain-start rows the prover may write any value.
    pub mmcs_index_sum: T,
}

/// Return the total number of columns in a single row.
///
/// Relies on the `size_of` trick: instantiate the struct with `u8` so
/// that every field occupies exactly one byte.
///
/// The struct size in bytes then equals the column count.
pub const fn num_cols<P>() -> usize {
    size_of::<Poseidon2CircuitCols<u8, P>>()
}

impl<T, P> Borrow<Poseidon2CircuitCols<T, P>> for [T] {
    fn borrow(&self) -> &Poseidon2CircuitCols<T, P> {
        let (prefix, shorts, suffix) = unsafe { self.align_to::<Poseidon2CircuitCols<T, P>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T, P> BorrowMut<Poseidon2CircuitCols<T, P>> for [T] {
    fn borrow_mut(&mut self) -> &mut Poseidon2CircuitCols<T, P> {
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<Poseidon2CircuitCols<T, P>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

/// Preprocessed columns for a single Poseidon2 **input** limb.
///
/// Each input limb carries its own copy of these four columns.
///
/// They encode three things:
///
/// 1. Which witness slot the limb reads from.
///
/// 2. Whether the limb participates in a cross-table lookup.
///
/// 3. Whether the limb is chained from the previous row in sponge mode
///    or in Merkle mode.
///
/// The two chain selectors are mutually exclusive.
///
/// They are precomputed to keep constraint degree at three.
///
/// ```text
///     sponge_chain  = !new_start && !merkle_path && !in_ctl
///     merkle_chain  = !new_start &&  merkle_path && !in_ctl
/// ```
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Poseidon2PrepInputLimb<T> {
    /// Witness index for this input limb.
    ///
    /// Used in the cross-table lookup.
    ///
    /// Scaled by the extension degree so that the key directly indexes
    /// into the flattened witness table.
    pub idx: T,

    /// Cross-table lookup enable flag.
    ///
    /// When set, this limb is looked up in the Witness table.
    ///
    /// When clear, the limb's value comes from chaining or is unconstrained.
    pub in_ctl: T,

    /// Sponge-mode chain selector.
    ///
    /// When set, the AIR enforces that the next row's input equals
    /// the current row's output for this limb.
    ///
    /// This is standard sponge chaining across all base-field elements.
    pub normal_chain_sel: T,

    /// Merkle-mode chain selector.
    ///
    /// When set, the AIR enforces directional chaining gated by the
    /// direction bit.
    ///
    /// If the direction bit is zero (left child), the output chains to
    /// the first half of the next input.
    ///
    /// If the direction bit is one (right child), the output chains to
    /// the second half of the next input.
    ///
    /// Only the first two limbs carry a meaningful Merkle selector.
    ///
    /// The last two limbs reuse the first two selectors, gated on the
    /// opposite direction.
    pub merkle_chain_sel: T,
}

/// Preprocessed columns for a single Poseidon2 **output** limb.
///
/// Only the first two output limbs are exposed via cross-table lookup.
///
/// The remaining outputs are consumed internally by the chaining
/// constraints.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Poseidon2PrepOutputLimb<T> {
    /// Witness index for this output limb.
    ///
    /// Scaled by the extension degree, same convention as input limbs.
    pub idx: T,

    /// Cross-table lookup enable flag.
    ///
    /// When set, this limb is received from the Witness table.
    ///
    /// This proves the output matches a committed value.
    pub out_ctl: T,
}

/// Number of preprocessed columns for one Poseidon2 row.
///
/// `input_limbs` is the number of logical input limbs (`WIDTH_EXT` in the
/// AIR). Each [`Poseidon2PrepInputLimb`] occupies four scalar columns.
/// `output_limbs` is the number of rate output limbs exposed via CTL
/// (`RATE_EXT`). Each [`Poseidon2PrepOutputLimb`] occupies two columns.
/// The row ends with four single-column flags.
///
/// For D=1 width-16 / rate-8 Poseidon2, use [`poseidon2_preprocessed_row_width_for_air`] instead.
#[inline]
pub const fn poseidon2_preprocessed_row_width(input_limbs: usize, output_limbs: usize) -> usize {
    input_limbs * size_of::<Poseidon2PrepInputLimb<u8>>()
        + output_limbs * size_of::<Poseidon2PrepOutputLimb<u8>>()
        + 4
}

/// `true` when Poseidon2 uses the compact D=1 preprocessed layout.
#[inline]
pub const fn poseidon2_uses_compact_d1_preprocessed(
    poseidon_d: usize,
    width_ext: usize,
    rate_ext: usize,
) -> bool {
    poseidon_d == 1
        && width_ext == KoalaBearD1Width16::WIDTH_EXT
        && rate_ext == KoalaBearD1Width16::RATE_EXT
}

/// Scalar columns before input indices in the compact D=1 layout: `rate_ext` per-limb `in_ctl`,
/// unused `cap_in_ctl` (always zero; kept for fixed offset), `cap_chain_enable`, then `rate_ext`
/// sponge-chain helpers `(1 − new_start) * (1 − merkle_path) * (1 − in_ctl_i)` and `rate_ext`
/// Merkle-chain helpers `(1 − new_start) * merkle_path * (1 − in_ctl_i)` so transition gates stay degree-3.
#[inline]
pub const fn poseidon2_d1_compact_preprocessed_header_cols(rate_ext: usize) -> usize {
    rate_ext + 2 + rate_ext + rate_ext
}

/// Preprocessed row width for a [`crate::Poseidon2CircuitAir`] with the given const parameters.
#[inline]
pub const fn poseidon2_preprocessed_row_width_for_air(
    poseidon_d: usize,
    width_ext: usize,
    rate_ext: usize,
) -> usize {
    if poseidon2_uses_compact_d1_preprocessed(poseidon_d, width_ext, rate_ext) {
        // Compact D=1: per-rate-limb in_ctl + cap_in_ctl (zero) + cap_chain + input idx + output idx +
        // per-limb out_ctl (out_ctl stays per limb for prover multiplicity pass).
        poseidon2_d1_compact_preprocessed_header_cols(rate_ext)
            + width_ext
            + rate_ext
            + rate_ext
            + 4
    } else {
        poseidon2_preprocessed_row_width(width_ext, rate_ext)
    }
}

/// Full preprocessed row for the Poseidon2 circuit table.
///
/// One row per Poseidon2 permutation invocation.
///
/// `INPUT_LIMBS` matches the AIR's `WIDTH_EXT` (state width in logical
/// limbs). `OUTPUT_LIMBS` matches `RATE_EXT` (rate in logical limbs).
///
/// # Padding
///
/// When padded to a power-of-two height, the **first** padding row sets
/// the chain-start flag to one.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Poseidon2PreprocessedRow<const INPUT_LIMBS: usize, const OUTPUT_LIMBS: usize, T> {
    /// Per-limb preprocessed input columns (one logical limb = `D` bases in the trace).
    pub input_limbs: [Poseidon2PrepInputLimb<T>; INPUT_LIMBS],

    /// Per-limb preprocessed output columns for rate outputs under CTL.
    pub output_limbs: [Poseidon2PrepOutputLimb<T>; OUTPUT_LIMBS],

    /// Witness index for the MMCS accumulator column.
    ///
    /// Used in the cross-table lookup that exposes the accumulator
    /// to the Witness table at the end of a Merkle chain.
    pub mmcs_index_sum_ctl_idx: T,

    /// Precomputed product of the MMCS-enabled flag and the Merkle-path
    /// flag.
    ///
    /// This is the row-local part of the multiplicity expression for the
    /// accumulator lookup.
    ///
    /// The full multiplicity also involves the next row's chain-start
    /// flag, so the lookup fires on the last Merkle row before a chain
    /// boundary.
    ///
    /// Precomputing this product keeps the overall multiplicity at
    /// degree two.
    pub mmcs_merkle_flag: T,

    /// Chain boundary flag.
    ///
    /// Set on the first row of a new sponge or Merkle chain.
    ///
    /// When set, all chaining constraints and the MMCS accumulator
    /// update are disabled.
    pub new_start: T,

    /// Merkle-path flag.
    ///
    /// Set when this row is a Merkle-path step with directional hashing.
    ///
    /// Clear for standard sponge rows.
    pub merkle_path: T,
}

impl<const INPUT_LIMBS: usize, const OUTPUT_LIMBS: usize, T: Copy + Default> Default
    for Poseidon2PreprocessedRow<INPUT_LIMBS, OUTPUT_LIMBS, T>
{
    fn default() -> Self {
        Self {
            input_limbs: [Poseidon2PrepInputLimb::default(); INPUT_LIMBS],
            output_limbs: [Poseidon2PrepOutputLimb::default(); OUTPUT_LIMBS],
            mmcs_index_sum_ctl_idx: T::default(),
            mmcs_merkle_flag: T::default(),
            new_start: T::default(),
            merkle_path: T::default(),
        }
    }
}

impl<const INPUT_LIMBS: usize, const OUTPUT_LIMBS: usize, T: Copy>
    Poseidon2PreprocessedRow<INPUT_LIMBS, OUTPUT_LIMBS, T>
{
    /// Flatten this row into a buffer, preserving the field order.
    ///
    /// Uses a raw pointer cast instead of pushing fields one by one.
    ///
    /// This is automatically correct for any field ordering because
    /// `#[repr(C)]` guarantees the in-memory layout matches the
    /// declaration order.
    ///
    /// A manual push sequence would need to be kept in sync with the
    /// struct definition. The pointer cast avoids that fragility.
    pub fn write_into(self, buf: &mut Vec<T>) {
        // Compute the number of elements in the struct.
        //
        // For single-byte types this equals the struct size directly.
        // For larger field types we divide out the element size.
        let num_elements = size_of::<Self>() / size_of::<T>();

        // SAFETY: the struct is `#[repr(C)]` with `T: Copy` and all fields
        // are plain `T` values. No padding exists between same-typed fields.
        // The resulting slice covers exactly `num_elements` contiguous items.
        let ptr = &self as *const Self as *const T;
        let slice = unsafe { core::slice::from_raw_parts(ptr, num_elements) };
        buf.extend_from_slice(slice);
    }
}

impl<const INPUT_LIMBS: usize, const OUTPUT_LIMBS: usize, T>
    Borrow<Poseidon2PreprocessedRow<INPUT_LIMBS, OUTPUT_LIMBS, T>> for [T]
{
    fn borrow(&self) -> &Poseidon2PreprocessedRow<INPUT_LIMBS, OUTPUT_LIMBS, T> {
        debug_assert_eq!(
            self.len(),
            poseidon2_preprocessed_row_width(INPUT_LIMBS, OUTPUT_LIMBS)
        );
        let (prefix, rows, suffix) =
            unsafe { self.align_to::<Poseidon2PreprocessedRow<INPUT_LIMBS, OUTPUT_LIMBS, T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(rows.len(), 1);
        &rows[0]
    }
}

impl<const INPUT_LIMBS: usize, const OUTPUT_LIMBS: usize, T>
    BorrowMut<Poseidon2PreprocessedRow<INPUT_LIMBS, OUTPUT_LIMBS, T>> for [T]
{
    fn borrow_mut(&mut self) -> &mut Poseidon2PreprocessedRow<INPUT_LIMBS, OUTPUT_LIMBS, T> {
        debug_assert_eq!(
            self.len(),
            poseidon2_preprocessed_row_width(INPUT_LIMBS, OUTPUT_LIMBS)
        );
        let (prefix, rows, suffix) = unsafe {
            self.align_to_mut::<Poseidon2PreprocessedRow<INPUT_LIMBS, OUTPUT_LIMBS, T>>()
        };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(rows.len(), 1);
        &mut rows[0]
    }
}
