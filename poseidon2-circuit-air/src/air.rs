use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::borrow::Borrow;
use core::iter;
use core::mem::MaybeUninit;

use p3_air::{Air, AirBuilder, AirLayout, BaseAir, BaseLeaf, WindowAccess};
use p3_circuit::ops::Poseidon2CircuitRow;
use p3_field::{Dup, Field, PrimeCharacteristicRing, PrimeField};
use p3_lookup::LookupAir;
use p3_lookup::lookup_traits::{Direction, Kind, Lookup};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_poseidon2::GenericPoseidon2LinearLayers;
use p3_poseidon2_air::{Poseidon2Air, Poseidon2Cols, RoundConstants, generate_trace_rows_for_perm};
use p3_uni_stark::{SubAirBuilder, SymbolicAirBuilder, SymbolicExpression, SymbolicVariable};
use tracing::instrument;

use crate::columns::{
    Poseidon2PrepInputLimb, Poseidon2PrepOutputLimb, Poseidon2PreprocessedRow,
    poseidon2_d1_compact_preprocessed_header_cols, poseidon2_preprocessed_row_width_for_air,
    poseidon2_uses_compact_d1_preprocessed,
};
use crate::{Poseidon2CircuitCols, num_cols};

/// Poseidon2 circuit AIR for recursive proof composition.
///
/// Wraps the upstream permutation AIR and adds four groups of constraints:
///
/// - Sponge chaining.
/// - Merkle-path chaining.
/// - MMCS leaf-index accumulator.
/// - Cross-table lookup interactions.
///
/// # Const Generic Parameters
///
/// - **D** — extension degree.
///   Number of base-field elements per extension-field element.
///
/// - **WIDTH** — state width in base-field elements.
///   Equals the rate plus capacity, counted in base-field elements.
///
/// - **WIDTH_EXT** — state width in extension-field elements.
///   Must satisfy `WIDTH_EXT × D = WIDTH`.
///
/// - **RATE_EXT / CAPACITY_EXT** — rate and capacity in extension elements.
///   Their sum must equal WIDTH_EXT.
///
/// - **SBOX_DEGREE** — algebraic degree of the S-box polynomial.
///   For example, 7 for BabyBear, 3 for KoalaBear.
///
/// - **SBOX_REGISTERS** — number of intermediate registers.
///   Used to decompose the high-degree S-box into lower-degree steps.
///
/// - **HALF_FULL_ROUNDS** — full rounds per half.
///   Applied at the beginning and again at the end of the permutation.
///
/// - **PARTIAL_ROUNDS** — number of partial rounds.
///   The S-box is applied to only one state element in each partial round.
///
/// # Invariants
///
/// Checked at compile time during construction:
///
/// ```text
///     WIDTH_EXT × D = WIDTH
///     RATE_EXT + CAPACITY_EXT = WIDTH_EXT
/// ```
#[derive(Debug)]
pub struct Poseidon2CircuitAir<
    F: PrimeCharacteristicRing,
    LinearLayers,
    const D: usize,
    const WIDTH: usize,
    const WIDTH_EXT: usize,
    const RATE_EXT: usize,
    const CAPACITY_EXT: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const WITNESS_EXT_D: usize,
> {
    /// The inner permutation AIR.
    ///
    /// Stores the round constants.
    ///
    /// Enforces the core constraint:
    /// - The output state must equal the Poseidon2 permutation of the input state.
    ///
    /// All circuit-level constraints (chaining, accumulator, cross-table
    /// lookups) are layered on top by this crate.
    p3_poseidon2: Poseidon2Air<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,

    /// Number of lookup columns registered so far.
    ///
    /// Each cross-table interaction adds one permutation argument column.
    ///
    /// There is:
    /// - one interaction per input limb (or per packed group when D=1 and the bus width divides
    ///   the logical width),
    /// - likewise for rate output limbs,
    /// - one interaction for the MMCS accumulator.
    pub(crate) num_lookup_cols: usize,

    /// Flat preprocessed trace data in row-major order.
    ///
    /// Only needed by the prover.
    ///
    /// The verifier works with the committed digest instead, so this
    /// vector may be empty for verification-only instances.
    preprocessed: Vec<F>,

    /// Minimum trace height for FRI compatibility.
    ///
    /// Some FRI configurations require a minimum domain size.
    ///
    /// The actual height is the maximum of the natural row count (rounded
    /// up to a power of two) and this value.
    min_height: usize,
}

impl<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const D: usize,
    const WIDTH: usize,
    const WIDTH_EXT: usize,
    const RATE_EXT: usize,
    const CAPACITY_EXT: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const WITNESS_EXT_D: usize,
> Clone
    for Poseidon2CircuitAir<
        F,
        LinearLayers,
        D,
        WIDTH,
        WIDTH_EXT,
        RATE_EXT,
        CAPACITY_EXT,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        WITNESS_EXT_D,
    >
{
    fn clone(&self) -> Self {
        Self {
            p3_poseidon2: self.p3_poseidon2.clone(),
            num_lookup_cols: self.num_lookup_cols,
            preprocessed: self.preprocessed.clone(),
            min_height: self.min_height,
        }
    }
}

impl<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const D: usize,
    const WIDTH: usize,
    const WIDTH_EXT: usize,
    const RATE_EXT: usize,
    const CAPACITY_EXT: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const WITNESS_EXT_D: usize,
>
    Poseidon2CircuitAir<
        F,
        LinearLayers,
        D,
        WIDTH,
        WIDTH_EXT,
        RATE_EXT,
        CAPACITY_EXT,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        WITNESS_EXT_D,
    >
{
    /// Create a new AIR with the given round constants.
    ///
    /// The preprocessed trace starts empty.
    ///
    /// You can supply it later via the preprocessed constructor variant,
    /// or by building it from circuit rows with the extraction helper.
    ///
    /// Two compile-time assertions fire if the generic invariants are violated:
    /// - The rate plus capacity must equal the extension width,
    /// - The extension width times the degree must equal the state width.
    pub const fn new(
        constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    ) -> Self {
        const {
            assert!(CAPACITY_EXT + RATE_EXT == WIDTH_EXT);
            assert!(WIDTH_EXT * D == WIDTH);
            assert!(WITNESS_EXT_D >= D);
        }

        Self {
            p3_poseidon2: Poseidon2Air::new(constants),
            num_lookup_cols: 0,
            preprocessed: Vec::new(),
            min_height: 1,
        }
    }

    /// Set the minimum trace height.
    ///
    /// The value is rounded up to a power of two.
    ///
    /// Use this when FRI requires a domain larger than the natural number
    /// of permutation rows.
    pub fn with_min_height(mut self, min_height: usize) -> Self {
        self.min_height = min_height.next_power_of_two().max(1);
        self
    }

    /// Create a new AIR with pre-populated preprocessed trace data.
    ///
    /// The preprocessed vector must be flat and row-major.
    ///
    /// Its length must be a multiple of the preprocessed width.
    ///
    /// For verification-only instances an empty vector is fine — the
    /// verifier only needs the committed digest.
    ///
    /// The same compile-time invariant checks apply as for the basic
    /// constructor.
    pub const fn new_with_preprocessed(
        constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
        preprocessed: Vec<F>,
    ) -> Self {
        const {
            assert!(CAPACITY_EXT + RATE_EXT == WIDTH_EXT);
            assert!(WIDTH_EXT * D == WIDTH);
            assert!(WITNESS_EXT_D >= D);
        }

        Self {
            p3_poseidon2: Poseidon2Air::new(constants),
            num_lookup_cols: 0,
            preprocessed,
            min_height: 1,
        }
    }

    /// Return the number of preprocessed columns per row.
    pub const fn preprocessed_width() -> usize {
        poseidon2_preprocessed_row_width_for_air(D, WIDTH_EXT, RATE_EXT)
    }

    /// Generate the execution trace matrix from a sequence of circuit rows.
    ///
    /// # Two-Pass Strategy
    ///
    /// ```text
    ///     Pass 1 (sequential)
    ///         Write the direction bit, the MMCS accumulator, and the
    ///         Poseidon2 input state into uninitialized trace memory.
    ///
    ///     Pass 2 (parallel)
    ///         Read the inputs back.
    ///         Compute the full Poseidon2 permutation for every row.
    /// ```
    ///
    /// Pass 1 is sequential because the MMCS accumulator depends on the previous row.
    ///
    /// Pass 2 is parallel because each permutation is independent.
    ///
    /// # Panics
    ///
    /// - If the number of rows is not a power of two.
    /// - If any row's input state has the wrong number of elements.
    #[instrument(skip_all, name = "Poseidon2CircuitAir::build_trace")]
    pub fn generate_trace_rows(
        &self,
        sponge_ops: &[Poseidon2CircuitRow<F>],
        constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F> {
        let n = sponge_ops.len();
        assert!(
            n.is_power_of_two(),
            "Callers expected to pad inputs to a power of two"
        );

        // Each row has two segments:
        //
        //     [ --- permutation columns --- | direction bit | accumulator ]
        //
        // The permutation segment holds the full Poseidon2 state.
        //
        // That includes the 16 input elements, all intermediate values
        // produced during the rounds, and the 16 output elements.
        //
        // After the permutation block come two circuit-specific columns.
        //
        // The direction bit says whether the current node is a left or
        // right child in a Merkle tree (only meaningful in Merkle mode).
        //
        // The accumulator reconstructs the Merkle leaf index one bit at
        // a time as the circuit walks up the authentication path.

        let p2_ncols = p3_poseidon2_air::num_cols::<
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >();
        let ncols = self.width();
        let circuit_ncols = ncols - p2_ncols;

        // Allocate the final trace as uninitialized memory.
        //
        // We use uninitialized memory because both passes will write
        // every element before it is read.
        //
        // The extra capacity bits enlarge only the permutation segment.
        //
        // The circuit columns are always two wide (direction bit and accumulator).
        let mut trace_vec: Vec<F> =
            Vec::with_capacity(n * ((p2_ncols << extra_capacity_bits) + circuit_ncols));
        let trace_slice = trace_vec.spare_capacity_mut();

        // Pass 1: Sequential
        //
        // This pass must run row by row because each row's accumulator
        // depends on the previous row's value.
        //
        // For each row we write three things into the uninitialized memory:
        //
        //   1. The Poseidon2 input state — the 16 field elements that will
        //      be fed into the permutation. These come directly from the
        //      operation struct provided by the caller.
        //
        //   2. The direction bit — zero or one, indicating left or right
        //      child in a Merkle tree.
        //
        //   3. The MMCS accumulator — a running value that reconstructs
        //      the Merkle leaf index from the direction bits.
        //
        // The accumulator follows a simple recurrence:
        //
        //     next_sum = current_sum × 2 + next_bit
        //
        // This is just binary-to-integer conversion built up one bit at
        // a time. For example, leaf index 5 (binary 101) is reconstructed
        // as: 1 → 1×2+0=2 → 2×2+1=5.
        //
        // On chain boundaries or non-Merkle rows the accumulator resets
        // to whatever value the operation struct carries.

        // Tracks the accumulator value from the previous row.
        //
        // Starts at zero. Updated on each iteration.
        let mut prev_mmcs_index_sum = F::ZERO;

        // View the flat allocation as individual rows, each with the
        // right number of columns.
        let rows = trace_slice[..n * ncols].chunks_exact_mut(ncols);

        for (row_index, (op, row)) in sponge_ops.iter().zip(rows).enumerate() {
            let Poseidon2CircuitRow {
                new_start,
                merkle_path,
                mmcs_bit,
                mmcs_index_sum,
                input_values,
                ..
            } = op;

            assert_eq!(
                input_values.len(),
                WIDTH,
                "Trace row input_values must have length WIDTH"
            );

            // Update the accumulator.
            //
            // If this is a Merkle row that continues a chain (not the
            // first row, not a chain boundary), apply the recurrence:
            //
            //     new_value = old_value × 2 + direction_bit
            //
            // Otherwise reset the accumulator. This happens on:
            //   - The very first row (no previous value exists).
            //   - Chain boundaries (a new Merkle proof starts).
            //   - Non-Merkle rows (sponge mode, no index to track).
            if row_index > 0 && *merkle_path && !*new_start {
                prev_mmcs_index_sum = prev_mmcs_index_sum.double() + F::from_bool(*mmcs_bit);
            } else {
                prev_mmcs_index_sum = *mmcs_index_sum;
            }

            // Write the 16 input field elements into the first 16 slots
            // of the row. These will be read back in pass 2 when the
            // permutation is computed.
            for (i, &val) in input_values.iter().enumerate() {
                row[i].write(val);
            }

            // Write the two circuit columns at the end of the row.
            //
            // First circuit column: the direction bit (0 = left, 1 = right).
            //
            // Second circuit column: the running accumulator value.
            let (_p2_part, circuit_part) = row.split_at_mut(p2_ncols);
            circuit_part[0].write(F::from_bool(*mmcs_bit));
            circuit_part[1].write(prev_mmcs_index_sum);
        }

        // Pass 2: Parallel
        //
        // Each row's permutation is independent of every other row.
        //
        // That means we can compute all of them in parallel.
        //
        // For each row:
        //
        //   1. Read back the 16 input elements written during pass 1.
        //
        //   2. Run the full Poseidon2 permutation: external rounds (S-box
        //      applied to all 16 elements), then partial rounds (S-box
        //      applied to just one element), then external rounds again.
        //
        //   3. Write all intermediate round states and the 16 output
        //      elements into the remaining permutation columns.

        trace_slice[..n * ncols]
            .par_chunks_exact_mut(ncols)
            .for_each(|row| {
                // Split the row into permutation columns and circuit columns.
                //
                // We only need the permutation part here. The circuit
                // columns were already finalized in pass 1.
                let (p2_part, _circuit_part) = row.split_at_mut(p2_ncols);

                // Read back the 16 input elements that pass 1 wrote.
                //
                // SAFETY: Pass 1 initialized exactly these positions.
                let input: [F; WIDTH] =
                    core::array::from_fn(|i| unsafe { p2_part[i].assume_init() });

                // Reinterpret the flat slice as the typed permutation column struct.
                //
                // This is a zero-copy cast. The struct is `#[repr(C)]`
                // and the slice has exactly the right number of elements.
                let (prefix, p2_cols, suffix) = unsafe {
                    p2_part.align_to_mut::<Poseidon2Cols<
                        MaybeUninit<F>,
                        WIDTH,
                        SBOX_DEGREE,
                        SBOX_REGISTERS,
                        HALF_FULL_ROUNDS,
                        PARTIAL_ROUNDS,
                    >>()
                };

                // Verify the cast produced exactly one struct with no
                // leftover bytes on either side.
                debug_assert!(prefix.is_empty(), "Alignment mismatch");
                debug_assert!(suffix.is_empty(), "Alignment mismatch");
                debug_assert_eq!(p2_cols.len(), 1);

                // Run the Poseidon2 permutation on the input.
                //
                // This fills in every column of the permutation struct:
                // - the beginning full rounds,
                // - the partial rounds,
                // - the ending full rounds,
                // - the final output state.
                generate_trace_rows_for_perm::<
                    F,
                    LinearLayers,
                    WIDTH,
                    SBOX_DEGREE,
                    SBOX_REGISTERS,
                    HALF_FULL_ROUNDS,
                    PARTIAL_ROUNDS,
                >(&mut p2_cols[0], input, constants);
            });

        // SAFETY: At this point every element has been initialized.
        //
        // Pass 1 wrote the input state, direction bit, and accumulator.
        //
        // Pass 2 wrote all intermediate round states and the output.
        //
        // We can now safely tell the allocator that these bytes are live.
        unsafe {
            trace_vec.set_len(n * ncols);
        }

        RowMajorMatrix::new(trace_vec, ncols)
    }
}

impl<
    F: PrimeField + Sync,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH> + Sync,
    const D: usize,
    const WIDTH: usize,
    const WIDTH_EXT: usize,
    const RATE_EXT: usize,
    const CAPACITY_EXT: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const WITNESS_EXT_D: usize,
> BaseAir<F>
    for Poseidon2CircuitAir<
        F,
        LinearLayers,
        D,
        WIDTH,
        WIDTH_EXT,
        RATE_EXT,
        CAPACITY_EXT,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        WITNESS_EXT_D,
    >
{
    /// Total number of value columns per row.
    ///
    /// Includes all Poseidon2 permutation columns (input, round
    /// intermediates, output).
    ///
    /// Also includes the two circuit-specific columns:
    /// - The direction bit,
    /// - The MMCS accumulator.
    fn width(&self) -> usize {
        num_cols::<
            Poseidon2Cols<u8, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
        >()
    }

    /// Build the preprocessed trace matrix.
    ///
    /// Pads to a power-of-two height.
    ///
    /// # Padding Strategy
    ///
    /// ```text
    ///     Row 0 .. n-1        actual preprocessed data
    ///     Row n (first pad)   chain boundary flag = 1, rest zero
    ///     Row n+1 .. end      all zeros
    /// ```
    ///
    /// The first padding row marks a chain boundary.
    ///
    /// This prevents chaining constraints from firing across the
    /// real-to-padding boundary.
    ///
    /// All subsequent padding rows are fully zero. Every selector is
    /// inactive, so every constraint is trivially satisfied.
    ///
    /// The chain boundary flag is the second-to-last field in each
    /// preprocessed row.
    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = Self::preprocessed_width();
        let len = self.preprocessed.len();

        debug_assert!(
            len.is_multiple_of(width),
            "Preprocessed trace length {len} is not a multiple of preprocessed width {width}."
        );

        let natural_rows = len / width;

        // The minimum height is already rounded to a power of two in
        // the builder method, so we can use it directly here.
        let padded_rows = natural_rows.next_power_of_two().max(self.min_height);

        // Clone the existing preprocessed data.
        let mut data = self.preprocessed.clone();

        // Pad with zeros up to the required power-of-two height.
        //
        // All-zero padding rows have every selector inactive, so
        // every constraint is trivially satisfied on those rows.
        data.resize(padded_rows * width, F::ZERO);

        // Mark the first padding row as a chain boundary.
        //
        // Without this, the chaining constraint would try to connect
        // the last real row to the first padding row. Setting the
        // chain-start flag to one disables that connection.
        //
        // The flag is the second-to-last field in each row.
        if padded_rows > natural_rows {
            data[len + width - 2] = F::ONE;
        }

        Some(RowMajorMatrix::new(data, width))
    }
}

/// Build the preprocessed trace from a sequence of circuit operations.
///
/// Each operation becomes one preprocessed row. The results are flattened
/// into a single vector in row-major order.
///
/// # Index Scaling
///
/// All witness indices are multiplied by the extension degree.
///
/// This way the CTL keys directly index into the flattened witness table.
/// For example, with extension degree 4 and logical index 5, the stored
/// value is 20.
///
/// # Grouped D=1 CTL
///
/// When `poseidon_extension_degree == 1` (base-field Poseidon2 slots) and `witness_bus_value_slots`
/// divides `IL` / `OL`, each pack uses **all-or-nothing** `in_ctl` / `out_ctl`, and witness indices
/// within a pack must be consecutive. `witness_bus_value_slots` is the AIR's `WITNESS_EXT_D`
/// (e.g. 5 for WitnessBus5). `d` scales stored indices and must match the circuit's extension degree.
///
/// # Compact D=1 width-16 / rate-8 layout
///
/// When `IL == 16`, `OL == 8`, and `poseidon_extension_degree == 1`, rows use a compact layout:
/// `OL` per-rate-limb `in_ctl`, `cap_in_ctl` (always zero; column retained for layout), `cap_chain_enable`,
/// `OL` sponge chain helpers `(1 − new_start)(1 − merkle_path)(1 − in_ctl_i)`, `OL` Merkle chain helpers
/// `(1 − new_start)(merkle_path)(1 − in_ctl_i)`, then `IL` input indices, `OL` output indices,
/// `OL` per-limb `out_ctl`, then four tail flags. Rate `in_ctl` may vary (e.g. partial-chunk overwrite).
/// Capacity inputs are not CTL-verified; sponge `new_start` rows enforce zero capacity in `eval`.
///
/// # Precomputed Selectors
///
/// Several boolean products are precomputed here to keep the constraint
/// degree at 3 during evaluation.
///
/// - The **sponge chain selector** is true when the row is not a chain
///   boundary, not a Merkle row, and the limb is not looked up via CTL.
///
/// - The **Merkle chain selector** is true when the row is not a chain
///   boundary, is a Merkle row, and the limb is not looked up via CTL.
///
/// - The **MMCS Merkle flag** is true when MMCS CTL is enabled and the
///   row is a Merkle row.
///
/// Computing these products at setup time avoids degree-4 expressions
/// in the constraint polynomial.
pub fn extract_preprocessed_from_operations<
    const IL: usize,
    const OL: usize,
    F: Field,
    OF: Field,
>(
    operations: &[Poseidon2CircuitRow<OF>],
    d: u32,
    poseidon_extension_degree: usize,
) -> Vec<F> {
    let row_width = poseidon2_preprocessed_row_width_for_air(poseidon_extension_degree, IL, OL);
    let mut preprocessed = Vec::with_capacity(operations.len() * row_width);

    let compact_d1 = poseidon2_uses_compact_d1_preprocessed(poseidon_extension_degree, IL, OL);

    for operation in operations {
        let Poseidon2CircuitRow {
            in_ctl,
            input_indices,
            out_ctl,
            output_indices,
            mmcs_index_sum_idx,
            mmcs_ctl_enabled,
            new_start,
            merkle_path,
            ..
        } = operation;

        debug_assert_eq!(in_ctl.len(), IL);
        debug_assert_eq!(input_indices.len(), IL);
        debug_assert_eq!(out_ctl.len(), OL);
        debug_assert_eq!(output_indices.len(), OL);

        if compact_d1 {
            for ctl in in_ctl.iter().take(OL) {
                preprocessed.push(F::from_bool(*ctl));
            }
            if !*merkle_path {
                for ctl in in_ctl.iter().take(IL).skip(OL) {
                    debug_assert!(
                        !ctl,
                        "compact D=1 Poseidon2: capacity must not be witness-fed on sponge rows"
                    );
                }
            }
            let cap_chain_enable = !*new_start;
            preprocessed.push(F::ZERO);
            preprocessed.push(F::from_bool(cap_chain_enable));
            for ctl in in_ctl.iter().take(OL) {
                preprocessed.push(F::from_bool(!*new_start && !*merkle_path && !ctl));
            }
            for ctl in in_ctl.iter().take(OL) {
                preprocessed.push(F::from_bool(!*new_start && *merkle_path && !ctl));
            }
            for input_index in input_indices.iter().take(IL) {
                preprocessed.push(F::from_u32(input_index * d));
            }
            for output_index in output_indices.iter().take(OL) {
                preprocessed.push(F::from_u32(output_index * d));
            }
            for ctl in out_ctl.iter().take(OL) {
                preprocessed.push(F::from_bool(*ctl));
            }
            preprocessed.push(F::from_u64(*mmcs_index_sum_idx as u64 * d as u64));
            preprocessed.push(F::from_bool(*mmcs_ctl_enabled && *merkle_path));
            preprocessed.push(F::from_bool(*new_start));
            preprocessed.push(F::from_bool(*merkle_path));
        } else {
            let row = Poseidon2PreprocessedRow::<IL, OL, F> {
                input_limbs: core::array::from_fn(|i| {
                    let ctl = in_ctl[i];
                    Poseidon2PrepInputLimb {
                        idx: F::from_u32(input_indices[i] * d),
                        in_ctl: F::from_bool(ctl),
                        normal_chain_sel: F::from_bool(!*new_start && !*merkle_path && !ctl),
                        merkle_chain_sel: F::from_bool(!*new_start && *merkle_path && !ctl),
                    }
                }),

                output_limbs: core::array::from_fn(|i| Poseidon2PrepOutputLimb {
                    idx: F::from_u32(output_indices[i] * d),
                    out_ctl: F::from_bool(out_ctl[i]),
                }),

                mmcs_index_sum_ctl_idx: F::from_u64(*mmcs_index_sum_idx as u64 * d as u64),

                mmcs_merkle_flag: F::from_bool(*mmcs_ctl_enabled && *merkle_path),

                new_start: F::from_bool(*new_start),

                merkle_path: F::from_bool(*merkle_path),
            };
            row.write_into(&mut preprocessed);
        }
    }

    preprocessed
}

/// Evaluate all circuit-level constraints for one pair of adjacent rows.
///
/// This is the core constraint function.
///
/// It enforces five groups of constraints on the builder.
///
/// 1. **Boolean** — the direction bit must be 0 or 1.
///
/// 2. **Sponge chaining** — when the sponge chain selector is active,
///    the next row's input equals the current row's output for that limb.
///    Checked element by element across the extension degree.
///
/// 3. **Merkle-path chaining** — when the Merkle chain selector is
///    active, the chaining direction depends on the direction bit:
///
///    ```text
///        bit = 0 (left child)    next input limbs 0-1 ← current output 0-1
///        bit = 1 (right child)   next input limbs 2-3 ← current output 0-1
///    ```
///
///    Only the first two limbs carry the Merkle selector.
///    Limbs 2-3 reuse the same selectors gated on the opposite direction.
///
/// 4. **MMCS accumulator** — on Merkle rows that are not chain boundaries,
///    the next accumulator equals twice the current plus the next bit.
///
/// 5. **Poseidon2 permutation** — delegated to the inner permutation AIR
///    via a sub-builder restricted to the permutation columns.
///    Unconditional on every row.
///
/// Chain selectors and the Merkle flag are preprocessed columns. They are
/// known to the verifier and do not need boolean assertions.
///
/// The direction bit is a prover-supplied value column. It must be
/// explicitly constrained to be boolean.
#[unroll::unroll_for_loops]
pub(crate) fn eval<
    AB: AirBuilder,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const D: usize,
    const WIDTH: usize,
    const WIDTH_EXT: usize,
    const RATE_EXT: usize,
    const CAPACITY_EXT: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const WITNESS_EXT_D: usize,
>(
    air: &Poseidon2CircuitAir<
        AB::F,
        LinearLayers,
        D,
        WIDTH,
        WIDTH_EXT,
        RATE_EXT,
        CAPACITY_EXT,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        WITNESS_EXT_D,
    >,
    builder: &mut AB,
    local: &Poseidon2CircuitCols<
        AB::Var,
        Poseidon2Cols<
            AB::Var,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >,
    >,
    next: &Poseidon2CircuitCols<
        AB::Var,
        Poseidon2Cols<
            AB::Var,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >,
    >,
    next_preprocessed: &[AB::Var],
) {
    // Extract the three things we'll reference repeatedly:
    //
    //   - The direction bit from the next row (left vs right child).
    //
    //   - The current row's output state — the 16 field elements
    //     produced by the Poseidon2 permutation on this row. Located
    //     in the last full-round's post-state.
    //
    //   - The next row's input state — the 16 field elements that
    //     will be fed into the next permutation. Chaining constraints
    //     tie these to the current output.
    let next_bit = next.mmcs_bit;
    let local_out = &local.poseidon2.ending_full_rounds[HALF_FULL_ROUNDS - 1].post;
    let next_in = &next.poseidon2.inputs;

    // Boolean constraint
    //
    // The direction bit is a value column filled by the prover at
    // runtime. A cheating prover could put any field element here.
    //
    // We constrain it to be 0 or 1 by asserting:
    //
    //     bit × (1 − bit) = 0
    //
    // Preprocessed flags don't need this check — they were committed
    // at setup time and cannot be changed.

    builder.assert_bool(local.mmcs_bit);

    if poseidon2_uses_compact_d1_preprocessed(D, WIDTH_EXT, RATE_EXT) {
        let hdr = poseidon2_d1_compact_preprocessed_header_cols(RATE_EXT);
        debug_assert_eq!(
            next_preprocessed.len(),
            hdr + WIDTH_EXT + RATE_EXT + RATE_EXT + 4
        );
        let s = next_preprocessed;
        let cap_chain_enable = s[RATE_EXT + 1];
        let rate_sponge_base = RATE_EXT + 2;
        let rate_merkle_base = rate_sponge_base + RATE_EXT;
        let tail = hdr + WIDTH_EXT + RATE_EXT + RATE_EXT;
        let next_new_start = s[tail + 2];
        let next_merkle_path = s[tail + 3];
        let not_next_new_start = AB::Expr::ONE - next_new_start.into();

        // Sponge chaining (compact): rate uses precomputed `(1−ns)(1−merkle)(1−ctl_i)`; capacity shares `cap_chain_enable`.
        for limb in 0..RATE_EXT {
            let chain_en = s[rate_sponge_base + limb];
            for d in 0..D {
                builder
                    .when_transition()
                    .when(chain_en)
                    .assert_zero(next_in[limb * D + d] - local_out[limb * D + d]);
            }
        }
        let not_merkle = AB::Expr::ONE - next_merkle_path.into();
        for limb in RATE_EXT..WIDTH_EXT {
            let chain_en = cap_chain_enable * not_merkle.clone();
            for d in 0..D {
                builder
                    .when_transition()
                    .when(chain_en.clone())
                    .assert_zero(next_in[limb * D + d] - local_out[limb * D + d]);
            }
        }

        // Merkle-path chaining (compact): precomputed `(1−ns)(merkle)(1−ctl_i)` × direction bit (degree 3).
        let is_left = AB::Expr::ONE - next_bit.into();
        for i in 0..RATE_EXT {
            let merkle_chain_i = s[rate_merkle_base + i];
            let gate_left_i = merkle_chain_i * is_left.clone();
            let gate_right_i = merkle_chain_i * next_bit;

            for d in 0..D {
                builder
                    .when_transition()
                    .when(gate_left_i.clone())
                    .assert_zero(next_in[i * D + d] - local_out[i * D + d]);

                builder
                    .when_transition()
                    .when(gate_right_i.clone())
                    .assert_zero(next_in[(RATE_EXT + i) * D + d] - local_out[i * D + d]);
            }
        }

        // Sponge chain starts (next row new_start, not Merkle): capacity is never witness-fed;
        // replacing the former zero-constant CTL sends for slots RATE_EXT..WIDTH_EXT.
        for slot in RATE_EXT..WIDTH_EXT {
            for d in 0..D {
                builder
                    .when_transition()
                    .when(next_new_start)
                    .when(not_merkle.clone())
                    .assert_zero(next_in[slot * D + d]);
            }
        }

        builder
            .when_transition()
            .when(not_next_new_start)
            .when(next_merkle_path)
            .assert_zero(
                next.mmcs_index_sum - (local.mmcs_index_sum * AB::Expr::TWO + next.mmcs_bit.into()),
            );
    } else {
        let next_prep: &Poseidon2PreprocessedRow<WIDTH_EXT, RATE_EXT, AB::Var> =
            next_preprocessed.borrow();

        // Sponge chaining
        //
        // In sponge mode the output of one permutation feeds directly
        // into the input of the next permutation.
        //
        // For example, if row 0 outputs [a, b, c, ...] then row 1 must
        // have input [a, b, c, ...].
        //
        // We check this element by element. Each limb has D base-field
        // elements (the extension degree), so we loop over all of them.
        //
        // The sponge chain selector gates the constraint. It is only
        // active on continuation rows in sponge mode. On chain boundaries,
        // Merkle rows, or CTL-loaded limbs, the selector is zero and the
        // constraint is trivially satisfied.

        for limb in 0..WIDTH_EXT {
            for d in 0..D {
                let gate = next_prep.input_limbs[limb].normal_chain_sel;
                builder
                    .when_transition()
                    .when(gate)
                    .assert_zero(next_in[limb * D + d] - local_out[limb * D + d]);
            }
        }

        // Merkle-path chaining: first `RATE_EXT` logical limbs of the output
        // form our digest; the sibling occupies the next `RATE_EXT` limbs of
        // the next row's input. The direction bit selects left vs right placement.
        let is_left = AB::Expr::ONE - next_bit.into();

        for i in 0..RATE_EXT {
            let gate_left_i = next_prep.input_limbs[i].merkle_chain_sel * is_left.clone();
            for d in 0..D {
                builder
                    .when_transition()
                    .when(gate_left_i.clone())
                    .assert_zero(next_in[i * D + d] - local_out[i * D + d]);
            }
        }
        for i in 0..RATE_EXT {
            let gate_right_i = next_prep.input_limbs[i].merkle_chain_sel * next_bit;
            for d in 0..D {
                builder
                    .when_transition()
                    .when(gate_right_i.clone())
                    .assert_zero(next_in[(RATE_EXT + i) * D + d] - local_out[i * D + d]);
            }
        }

        // MMCS accumulator
        //
        // As the circuit walks up a Merkle tree, it sees one direction bit
        // per level. These bits form the binary representation of the leaf
        // index being authenticated.
        //
        // The accumulator reconstructs that index with the recurrence:
        //
        //     next_sum = current_sum × 2 + next_bit
        //
        // For example, authenticating leaf 5 (binary 101):
        //
        //     row 0:  acc = 1            (first bit)
        //     row 1:  acc = 1×2 + 0 = 2  (second bit)
        //     row 2:  acc = 2×2 + 1 = 5  (third bit → final index)
        //
        // The constraint only fires when the next row is a Merkle row
        // that is not a chain boundary. On chain boundaries the
        // accumulator resets, and on non-Merkle rows it is unused.

        // Compute (1 − next_new_start). This is 1 when the next row
        // continues a chain, 0 when it starts a new one.
        let not_next_new_start = AB::Expr::ONE - next_prep.new_start.into();

        // The constraint:
        //
        //     next_accumulator = current_accumulator × 2 + next_direction_bit
        //
        // Rearranged for assert_zero:
        //
        //     next_acc − (current_acc × 2 + next_bit) = 0
        //
        // Gated on: not a chain boundary AND is a Merkle row.
        builder
            .when_transition()
            .when(not_next_new_start)
            .when(next_prep.merkle_path)
            .assert_zero(
                next.mmcs_index_sum - (local.mmcs_index_sum * AB::Expr::TWO + next.mmcs_bit.into()),
            );
    }

    // Poseidon2 permutation
    //
    // Every row must satisfy the Poseidon2 permutation constraint:
    // the output state must be the correct hash of the input state.
    //
    // This is unconditional — it applies regardless of whether the
    // row is sponge, Merkle, padding, or anything else.
    //
    // The permutation constraint is handled by a separate AIR. We
    // give it a sub-builder that only sees the permutation columns
    // (not the two circuit columns at the end of the row).

    let p3_poseidon2_num_cols = p3_poseidon2_air::num_cols::<
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >();
    let mut sub_builder = SubAirBuilder::<
        AB,
        Poseidon2Air<
            AB::F,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >,
        AB::Var,
    >::new(builder, 0..p3_poseidon2_num_cols);

    air.p3_poseidon2.eval(&mut sub_builder);
}

/// Unchecked constraint evaluation with a concrete builder type.
///
/// Exists to support the batch prover.
///
/// In the batch prover the constraint evaluation dispatch erases the
/// concrete builder type behind a trait object. The caller provides two
/// builder types: the erased one and a concrete one that carries the
/// required field bounds.
///
/// At runtime both must be the same type with the same field.
///
/// All five arguments are transmuted from the erased types to the
/// concrete types before calling the main evaluation function.
///
/// This is sound only if the types are truly identical at runtime.
///
/// # Safety
///
/// The caller must guarantee:
///
/// - The AIR's field type, the erased builder's field type, and the
///   concrete builder's field type are all the same.
///
/// - The erased and concrete builder types have identical memory layout.
#[allow(clippy::missing_transmute_annotations)]
pub unsafe fn eval_unchecked_with_concrete<
    F: PrimeField,
    AB: AirBuilder,
    ABConcrete: AirBuilder,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const D: usize,
    const WIDTH: usize,
    const WIDTH_EXT: usize,
    const RATE_EXT: usize,
    const CAPACITY_EXT: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const WITNESS_EXT_D: usize,
>(
    air: &Poseidon2CircuitAir<
        F,
        LinearLayers,
        D,
        WIDTH,
        WIDTH_EXT,
        RATE_EXT,
        CAPACITY_EXT,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        WITNESS_EXT_D,
    >,
    builder: &mut AB,
    local: &Poseidon2CircuitCols<
        AB::Var,
        Poseidon2Cols<
            AB::Var,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >,
    >,
    next: &Poseidon2CircuitCols<
        AB::Var,
        Poseidon2Cols<
            AB::Var,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >,
    >,
    next_preprocessed: &[AB::Var],
) where
    ABConcrete::F: PrimeField,
{
    // SAFETY: The caller guarantees all erased types are identical to
    // their concrete counterparts at runtime.
    //
    // Each transmute reinterprets the same memory under the concrete
    // type so the main evaluation function can be called with proper
    // trait bounds.
    unsafe {
        let builder_c = core::mem::transmute(builder);
        let local_c = core::mem::transmute(local);
        let next_c = core::mem::transmute(next);
        let next_preprocessed_c = core::mem::transmute(next_preprocessed);
        let air_c = core::mem::transmute(air);
        eval::<
            ABConcrete,
            LinearLayers,
            D,
            WIDTH,
            WIDTH_EXT,
            RATE_EXT,
            CAPACITY_EXT,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            WITNESS_EXT_D,
        >(air_c, builder_c, local_c, next_c, next_preprocessed_c);
    }
}

/// Unchecked constraint evaluation with a field type mismatch.
///
/// The AIR's field type may differ from the builder's field type at
/// compile time. At runtime they must be the same.
///
/// This function transmutes the AIR reference so its field matches the
/// builder, then calls the main evaluation function.
///
/// Simpler than the concrete-builder variant above: only the AIR needs
/// to be transmuted. The builder already has the correct associated types.
///
/// # Safety
///
/// The caller must guarantee that the AIR's field type and the builder's
/// field type are the same at runtime.
///
/// Violating this leads to undefined behavior.
#[allow(clippy::missing_transmute_annotations)]
pub unsafe fn eval_unchecked<
    F: PrimeField,
    AB: AirBuilder,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const D: usize,
    const WIDTH: usize,
    const WIDTH_EXT: usize,
    const RATE_EXT: usize,
    const CAPACITY_EXT: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const WITNESS_EXT_D: usize,
>(
    air: &Poseidon2CircuitAir<
        F,
        LinearLayers,
        D,
        WIDTH,
        WIDTH_EXT,
        RATE_EXT,
        CAPACITY_EXT,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        WITNESS_EXT_D,
    >,
    builder: &mut AB,
    local: &Poseidon2CircuitCols<
        AB::Var,
        Poseidon2Cols<
            AB::Var,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >,
    >,
    next: &Poseidon2CircuitCols<
        AB::Var,
        Poseidon2Cols<
            AB::Var,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >,
    >,
    next_preprocessed: &[AB::Var],
) where
    AB::F: PrimeField,
{
    // SAFETY: The caller guarantees the two field types are identical at
    // runtime, so the AIR struct has the same memory layout under both.
    unsafe {
        let air_transmuted = core::mem::transmute(air);

        eval::<
            AB,
            LinearLayers,
            D,
            WIDTH,
            WIDTH_EXT,
            RATE_EXT,
            CAPACITY_EXT,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            WITNESS_EXT_D,
        >(air_transmuted, builder, local, next, next_preprocessed);
    }
}

impl<
    AB: AirBuilder,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const D: usize,
    const WIDTH: usize,
    const WIDTH_EXT: usize,
    const RATE_EXT: usize,
    const CAPACITY_EXT: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const WITNESS_EXT_D: usize,
> Air<AB>
    for Poseidon2CircuitAir<
        AB::F,
        LinearLayers,
        D,
        WIDTH,
        WIDTH_EXT,
        RATE_EXT,
        CAPACITY_EXT,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        WITNESS_EXT_D,
    >
where
    AB::F: PrimeField,
{
    #[inline]
    fn eval(&self, builder: &mut AB) {
        // Get the main trace window.
        //
        // It provides the current row and the next row as flat slices.
        let main = builder.main();

        // Reinterpret the flat slices as typed column structs.
        //
        // This is a zero-copy cast enabled by the `#[repr(C)]` layout
        // and the `Borrow` implementations in the columns module.
        let local = main.current_slice().borrow();
        let next = main.next_slice().borrow();

        // Get the preprocessed trace window and extract the next row.
        //
        // The clone here copies a small window struct (two slice
        // pointers), not the full preprocessed matrix.
        let preprocessed = builder.preprocessed().clone();
        let next_preprocessed = preprocessed.next_slice();

        // Delegate to the core constraint function, which enforces all
        // five constraint groups.
        eval::<
            _,
            _,
            D,
            WIDTH,
            WIDTH_EXT,
            RATE_EXT,
            CAPACITY_EXT,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            WITNESS_EXT_D,
        >(self, builder, local, next, next_preprocessed);
    }
}

impl<
    F: PrimeField + PrimeCharacteristicRing,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const D: usize,
    const WIDTH: usize,
    const WIDTH_EXT: usize,
    const RATE_EXT: usize,
    const CAPACITY_EXT: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const WITNESS_EXT_D: usize,
> LookupAir<F>
    for Poseidon2CircuitAir<
        F,
        LinearLayers,
        D,
        WIDTH,
        WIDTH_EXT,
        RATE_EXT,
        CAPACITY_EXT,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        WITNESS_EXT_D,
    >
{
    /// Allocate one permutation argument column and return its index.
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let lookup_column_idx = self.num_lookup_cols;
        self.num_lookup_cols += 1;
        vec![lookup_column_idx]
    }

    /// Build symbolic lookup descriptions for all CTL interactions.
    ///
    /// Uses a symbolic AIR builder to produce symbolic expressions for
    /// each lookup's key, multiplicity, and direction.
    ///
    /// The STARK framework compiles these into permutation argument constraints.
    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        // Build a symbolic AIR builder.
        //
        // This creates one symbolic variable per column. We use these
        // symbolic variables to express lookup keys and multiplicities
        // as formal polynomials.
        //
        // The STARK framework later evaluates these polynomials at
        // concrete points to enforce the permutation argument.
        let air_layout = AirLayout {
            preprocessed_width: Self::preprocessed_width(),
            main_width: BaseAir::<F>::width(self),
            num_public_values: 0,
            permutation_width: 0,
            num_permutation_challenges: 0,
            num_permutation_values: 0,
            num_periodic_columns: 0,
        };
        let symbolic_air_builder = SymbolicAirBuilder::<F>::new(air_layout);
        let symbolic_main = symbolic_air_builder.main();
        let symbolic_main_local = symbolic_main.current_slice();

        let local: &Poseidon2CircuitCols<
            SymbolicVariable<F>,
            Poseidon2Cols<
                SymbolicVariable<F>,
                WIDTH,
                SBOX_DEGREE,
                SBOX_REGISTERS,
                HALF_FULL_ROUNDS,
                PARTIAL_ROUNDS,
            >,
        > = (*symbolic_main_local).borrow();

        // Extract the current row (row 0) and next row (row 1) from the
        // preprocessed trace, then cast them to typed structs.
        //
        // We need both rows because some multiplicities depend on the
        // next row's chain-start flag.
        let preprocessed = symbolic_air_builder.preprocessed();
        let local_preprocessed = preprocessed
            .row_slice(0)
            .expect("The preprocessed matrix has only one row?");
        let local_flat: &[SymbolicVariable<F>] = (*local_preprocessed).borrow();
        let next_preprocessed = preprocessed
            .row_slice(1)
            .expect("The preprocessed matrix has only one row?");
        let next_flat: &[SymbolicVariable<F>] = (*next_preprocessed).borrow();

        // Total lookups: one per input limb + one per rate output limb + MMCS
        // (compact D=1 layout counts input CTLs only over the rate).
        let compact_d1 = poseidon2_uses_compact_d1_preprocessed(D, WIDTH_EXT, RATE_EXT);
        let input_lookup_count = if compact_d1 { RATE_EXT } else { WIDTH_EXT };
        let output_lookup_count = RATE_EXT;
        let mut lookups = Vec::with_capacity(input_lookup_count + output_lookup_count + 1);

        if compact_d1 {
            let hdr = poseidon2_d1_compact_preprocessed_header_cols(RATE_EXT);
            let tail = hdr + WIDTH_EXT + RATE_EXT + RATE_EXT;
            debug_assert_eq!(local_flat.len(), tail + 4);
            debug_assert_eq!(next_flat.len(), tail + 4);

            let merkle_path_p = local_flat[tail + 3];
            let not_merkle = SymbolicExpression::Leaf(BaseLeaf::Constant(F::ONE))
                - SymbolicExpression::from(merkle_path_p);
            let idx_base = hdr;

            // Input limb lookups (rate only; sponge new_start capacity is zero-asserted in eval)
            for limb_idx in 0..RATE_EXT {
                let idx = local_flat[idx_base + limb_idx];
                let in_ctl = local_flat[limb_idx];
                let mut input_idx_limb: Vec<SymbolicExpression<F>> = iter::once(idx)
                    .chain(
                        local.poseidon2.inputs[limb_idx * D..(limb_idx + 1) * D]
                            .iter()
                            .copied(),
                    )
                    .map(SymbolicExpression::from)
                    .collect();
                input_idx_limb.extend(iter::repeat_n(
                    SymbolicExpression::Leaf(BaseLeaf::Constant(F::ZERO)),
                    WITNESS_EXT_D - D,
                ));
                let mult = SymbolicExpression::from(in_ctl) * not_merkle.dup();
                lookups.push(LookupAir::register_lookup(
                    self,
                    Kind::Global("WitnessChecks".to_string()),
                    &[(input_idx_limb, mult, Direction::Send)],
                ));
            }

            // Output limb lookups
            let out_idx_base = idx_base + WIDTH_EXT;
            let out_ctl_base = out_idx_base + RATE_EXT;
            for limb_idx in 0..RATE_EXT {
                let idx = local_flat[out_idx_base + limb_idx];
                let out_ctl = local_flat[out_ctl_base + limb_idx];
                let mut output_idx_limb: Vec<SymbolicExpression<F>> = iter::once(idx)
                    .chain(
                        local.poseidon2.ending_full_rounds[HALF_FULL_ROUNDS - 1].post
                            [limb_idx * D..(limb_idx + 1) * D]
                            .iter()
                            .copied(),
                    )
                    .map(SymbolicExpression::from)
                    .collect();
                output_idx_limb.extend(iter::repeat_n(
                    SymbolicExpression::Leaf(BaseLeaf::Constant(F::ZERO)),
                    WITNESS_EXT_D - D,
                ));
                lookups.push(LookupAir::register_lookup(
                    self,
                    Kind::Global("WitnessChecks".to_string()),
                    &[(
                        output_idx_limb,
                        SymbolicExpression::from(out_ctl),
                        Direction::Receive,
                    )],
                ));
            }

            let multiplicity = SymbolicExpression::from(local_flat[tail + 1])
                * SymbolicExpression::from(next_flat[tail + 2]);
            let mut mmcs_index_sum_lookup = vec![
                SymbolicExpression::from(local_flat[tail]),
                SymbolicExpression::from(local.mmcs_index_sum),
            ];
            mmcs_index_sum_lookup.extend(iter::repeat_n(
                SymbolicExpression::Leaf(BaseLeaf::Constant(F::ZERO)),
                WITNESS_EXT_D - 1,
            ));
            lookups.push(LookupAir::register_lookup(
                self,
                Kind::Global("WitnessChecks".to_string()),
                &[(mmcs_index_sum_lookup, multiplicity, Direction::Send)],
            ));
        } else {
            let local_preprocessed: &Poseidon2PreprocessedRow<
                WIDTH_EXT,
                RATE_EXT,
                SymbolicVariable<F>,
            > = local_flat.borrow();
            let next_preprocessed: &Poseidon2PreprocessedRow<
                WIDTH_EXT,
                RATE_EXT,
                SymbolicVariable<F>,
            > = next_flat.borrow();

            // Input limb lookups
            //
            // Input CTL lookups are disabled on Merkle rows. If they were
            // active, permuting the CTL metadata based on the runtime
            // direction bit would push the constraint degree above 3.
            //
            // Disabling input CTL on Merkle rows is sound because:
            //
            //   - Digest values in the row were already CTL-verified when
            //     they were first created (on a non-Merkle row).
            //
            //   - Sibling values are private proof data — wrong siblings
            //     simply produce the wrong Merkle root.
            //
            //   - Chained values are AIR-constrained to equal the previous
            //     permutation output.

            // Compute (1 − merkle_path). This is 1 on sponge rows and 0 on
            // Merkle rows. Used to disable input CTL on Merkle rows.
            let not_merkle = SymbolicExpression::Leaf(BaseLeaf::Constant(F::ONE))
                - SymbolicExpression::from(local_preprocessed.merkle_path);

            for limb_idx in 0..WIDTH_EXT {
                let limb = &local_preprocessed.input_limbs[limb_idx];

                // Build the lookup key: [witness_index, elem_0, elem_1, ..., elem_{D-1}].
                //
                // The witness index tells the Witness table which slot to
                // look up. The extension-field elements are the actual values
                // being checked.
                let mut input_idx_limb: Vec<SymbolicExpression<F>> = iter::once(limb.idx)
                    .chain(
                        local.poseidon2.inputs[limb_idx * D..(limb_idx + 1) * D]
                            .iter()
                            .copied(),
                    )
                    .map(SymbolicExpression::from)
                    .collect();
                input_idx_limb.extend(iter::repeat_n(
                    SymbolicExpression::Leaf(BaseLeaf::Constant(F::ZERO)),
                    WITNESS_EXT_D - D,
                ));

                // Multiplicity = CTL enable flag × (1 − merkle_path).
                //
                // This is zero on Merkle rows (disabling the lookup) and
                // equal to the CTL flag on sponge rows.
                //
                // Both factors are preprocessed, so this product costs
                // nothing at constraint-evaluation time.
                let mult = SymbolicExpression::from(limb.in_ctl) * not_merkle.dup();

                // Direction::Send means this table is the sender:
                //
                // "I claim my input limb matches the value in the Witness table at this index."
                lookups.push(LookupAir::register_lookup(
                    self,
                    Kind::Global("WitnessChecks".to_string()),
                    &[(input_idx_limb, mult, Direction::Send)],
                ));
            }

            // Output limb lookups
            //
            // Each publicly exposed output limb receives its value from the
            // Witness table.
            //
            // The key has the same format as input lookups: witness index
            // followed by the extension-field elements.
            //
            // Direction::Receive means "the Witness table sent this value
            // to me." If the output doesn't match, the permutation argument
            // will fail.

            for limb_idx in 0..RATE_EXT {
                let limb = &local_preprocessed.output_limbs[limb_idx];

                // Build the lookup key from the output state.
                //
                // The output lives in the last full round's post-state.
                let mut output_idx_limb: Vec<SymbolicExpression<F>> = iter::once(limb.idx)
                    .chain(
                        local.poseidon2.ending_full_rounds[HALF_FULL_ROUNDS - 1].post
                            [limb_idx * D..(limb_idx + 1) * D]
                            .iter()
                            .copied(),
                    )
                    .map(SymbolicExpression::from)
                    .collect();
                output_idx_limb.extend(iter::repeat_n(
                    SymbolicExpression::Leaf(BaseLeaf::Constant(F::ZERO)),
                    WITNESS_EXT_D - D,
                ));

                lookups.push(LookupAir::register_lookup(
                    self,
                    Kind::Global("WitnessChecks".to_string()),
                    &[(
                        output_idx_limb,
                        SymbolicExpression::from(limb.out_ctl),
                        Direction::Receive,
                    )],
                ));
            }

            // MMCS accumulator lookup
            //
            // At the end of a Merkle chain the accumulated leaf index must
            // be sent to the Witness table for verification.
            //
            // The lookup fires on the last Merkle row of a chain. We
            // detect this as: the current row has the MMCS Merkle flag set
            // AND the next row starts a new chain.
            //
            // Both flags are preprocessed, so the multiplicity is a
            // degree-2 expression that costs nothing extra.
            //
            // The accumulator is a single base-field element. The Witness
            // table expects extension-field-width keys, so we pad with
            // zeros to fill the remaining extension-degree minus one slots.

            let multiplicity = local_preprocessed.mmcs_merkle_flag * next_preprocessed.new_start;

            // Build the lookup key: [witness_index, accumulator, 0, 0, ...].
            //
            // The accumulator is one base-field element. We zero-pad to
            // match the extension-field width expected by the Witness table.
            let mut mmcs_index_sum_lookup = vec![
                SymbolicExpression::from(local_preprocessed.mmcs_index_sum_ctl_idx),
                SymbolicExpression::from(local.mmcs_index_sum),
            ];
            mmcs_index_sum_lookup.extend(iter::repeat_n(
                SymbolicExpression::Leaf(BaseLeaf::Constant(F::ZERO)),
                WITNESS_EXT_D - 1,
            ));

            lookups.push(LookupAir::register_lookup(
                self,
                Kind::Global("WitnessChecks".to_string()),
                &[(mmcs_index_sum_lookup, multiplicity, Direction::Send)],
            ));
        }

        lookups
    }
}

#[cfg(test)]
mod test {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_commit::ExtensionMmcs;
    use p3_field::extension::BinomialExtensionField;
    use p3_fri::{FriParameters, TwoAdicFriPcs};
    use p3_keccak::{Keccak256Hash, KeccakF};
    use p3_merkle_tree::MerkleTreeHidingMmcs;
    use p3_poseidon2::ExternalLayerConstants;
    use p3_poseidon2_air::RoundConstants;
    use p3_symmetric::{
        CompressionFunctionFromHasher, PaddingFreeSponge, Permutation, SerializingHasher,
    };
    use p3_uni_stark::{
        StarkConfig, prove_with_preprocessed, setup_preprocessed, verify_with_preprocessed,
    };
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::Poseidon2CircuitAirBabyBearD4Width16;
    use crate::columns::{POSEIDON2_LIMBS, POSEIDON2_PUBLIC_OUTPUT_LIMBS};

    const WIDTH: usize = 16;

    #[test]
    fn prove_poseidon2_sponge() -> Result<
        (),
        p3_uni_stark::VerificationError<
            p3_fri::verifier::FriError<
                p3_merkle_tree::MerkleTreeError,
                p3_merkle_tree::MerkleTreeError,
            >,
        >,
    > {
        type Val = BabyBear;
        type Challenge = BinomialExtensionField<Val, 4>;

        type ByteHash = Keccak256Hash;
        let byte_hash = ByteHash {};

        type U64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
        let u64_hash = U64Hash::new(KeccakF {});

        type FieldHash = SerializingHasher<U64Hash>;
        let field_hash = FieldHash::new(u64_hash);

        type MyCompress = CompressionFunctionFromHasher<U64Hash, 2, 4>;
        let compress = MyCompress::new(u64_hash);

        // WARNING: DO NOT USE SmallRng in proper applications! Use a real PRNG instead!
        type MyMmcs = MerkleTreeHidingMmcs<
            [Val; p3_keccak::VECTOR_LEN],
            [u64; p3_keccak::VECTOR_LEN],
            FieldHash,
            MyCompress,
            SmallRng,
            2,
            4,
            4,
        >;
        let mut rng = SmallRng::seed_from_u64(1);
        let val_mmcs = MyMmcs::new(field_hash, compress, 0, rng.clone());

        type ChallengeMmcs = ExtensionMmcs<Val, Challenge, MyMmcs>;
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
        let challenger = Challenger::from_hasher(vec![], byte_hash);

        let mut fri_params = FriParameters::new_benchmark(challenge_mmcs);
        fri_params.log_blowup = 4;

        let beginning_full_constants = rng.random();
        let partial_constants = rng.random();
        let ending_full_constants = rng.random();

        let constants = RoundConstants::new(
            beginning_full_constants,
            partial_constants,
            ending_full_constants,
        );

        let perm = Poseidon2BabyBear::<WIDTH>::new(
            ExternalLayerConstants::new(
                beginning_full_constants.to_vec(),
                ending_full_constants.to_vec(),
            ),
            partial_constants.to_vec(),
        );

        // Generate random inputs.
        let mut rng = SmallRng::seed_from_u64(1);

        // Row A: new_start=true, sponge mode - use random initial state
        let state_a: [Val; WIDTH] = core::array::from_fn(|_| rng.random());
        let output_a = perm.permute(state_a);

        let sponge_a: Poseidon2CircuitRow<Val> = Poseidon2CircuitRow {
            new_start: true,
            merkle_path: false,
            mmcs_bit: false,
            mmcs_index_sum: Val::ZERO,
            input_values: state_a.to_vec(),
            in_ctl: vec![false; POSEIDON2_LIMBS],
            input_indices: vec![0; POSEIDON2_LIMBS],
            out_ctl: vec![false; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            output_indices: vec![0; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            mmcs_index_sum_idx: 0,
            mmcs_ctl_enabled: false,
        };

        // Row B: new_start=false, sponge mode - chain from output_a
        let state_b = output_a;
        let output_b = perm.permute(state_b);

        let sponge_b: Poseidon2CircuitRow<Val> = Poseidon2CircuitRow {
            new_start: false,
            merkle_path: false,
            mmcs_bit: true,
            mmcs_index_sum: Val::ZERO,
            input_values: state_b.to_vec(),
            in_ctl: vec![false; POSEIDON2_LIMBS],
            input_indices: vec![0; POSEIDON2_LIMBS],
            out_ctl: vec![false; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            output_indices: vec![0; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            mmcs_index_sum_idx: 0,
            mmcs_ctl_enabled: false,
        };

        // Row C: new_start=false, merkle mode, mmcs_bit=false
        // In merkle mode with mmcs_bit=0: prev digest (out[0..1]) goes to input limbs 0..1
        // The rest (limbs 2..3) can be zeros (sibling)
        const D: usize = 4; // extension degree
        let mut state_c = [Val::ZERO; WIDTH];
        // Chain prev output[0..2*D] into input[0..2*D] (limbs 0-1)
        state_c[0..2 * D].copy_from_slice(&output_b[0..2 * D]);
        let output_c = perm.permute(state_c);

        let sponge_c: Poseidon2CircuitRow<Val> = Poseidon2CircuitRow {
            new_start: false,
            merkle_path: true,
            mmcs_bit: false,
            mmcs_index_sum: Val::ZERO,
            input_values: state_c.to_vec(),
            in_ctl: vec![false; POSEIDON2_LIMBS],
            input_indices: vec![0; POSEIDON2_LIMBS],
            out_ctl: vec![false; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            output_indices: vec![0; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            mmcs_index_sum_idx: 0,
            mmcs_ctl_enabled: false,
        };

        // Row D: new_start=false, sponge mode - chain from output_c
        let state_d = output_c;

        let sponge_d: Poseidon2CircuitRow<Val> = Poseidon2CircuitRow {
            new_start: false,
            merkle_path: false,
            mmcs_bit: false,
            mmcs_index_sum: Val::ZERO,
            input_values: state_d.to_vec(),
            in_ctl: vec![false; POSEIDON2_LIMBS],
            input_indices: vec![0; POSEIDON2_LIMBS],
            out_ctl: vec![false; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            output_indices: vec![0; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            mmcs_index_sum_idx: 0,
            mmcs_ctl_enabled: false,
        };

        let mut rows = vec![sponge_a, sponge_b, sponge_c, sponge_d];
        let degree_bits = 5;
        let target_rows = 1 << degree_bits;
        if rows.len() < target_rows {
            // Filler rows must have new_start=true to avoid chaining constraints
            let filler = Poseidon2CircuitRow {
                new_start: true,
                merkle_path: false,
                mmcs_bit: false,
                mmcs_index_sum: Val::ZERO,
                input_values: Val::zero_vec(WIDTH),
                in_ctl: vec![false; POSEIDON2_LIMBS],
                input_indices: vec![0; POSEIDON2_LIMBS],
                out_ctl: vec![false; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
                output_indices: vec![0; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
                mmcs_index_sum_idx: 0,
                mmcs_ctl_enabled: false,
            };
            rows.resize(target_rows, filler);
        }

        let preprocessed = extract_preprocessed_from_operations::<4, 2, Val, Val>(&rows, 4, 4);
        let air = Poseidon2CircuitAirBabyBearD4Width16::new_with_preprocessed(
            constants.clone(),
            preprocessed,
        );

        let trace = air.generate_trace_rows(&rows, &constants, fri_params.log_blowup);

        type Dft = p3_dft::Radix2Bowers;
        let dft = Dft::default();

        type Pcs = TwoAdicFriPcs<Val, Dft, MyMmcs, ChallengeMmcs>;
        let pcs = Pcs::new(dft, val_mmcs, fri_params);

        type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
        let config = MyConfig::new(pcs, challenger);

        let (preprocessed_prover, preprocessed_verifier) =
            setup_preprocessed(&config, &air, degree_bits).unzip();
        let proof =
            prove_with_preprocessed(&config, &air, trace, &[], preprocessed_prover.as_ref());

        verify_with_preprocessed(&config, &air, &proof, &[], preprocessed_verifier.as_ref())
    }

    #[test]
    fn test_air_constraint_degree() {
        let mut rng = SmallRng::seed_from_u64(1);
        let constants = RoundConstants::new(rng.random(), rng.random(), rng.random());

        let air = Poseidon2CircuitAirBabyBearD4Width16::new(constants);
        p3_test_utils::assert_air_constraint_degree!(air, "Poseidon2CircuitAir");
    }

    /// Helper: set up STARK infrastructure and prove/verify a set of rows.
    fn prove_and_verify(
        rows: &[Poseidon2CircuitRow<BabyBear>],
        constants: &RoundConstants<BabyBear, WIDTH, 4, 13>,
        perm: &Poseidon2BabyBear<WIDTH>,
    ) -> Result<
        (),
        p3_uni_stark::VerificationError<
            p3_fri::verifier::FriError<
                p3_merkle_tree::MerkleTreeError,
                p3_merkle_tree::MerkleTreeError,
            >,
        >,
    > {
        let _ = perm; // used only by callers to build rows
        type Val = BabyBear;
        type Challenge = BinomialExtensionField<Val, 4>;
        type ByteHash = Keccak256Hash;
        let byte_hash = ByteHash {};
        type U64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
        let u64_hash = U64Hash::new(KeccakF {});
        type FieldHash = SerializingHasher<U64Hash>;
        let field_hash = FieldHash::new(u64_hash);
        type MyCompress = CompressionFunctionFromHasher<U64Hash, 2, 4>;
        let compress = MyCompress::new(u64_hash);
        type MyMmcs = MerkleTreeHidingMmcs<
            [Val; p3_keccak::VECTOR_LEN],
            [u64; p3_keccak::VECTOR_LEN],
            FieldHash,
            MyCompress,
            SmallRng,
            2,
            4,
            4,
        >;
        let rng = SmallRng::seed_from_u64(1);
        let val_mmcs = MyMmcs::new(field_hash, compress, 0, rng);
        type ChallengeMmcs = ExtensionMmcs<Val, Challenge, MyMmcs>;
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
        let challenger = Challenger::from_hasher(vec![], byte_hash);
        let mut fri_params = FriParameters::new_benchmark(challenge_mmcs);
        fri_params.log_blowup = 4;

        let degree_bits = 5;
        let target_rows = 1usize << degree_bits;
        let mut padded = rows.to_vec();
        if padded.len() < target_rows {
            let filler = Poseidon2CircuitRow {
                new_start: true,
                merkle_path: false,
                mmcs_bit: false,
                mmcs_index_sum: Val::ZERO,
                input_values: Val::zero_vec(WIDTH),
                in_ctl: vec![false; POSEIDON2_LIMBS],
                input_indices: vec![0; POSEIDON2_LIMBS],
                out_ctl: vec![false; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
                output_indices: vec![0; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
                mmcs_index_sum_idx: 0,
                mmcs_ctl_enabled: false,
            };
            padded.resize(target_rows, filler);
        }

        let preprocessed = extract_preprocessed_from_operations::<4, 2, Val, Val>(&padded, 4, 4);
        let air = Poseidon2CircuitAirBabyBearD4Width16::new_with_preprocessed(
            constants.clone(),
            preprocessed,
        );
        let trace = air.generate_trace_rows(&padded, constants, fri_params.log_blowup);

        type Dft = p3_dft::Radix2Bowers;
        let dft = Dft::default();
        type Pcs = TwoAdicFriPcs<Val, Dft, MyMmcs, ChallengeMmcs>;
        let pcs = Pcs::new(dft, val_mmcs, fri_params);
        type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
        let config = MyConfig::new(pcs, challenger);

        let (preprocessed_prover, preprocessed_verifier) =
            setup_preprocessed(&config, &air, degree_bits).unzip();
        let proof =
            prove_with_preprocessed(&config, &air, trace, &[], preprocessed_prover.as_ref());
        verify_with_preprocessed(&config, &air, &proof, &[], preprocessed_verifier.as_ref())
    }

    fn make_constants_and_perm(
        rng: &mut SmallRng,
    ) -> (
        RoundConstants<BabyBear, WIDTH, 4, 13>,
        Poseidon2BabyBear<WIDTH>,
    ) {
        let beginning: [[BabyBear; WIDTH]; 4] = rng.random();
        let partial: [BabyBear; 13] = rng.random();
        let ending: [[BabyBear; WIDTH]; 4] = rng.random();
        let constants = RoundConstants::new(beginning, partial, ending);
        let perm = Poseidon2BabyBear::<WIDTH>::new(
            ExternalLayerConstants::new(beginning.to_vec(), ending.to_vec()),
            partial.to_vec(),
        );
        (constants, perm)
    }

    #[test]
    fn prove_poseidon2_merkle_right() -> Result<
        (),
        p3_uni_stark::VerificationError<
            p3_fri::verifier::FriError<
                p3_merkle_tree::MerkleTreeError,
                p3_merkle_tree::MerkleTreeError,
            >,
        >,
    > {
        type Val = BabyBear;
        const D: usize = 4;
        let mut rng = SmallRng::seed_from_u64(42);
        let (constants, perm) = make_constants_and_perm(&mut rng);

        // Row A: new_start, random state
        let state_a: [Val; WIDTH] = core::array::from_fn(|_| rng.random());
        let output_a = perm.permute(state_a);

        let row_a = Poseidon2CircuitRow {
            new_start: true,
            merkle_path: false,
            mmcs_bit: false,
            mmcs_index_sum: Val::ZERO,
            input_values: state_a.to_vec(),
            in_ctl: vec![false; POSEIDON2_LIMBS],
            input_indices: vec![0; POSEIDON2_LIMBS],
            out_ctl: vec![false; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            output_indices: vec![0; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            mmcs_index_sum_idx: 0,
            mmcs_ctl_enabled: false,
        };

        // Row B: merkle mode, mmcs_bit=true (right child).
        // With mmcs_bit=1: prev output[0..D] → input[2D..3D], output[D..2D] → input[3D..4D].
        // Limbs 0-1 are sibling data (random).
        let sibling: [Val; 2 * D] = core::array::from_fn(|_| rng.random());
        let mut state_b = [Val::ZERO; WIDTH];
        state_b[0..2 * D].copy_from_slice(&sibling);
        state_b[2 * D..3 * D].copy_from_slice(&output_a[0..D]);
        state_b[3 * D..4 * D].copy_from_slice(&output_a[D..2 * D]);
        let output_b = perm.permute(state_b);

        let row_b = Poseidon2CircuitRow {
            new_start: false,
            merkle_path: true,
            mmcs_bit: true,
            mmcs_index_sum: Val::ZERO,
            input_values: state_b.to_vec(),
            in_ctl: vec![false; POSEIDON2_LIMBS],
            input_indices: vec![0; POSEIDON2_LIMBS],
            out_ctl: vec![false; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            output_indices: vec![0; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            mmcs_index_sum_idx: 0,
            mmcs_ctl_enabled: false,
        };

        // Row C: chain from output_b in sponge mode
        let row_c = Poseidon2CircuitRow {
            new_start: false,
            merkle_path: false,
            mmcs_bit: false,
            mmcs_index_sum: Val::ZERO,
            input_values: output_b.to_vec(),
            in_ctl: vec![false; POSEIDON2_LIMBS],
            input_indices: vec![0; POSEIDON2_LIMBS],
            out_ctl: vec![false; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            output_indices: vec![0; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            mmcs_index_sum_idx: 0,
            mmcs_ctl_enabled: false,
        };

        prove_and_verify(&[row_a, row_b, row_c], &constants, &perm)
    }

    #[test]
    fn prove_poseidon2_mmcs_accumulator() -> Result<
        (),
        p3_uni_stark::VerificationError<
            p3_fri::verifier::FriError<
                p3_merkle_tree::MerkleTreeError,
                p3_merkle_tree::MerkleTreeError,
            >,
        >,
    > {
        type Val = BabyBear;
        const D: usize = 4;
        let mut rng = SmallRng::seed_from_u64(99);
        let (constants, perm) = make_constants_and_perm(&mut rng);

        // Build a 3-row Merkle chain that exercises the MMCS accumulator:
        //   Row 0: new_start=true,  merkle, mmcs_bit=1 → mmcs_index_sum = 0 (reset)
        //   Row 1: new_start=false, merkle, mmcs_bit=0 → mmcs_index_sum = 0*2+0 = 0
        //   Row 2: new_start=false, merkle, mmcs_bit=1 → mmcs_index_sum = 0*2+1 = 1
        // Expected final accumulator = 1.
        let bits = [true, false, true];

        let state_0: [Val; WIDTH] = core::array::from_fn(|_| rng.random());
        let output_0 = perm.permute(state_0);
        let row_0 = Poseidon2CircuitRow {
            new_start: true,
            merkle_path: true,
            mmcs_bit: bits[0],
            mmcs_index_sum: Val::ZERO,
            input_values: state_0.to_vec(),
            in_ctl: vec![false; POSEIDON2_LIMBS],
            input_indices: vec![0; POSEIDON2_LIMBS],
            out_ctl: vec![false; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            output_indices: vec![0; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            mmcs_index_sum_idx: 0,
            mmcs_ctl_enabled: true,
        };

        // Row 1: left child (mmcs_bit=0), chain output[0..2D] → input[0..2D]
        let mut state_1 = [Val::ZERO; WIDTH];
        state_1[0..D].copy_from_slice(&output_0[0..D]);
        state_1[D..2 * D].copy_from_slice(&output_0[D..2 * D]);
        let output_1 = perm.permute(state_1);
        let row_1 = Poseidon2CircuitRow {
            new_start: false,
            merkle_path: true,
            mmcs_bit: bits[1],
            mmcs_index_sum: Val::ZERO, // will be computed by generate_trace_rows
            input_values: state_1.to_vec(),
            in_ctl: vec![false; POSEIDON2_LIMBS],
            input_indices: vec![0; POSEIDON2_LIMBS],
            out_ctl: vec![false; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            output_indices: vec![0; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            mmcs_index_sum_idx: 0,
            mmcs_ctl_enabled: true,
        };

        // Row 2: right child (mmcs_bit=1), chain output[0..D]→input[2D..3D], output[D..2D]→input[3D..4D]
        let sibling: [Val; 2 * D] = core::array::from_fn(|_| rng.random());
        let mut state_2 = [Val::ZERO; WIDTH];
        state_2[0..2 * D].copy_from_slice(&sibling);
        state_2[2 * D..3 * D].copy_from_slice(&output_1[0..D]);
        state_2[3 * D..4 * D].copy_from_slice(&output_1[D..2 * D]);
        let _output_2 = perm.permute(state_2);
        let row_2 = Poseidon2CircuitRow {
            new_start: false,
            merkle_path: true,
            mmcs_bit: bits[2],
            mmcs_index_sum: Val::ZERO,
            input_values: state_2.to_vec(),
            in_ctl: vec![false; POSEIDON2_LIMBS],
            input_indices: vec![0; POSEIDON2_LIMBS],
            out_ctl: vec![false; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            output_indices: vec![0; POSEIDON2_PUBLIC_OUTPUT_LIMBS],
            mmcs_index_sum_idx: 0,
            mmcs_ctl_enabled: true,
        };

        prove_and_verify(&[row_0, row_1, row_2], &constants, &perm)
    }

    /// Build an AIR with the given preprocessed data and optional minimum
    /// height, then return the materialized preprocessed trace matrix.
    fn build_preprocessed_trace(
        preprocessed: Vec<BabyBear>,
        min_height: usize,
    ) -> RowMajorMatrix<BabyBear> {
        let mut rng = SmallRng::seed_from_u64(1);
        let constants = RoundConstants::new(rng.random(), rng.random(), rng.random());

        Poseidon2CircuitAirBabyBearD4Width16::new_with_preprocessed(constants, preprocessed)
            .with_min_height(min_height)
            .preprocessed_trace()
            .expect("preprocessed_trace returned None")
    }

    /// The preprocessed width for BabyBear width-16 is 24 columns.
    ///
    /// 4 input limbs × 4 fields each  = 16
    /// 2 output limbs × 2 fields each =  4
    /// 4 scalar fields                =  4
    ///                           total = 24
    const PREP_WIDTH: usize = 24;

    #[test]
    fn preprocessed_trace_pads_to_power_of_two() {
        // Feed 3 rows of data. That's 3 × 24 = 72 field elements.
        //
        // 3 is not a power of two, so the method must round up to 4 rows.
        let three_rows = vec![BabyBear::ONE; 3 * PREP_WIDTH];
        let trace = build_preprocessed_trace(three_rows, 1);

        // The matrix should have 4 rows and 24 columns.
        assert_eq!(trace.height(), 4);
        assert_eq!(trace.width(), PREP_WIDTH);
    }

    #[test]
    fn preprocessed_trace_respects_min_height() {
        // Feed 2 rows of data (already a power of two).
        //
        // But request a minimum height of 8.
        //
        // The result must have 8 rows, not 2.
        let two_rows = vec![BabyBear::ONE; 2 * PREP_WIDTH];
        let trace = build_preprocessed_trace(two_rows, 8);

        assert_eq!(trace.height(), 8);
        assert_eq!(trace.width(), PREP_WIDTH);
    }

    #[test]
    fn preprocessed_trace_preserves_original_data() {
        // Fill one row with all-twos.
        //
        // After padding the result must still have those values in the
        // first row.
        let one_row = vec![BabyBear::TWO; PREP_WIDTH];
        let trace = build_preprocessed_trace(one_row.clone(), 1);

        // The first row should be exactly what we put in.
        let values = trace.values.as_slice();
        assert_eq!(&values[..PREP_WIDTH], &one_row[..]);
    }

    #[test]
    fn preprocessed_trace_sets_chain_boundary_on_first_padding_row() {
        // Feed 3 rows. The method pads to 4 rows (next power of two).
        //
        // The first padding row (row index 3) must have its chain-start
        // flag set to one. That flag is the second-to-last column.
        //
        // This prevents the chaining constraint from connecting the
        // last real row to the first padding row.
        let three_rows = vec![BabyBear::ZERO; 3 * PREP_WIDTH];
        let trace = build_preprocessed_trace(three_rows, 1);

        assert_eq!(trace.height(), 4);

        // Row 3 (first padding row): second-to-last column = 1.
        let values = trace.values.as_slice();
        let padding_row = &values[3 * PREP_WIDTH..4 * PREP_WIDTH];
        let chain_start_flag = padding_row[PREP_WIDTH - 2];
        assert_eq!(chain_start_flag, BabyBear::ONE);

        // All other columns in the padding row should be zero.
        for (i, &val) in padding_row.iter().enumerate() {
            if i != PREP_WIDTH - 2 {
                assert_eq!(val, BabyBear::ZERO, "padding row column {i} should be zero");
            }
        }
    }

    #[test]
    fn preprocessed_trace_no_padding_when_exact_power_of_two() {
        // Feed exactly 4 rows. Already a power of two.
        //
        // No padding should occur, so no chain-boundary flag is set on
        // any extra row.
        let four_rows = vec![BabyBear::ONE; 4 * PREP_WIDTH];
        let trace = build_preprocessed_trace(four_rows, 1);

        // Height should be exactly 4 — no extra rows.
        assert_eq!(trace.height(), 4);

        // All 4 rows should contain the original data (all ones).
        let values = trace.values.as_slice();
        for row_idx in 0..4 {
            let start = row_idx * PREP_WIDTH;
            let row = &values[start..start + PREP_WIDTH];
            assert!(
                row.iter().all(|&v| v == BabyBear::ONE),
                "row {row_idx} should be all ones"
            );
        }
    }

    #[test]
    fn preprocessed_trace_padding_rows_beyond_first_are_all_zero() {
        // Feed 1 row. Request minimum height of 8.
        //
        // Rows 1..8 are padding. Row 1 has the chain-boundary flag.
        // Rows 2..8 should be entirely zero.
        let one_row = vec![BabyBear::ONE; PREP_WIDTH];
        let trace = build_preprocessed_trace(one_row, 8);

        assert_eq!(trace.height(), 8);

        // Rows 2 through 7 should be completely zero.
        let values = trace.values.as_slice();
        for row_idx in 2..8 {
            let start = row_idx * PREP_WIDTH;
            let row = &values[start..start + PREP_WIDTH];
            assert!(
                row.iter().all(|&v| v == BabyBear::ZERO),
                "padding row {row_idx} should be all zeros"
            );
        }
    }
}
