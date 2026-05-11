//! # ALU AIR
//!
//! [`AluAir`] defines the unified AIR for proving arithmetic operations over both
//! base and extension fields.
//!
//! ## Operations
//!
//! Each row encodes one or more arithmetic constraints selected by preprocessed
//! selectors:
//!
//! | Operation      | Constraint                                 | Degree |
//! |----------------|--------------------------------------------|--------|
//! | **ADD**        | `a + b - out = 0`                          | 1      |
//! | **MUL**        | `a * b - out = 0`                          | 2      |
//! | **BOOL_CHECK** | `a * (a - 1) = 0`                          | 2      |
//! | **MUL_ADD**    | `a * b + c - out = 0`                      | 2      |
//! | **HORNER_ACC** | `prev_row_out * b + c - a - out = 0`       | 2      |
//!
//! All constraint degrees are ≤ 3 (after multiplying by a selector), compatible
//! with `log_blowup = 1`.
//!
//! ## Main trace layout
//!
//! Each lane occupies `4 * D` columns: `[a[D], b[D], c[D], out[D]]`.
//! After all lanes, there are `(num_int + 2 * (K_max - 1)) * D` **global** extra columns
//! on lane 0 for variable-arity packed HornerAcc (`K_max >= 2`, fixed per AIR):
//! `num_int = (K_max - 1) / 2` compressed intermediates `int_*[D]`, then `(a_t,c_t)` for
//! `t = 1..K_max-1` (operand pairs for packed steps after the first).
//!
//! Total main width = `lanes * 4D + (num_int + 2 * (K_max - 1) + 1) * D` (extra `+ D` is `b^2`
//! witness for lane 0, keeping packed Horner constraints at degree 3 with preprocessed selectors).
//!
//! ## Preprocessed trace layout
//!
//! Each lane occupies 13 columns (see [`AluPrepLaneCols`](super::alu_columns::AluPrepLaneCols)):
//!
//! | Offset | Name              | Purpose                                        |
//! |--------|-------------------|------------------------------------------------|
//! | 0      | `mult_a`          | Multiplicity for `a` (`-1` = reader, `0` = pad)|
//! | 1      | `sel_add_vs_mul`  | ADD selector                                   |
//! | 2      | `sel_bool`        | BOOL_CHECK selector                            |
//! | 3      | `sel_muladd`      | MUL_ADD selector                               |
//! | 4      | `sel_horner`      | HORNER_ACC selector                            |
//! | 5      | `a_idx`           | Witness index for `a` (D-scaled)               |
//! | 6      | `b_idx`           | Witness index for `b` (D-scaled)               |
//! | 7      | `c_idx`           | Witness index for `c` (D-scaled)               |
//! | 8      | `out_idx`         | Witness index for `out` (D-scaled)             |
//! | 9      | `mult_b`          | Multiplicity for `b`                           |
//! | 10     | `mult_out`        | Multiplicity for `out`                         |
//! | 11     | `a_is_reader`     | 1 if `a` reads from the WitnessChecks bus      |
//! | 12     | `c_is_reader`     | 1 if `c` reads from the WitnessChecks bus      |
//!
//! After all lanes, there are `(K_max - 1) + 4 * (K_max - 1)` **global** extra preprocessed
//! columns: `sel_k` for each arity `k = 2..K_max`, then for each step `t = 1..K_max-1` four
//! columns `a_t_idx`, `c_t_idx`, `a_t_reader`, `c_t_reader`
//! (see [`AluPackedHornerStepPrepCols`](super::alu_columns::AluPackedHornerStepPrepCols)).
//!
//! Total preprocessed width = `lanes * 13 + (K_max - 1) + 6 * (K_max - 1)`.
//!
//! ## K-step packed HornerAcc
//!
//! When HornerAcc operations are present, [`compute_schedule`] places Horner chains
//! on lane 0 and greedily packs each prefix of a chain into [`ScheduleEntry::PackedHorner`]
//! with arity `k ∈ {2..K_max}` (same `b` witness index, contiguous indices). Remainder ops
//! use single-step rows.
//!
//! Inter-row and intra-row constraints fold consecutive Horner steps in pairs (degree-3
//! where needed); see `eval` implementation for the exact selector layout per `k`.
//!
//! A leading [`ScheduleEntry::Separator`] prevents bogus inter-row Horner on row 0.
//!
//! ## WitnessChecks bus
//!
//! Each lane contributes 4 lookups. Packed Horner adds `2 * (K - 1)` extra lookups
//! for `(a_t, c_t)`, `t = 1..K-1`. Total = `lanes * 4 + 2 * (K - 1)`.

#![allow(clippy::needless_range_loop)]

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::borrow::{Borrow, BorrowMut};
use core::marker::PhantomData;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_circuit::tables::AluTrace;
use p3_field::{BasedVectorSpace, Dup, Field, PrimeCharacteristicRing};
use p3_lookup::LookupAir;
use p3_lookup::lookup_traits::{Direction, Kind, Lookup};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::SymbolicExpression;
use tracing::instrument;

use super::alu_columns::{
    AluMainLaneCols, AluPackedHornerStepPrepCols, AluPrepLaneCols, PACKED_HORNER_STEP_PREP_WIDTH,
    PREP_LANE_WIDTH, alu_main_lane_width, extra_prep_a_idx_for_step, extra_prep_sel_k_idx,
    horner_extra_prep_width, num_horner_intermediates,
};
use crate::air::utils::{create_symbolic_variables, get_alu_index_lookups};

/// Entry in the HornerAcc lane schedule.
#[derive(Debug, Clone, Copy)]
enum ScheduleEntry {
    /// A real ALU op at the given original index.
    Op(usize),
    /// `k` consecutive HornerAcc ops with indices `first..first+k` in the original trace (`k >= 2`).
    PackedHorner(usize, usize),
    /// A virtual zero-separator (multiplicity 0, all values 0).
    Separator,
}

/// How extension multiplication is reduced in the MUL / MUL_ADD / Horner paths.
#[derive(Clone, Copy, Debug)]
pub(crate) enum AluExtMulKind<F: Copy> {
    /// Base field only (`D == 1`).
    Base,
    /// Binomial extension `x^D = W` for `D > 1`.
    Binomial { w: F },
    /// Quintic trinomial `X^5 + X^2 - 1` (KoalaBear-style), `D == 5` only.
    QuinticTrinomial,
}

/// AIR for proving unified arithmetic operations.
///
/// Supports ADD, MUL, BOOL_CHECK, and MUL_ADD operations with preprocessed selectors.
#[derive(Debug, Clone)]
pub struct AluAir<F: Copy, const D: usize = 1> {
    /// Total number of logical ALU operations in the trace.
    pub(crate) num_ops: usize,
    /// Number of independent operations packed per trace row.
    pub(crate) lanes: usize,
    pub(crate) ext_mul_kind: AluExtMulKind<F>,
    /// Flattened preprocessed values (selectors + indices), in original op order.
    pub(crate) preprocessed: Vec<F>,
    /// Number of lookup columns registered so far.
    pub(crate) num_lookup_columns: usize,
    /// Minimum trace height (for FRI compatibility with higher log_final_poly_len).
    pub(crate) min_height: usize,
    /// HornerAcc lane schedule. When present, ops are reordered so that HornerAcc
    /// chains occupy lane 0 in consecutive rows, with zero-separators between chains.
    schedule: Option<Vec<ScheduleEntry>>,
    /// Pack size K for [`ScheduleEntry::PackedHorner`] (>= 2).
    pub(crate) horner_packed_steps: usize,
    _phantom: PhantomData<F>,
}

impl<F: Field + PrimeCharacteristicRing + Copy, const D: usize> AluAir<F, D> {
    /// Construct a new `AluAir` for base-field operations (D=1).
    pub const fn new(num_ops: usize, lanes: usize) -> Self {
        assert!(lanes > 0, "lane count must be non-zero");
        assert!(
            D == 1,
            "Base-field constructor requires D == 1; use new_binomial or new_quintic_trinomial"
        );
        Self {
            num_ops,
            lanes,
            ext_mul_kind: AluExtMulKind::Base,
            preprocessed: Vec::new(),
            num_lookup_columns: 0,
            min_height: 1,
            schedule: None,
            horner_packed_steps: 2,
            _phantom: PhantomData,
        }
    }

    /// Construct a new `AluAir` for base-field operations with preprocessed data.
    pub fn new_with_preprocessed(
        num_ops: usize,
        lanes: usize,
        preprocessed: Vec<F>,
        horner_packed_steps: usize,
    ) -> Self {
        assert!(lanes > 0, "lane count must be non-zero");
        assert!(
            D == 1,
            "Base-field constructor requires D == 1; use new_binomial_with_preprocessed or new_quintic_trinomial_with_preprocessed"
        );
        assert!(
            horner_packed_steps >= 2,
            "horner_packed_steps must be at least 2"
        );
        let schedule = Self::compute_schedule(&preprocessed, lanes, horner_packed_steps);
        Self {
            num_ops,
            lanes,
            ext_mul_kind: AluExtMulKind::Base,
            preprocessed,
            num_lookup_columns: 0,
            min_height: 1,
            schedule,
            horner_packed_steps,
            _phantom: PhantomData,
        }
    }

    /// Construct a new `AluAir` for binomial extension-field operations (D > 1).
    pub const fn new_binomial(num_ops: usize, lanes: usize, w: F) -> Self {
        assert!(lanes > 0, "lane count must be non-zero");
        assert!(D >= 2, "Binomial constructor requires D >= 2");
        Self {
            num_ops,
            lanes,
            ext_mul_kind: AluExtMulKind::Binomial { w },
            preprocessed: Vec::new(),
            num_lookup_columns: 0,
            min_height: 1,
            schedule: None,
            horner_packed_steps: 2,
            _phantom: PhantomData,
        }
    }

    /// Construct a new `AluAir` for binomial extension-field operations with preprocessed data.
    pub fn new_binomial_with_preprocessed(
        num_ops: usize,
        lanes: usize,
        w: F,
        preprocessed: Vec<F>,
        horner_packed_steps: usize,
    ) -> Self {
        assert!(lanes > 0, "lane count must be non-zero");
        assert!(D >= 2, "Binomial constructor requires D >= 2");
        assert!(
            horner_packed_steps >= 2,
            "horner_packed_steps must be at least 2"
        );
        let schedule = Self::compute_schedule(&preprocessed, lanes, horner_packed_steps);
        Self {
            num_ops,
            lanes,
            ext_mul_kind: AluExtMulKind::Binomial { w },
            preprocessed,
            num_lookup_columns: 0,
            min_height: 1,
            schedule,
            horner_packed_steps,
            _phantom: PhantomData,
        }
    }

    /// Quintic trinomial extension (`X^5 + X^2 - 1`), `D = 5` only.
    pub const fn new_quintic_trinomial(num_ops: usize, lanes: usize) -> Self {
        assert!(lanes > 0, "lane count must be non-zero");
        assert!(D == 5, "Quintic trinomial ALU requires D = 5");
        Self {
            num_ops,
            lanes,
            ext_mul_kind: AluExtMulKind::QuinticTrinomial,
            preprocessed: Vec::new(),
            num_lookup_columns: 0,
            min_height: 1,
            schedule: None,
            horner_packed_steps: 2,
            _phantom: PhantomData,
        }
    }

    /// Quintic trinomial extension with preprocessed columns, `D = 5` only.
    pub fn new_quintic_trinomial_with_preprocessed(
        num_ops: usize,
        lanes: usize,
        preprocessed: Vec<F>,
        horner_packed_steps: usize,
    ) -> Self {
        assert!(lanes > 0, "lane count must be non-zero");
        assert!(D == 5, "Quintic trinomial ALU requires D = 5");
        assert!(
            horner_packed_steps >= 2,
            "horner_packed_steps must be at least 2"
        );

        let schedule = Self::compute_schedule(&preprocessed, lanes, horner_packed_steps);
        Self {
            num_ops,
            lanes,
            ext_mul_kind: AluExtMulKind::QuinticTrinomial,
            preprocessed,
            num_lookup_columns: 0,
            min_height: 1,
            schedule,
            horner_packed_steps,
            _phantom: PhantomData,
        }
    }

    /// Set the minimum trace height for FRI compatibility.
    ///
    /// FRI requires: `log_trace_height > log_final_poly_len + log_blowup`
    /// So `min_height` should be >= `2^(log_final_poly_len + log_blowup + 1)`.
    pub const fn with_min_height(mut self, min_height: usize) -> Self {
        self.min_height = min_height;
        self
    }

    /// Override packed Horner chain length (default 2 from [`Self::new`] / [`Self::new_binomial`]).
    ///
    /// Batch verification rebuilds a symbolic ALU without committed preprocessed data; this must
    /// match [`crate::batch_stark_prover::TablePacking::horner_packed_steps`] from the proof.
    pub const fn with_horner_pack_k(mut self, k: usize) -> Self {
        assert!(k >= 2, "horner_packed_steps must be at least 2");
        self.horner_packed_steps = k;
        self
    }

    /// Number of main columns per lane: a[D], b[D], c[D], out[D]
    pub const fn lane_width() -> usize {
        alu_main_lane_width::<D>()
    }

    /// Total main trace width for this AIR instance.
    pub const fn total_width(&self) -> usize {
        let k = self.horner_packed_steps;
        let num_int = num_horner_intermediates(k);
        let extra = (num_int + 2 * (k - 1) + 1) * D;
        self.lanes * Self::lane_width() + extra
    }

    /// Number of preprocessed columns per lane (see [`AluPrepLaneCols`](super::alu_columns::AluPrepLaneCols)).
    pub const fn preprocessed_lane_width() -> usize {
        PREP_LANE_WIDTH
    }

    /// Total preprocessed width: per-lane base columns plus global packed Horner columns.
    pub const fn preprocessed_width(&self) -> usize {
        self.lanes * PREP_LANE_WIDTH + horner_extra_prep_width(self.horner_packed_steps)
    }

    /// Total entries in the scheduled trace (including separators).
    pub fn scheduled_entry_count(&self) -> usize {
        self.schedule.as_ref().map_or(self.num_ops, |s| s.len())
    }

    /// Compute a lane schedule that places HornerAcc chains in lane 0.
    ///
    /// Returns `None` if no HornerAcc ops are present.
    /// Even with `lanes == 1`, scheduling is required: chains must start at
    /// row 0 so the cyclic wrap from the last (zero-padded) row provides
    /// `prev_out = 0`, and separators must appear between chains.
    #[instrument(skip_all, name = "AluAir::compute_schedule")]
    fn compute_schedule(
        preprocessed: &[F],
        lanes: usize,
        pack_k: usize,
    ) -> Option<Vec<ScheduleEntry>> {
        let plw = PREP_LANE_WIDTH;
        let num_ops = preprocessed.len() / plw;
        if num_ops == 0 {
            return None;
        }

        let is_horner: Vec<bool> = (0..num_ops)
            .map(|i| {
                let prep: &AluPrepLaneCols<F> = preprocessed[i * plw..(i + 1) * plw].borrow();
                prep.sel_horner == F::ONE
            })
            .collect();

        if !is_horner.iter().any(|&h| h) {
            return None;
        }

        // Find maximal runs of consecutive HornerAcc ops (chains)
        let mut chains: Vec<Vec<usize>> = Vec::new();
        let mut current_chain: Vec<usize> = Vec::new();
        let mut non_chain: Vec<usize> = Vec::new();

        for (i, &h) in is_horner.iter().enumerate() {
            if h {
                current_chain.push(i);
            } else {
                if !current_chain.is_empty() {
                    chains.push(core::mem::take(&mut current_chain));
                }
                non_chain.push(i);
            }
        }
        if !current_chain.is_empty() {
            chains.push(current_chain);
        }

        let mut schedule: Vec<ScheduleEntry> = Vec::new();
        let mut nc = 0; // cursor into non_chain

        // Helper: fill remaining slots in current row with non-chain ops or separators
        let fill_row = |schedule: &mut Vec<ScheduleEntry>, nc: &mut usize, non_chain: &[usize]| {
            while !schedule.len().is_multiple_of(lanes) {
                if *nc < non_chain.len() {
                    schedule.push(ScheduleEntry::Op(non_chain[*nc]));
                    *nc += 1;
                } else {
                    schedule.push(ScheduleEntry::Separator);
                }
            }
        };

        // Leading separator before the first chain. The cyclic constraint
        // wraps from the last row back to row 0, and without this separator
        // the first chain would be a Horner row with a packed-arity selector set,
        schedule.push(ScheduleEntry::Separator);
        fill_row(&mut schedule, &mut nc, &non_chain);

        for (chain_idx, chain) in chains.iter().enumerate() {
            if chain_idx > 0 {
                // Complete previous row
                fill_row(&mut schedule, &mut nc, &non_chain);
                // Separator row: lane 0 = zero, other lanes = non-chain or zero
                schedule.push(ScheduleEntry::Separator);
                fill_row(&mut schedule, &mut nc, &non_chain);
            }

            let mut i = 0;
            while i < chain.len() {
                debug_assert_eq!(schedule.len() % lanes, 0, "chain op not at lane 0");
                let k_try = chain.len().saturating_sub(i).min(pack_k);
                let mut best_k = 1usize;
                for k in (2..=k_try).rev() {
                    let mut contiguous = true;
                    for j in 1..k {
                        if chain[i + j] != chain[i] + j {
                            contiguous = false;
                            break;
                        }
                    }
                    if contiguous
                        && Self::horner_ops_share_b_idx(preprocessed, plw, &chain[i..i + k])
                    {
                        best_k = k;
                        break;
                    }
                }
                if best_k >= 2 {
                    schedule.push(ScheduleEntry::PackedHorner(chain[i], best_k));
                    i += best_k;
                } else {
                    schedule.push(ScheduleEntry::Op(chain[i]));
                    i += 1;
                }
                fill_row(&mut schedule, &mut nc, &non_chain);
            }
        }

        // Complete last chain row
        fill_row(&mut schedule, &mut nc, &non_chain);

        // Remaining non-chain ops fill all lanes
        while nc < non_chain.len() {
            schedule.push(ScheduleEntry::Op(non_chain[nc]));
            nc += 1;
        }
        // Pad final row
        fill_row(&mut schedule, &mut nc, &non_chain);

        Some(schedule)
    }

    /// All ops in `op_indices` use the same `b` witness index in preprocessed data.
    fn horner_ops_share_b_idx(preprocessed: &[F], plw: usize, op_indices: &[usize]) -> bool {
        if op_indices.is_empty() {
            return true;
        }
        let prep0: &AluPrepLaneCols<F> =
            preprocessed[op_indices[0] * plw..(op_indices[0] + 1) * plw].borrow();
        let b0 = prep0.b_idx;
        op_indices.iter().all(|&idx| {
            let p: &AluPrepLaneCols<F> = preprocessed[idx * plw..(idx + 1) * plw].borrow();
            p.b_idx == b0
        })
    }

    /// Write the 4 operands `[a, b, c, out]` of operation `op_idx` into `dst`
    /// starting at `cursor`, advancing it by `4 * D`.
    #[inline(always)]
    fn write_operands<ExtF: BasedVectorSpace<F>>(
        dst: &mut [F],
        cursor: &mut usize,
        trace: &AluTrace<ExtF>,
        op_idx: usize,
    ) {
        for operand in 0..4 {
            let coeffs = trace.values[op_idx][operand].as_basis_coefficients_slice();
            dst[*cursor..*cursor + D].copy_from_slice(coeffs);
            *cursor += D;
        }
    }

    /// Convert an `AluTrace` into a `RowMajorMatrix` suitable for the STARK prover.
    #[instrument(skip_all, name = "AluAir::trace_to_matrix")]
    pub fn trace_to_matrix<ExtF: BasedVectorSpace<F> + Field>(
        &self,
        trace: &AluTrace<ExtF>,
        min_height: usize,
    ) -> RowMajorMatrix<F> {
        let lanes = self.lanes;
        assert!(lanes > 0, "lane count must be non-zero");

        let lane_width = Self::lane_width();
        let width = self.total_width();
        let entry_count = self.scheduled_entry_count();
        let row_count = entry_count.div_ceil(lanes);

        let mut values = F::zero_vec(width * row_count.max(1));

        if let Some(ref schedule) = self.schedule {
            let mut prev_lane0_out = [F::ZERO; D];
            for (pos, entry) in schedule.iter().enumerate() {
                let row = pos / lanes;
                let lane = pos % lanes;

                match entry {
                    ScheduleEntry::Op(i) => {
                        let mut cursor = row * width + lane * lane_width;
                        Self::write_operands(&mut values, &mut cursor, trace, *i);
                        if lane == 0 {
                            let out_start = row * width + 3 * D;
                            prev_lane0_out[..D].copy_from_slice(&values[out_start..out_start + D]);
                        }
                    }
                    ScheduleEntry::PackedHorner(first_idx, actual_k) => {
                        let k = *actual_k;
                        let k_max = self.horner_packed_steps;
                        let base = row * width + lane * lane_width;
                        let mut cursor = base;

                        for operand in 0..3 {
                            let coeffs =
                                trace.values[*first_idx][operand].as_basis_coefficients_slice();
                            values[cursor..cursor + D].copy_from_slice(coeffs);
                            cursor += D;
                        }
                        let last = first_idx + k - 1;
                        let out_last = trace.values[last][3].as_basis_coefficients_slice();
                        values[cursor..cursor + D].copy_from_slice(out_last);

                        if lane == 0 {
                            let extra = row * width + self.lanes * lane_width;
                            let num_int = num_horner_intermediates(k_max);
                            let prev_ext =
                                ExtF::from_basis_coefficients_slice(&prev_lane0_out[..D]).unwrap();
                            let b = trace.values[*first_idx][1];
                            let mut step = 0usize;
                            let mut acc = prev_ext;
                            for s in 0..num_int {
                                let i0 = *first_idx + step;
                                let i1 = *first_idx + step + 1;
                                let v0 = &trace.values[i0];
                                if i1 < *first_idx + k {
                                    let v1 = &trace.values[i1];
                                    let o0 = acc * b + v0[2] - v0[0];
                                    acc = o0 * b + v1[2] - v1[0];
                                    step += 2;
                                } else {
                                    acc = acc * b + v0[2] - v0[0];
                                    step += 1;
                                }
                                let off = extra + s * D;
                                values[off..off + D]
                                    .copy_from_slice(acc.as_basis_coefficients_slice());
                            }
                            let ac_base = extra + num_int * D;
                            for t in 1..k {
                                let op_t = *first_idx + t;
                                let a_t = trace.values[op_t][0].as_basis_coefficients_slice();
                                let c_t = trace.values[op_t][2].as_basis_coefficients_slice();
                                let off = ac_base + 2 * (t - 1) * D;
                                values[off..off + D].copy_from_slice(a_t);
                                values[off + D..off + 2 * D].copy_from_slice(c_t);
                            }
                            let b_sq_ext = b * b;
                            let b_sq_base = ac_base + 2 * (k_max - 1) * D;
                            values[b_sq_base..b_sq_base + D]
                                .copy_from_slice(b_sq_ext.as_basis_coefficients_slice());
                            let out_start = row * width + 3 * D;
                            prev_lane0_out[..D].copy_from_slice(&values[out_start..out_start + D]);
                        }
                    }
                    ScheduleEntry::Separator => {
                        if lane == 0 {
                            prev_lane0_out = [F::ZERO; D];
                        }
                    }
                }
            }
        } else {
            for op_idx in 0..trace.values.len() {
                let row = op_idx / lanes;
                let lane = op_idx % lanes;
                let mut cursor = row * width + lane * lane_width;
                Self::write_operands(&mut values, &mut cursor, trace, op_idx);
            }
        }

        let mut mat = RowMajorMatrix::new(values, width);
        mat.pad_to_min_power_of_two_height(
            core::cmp::max(min_height, mat.height().next_power_of_two()),
            F::ZERO,
        );

        mat
    }

    /// Build the preprocessed trace matrix with HornerAcc scheduling applied.
    ///
    /// Separator entries get multiplicity=0 (no lookups), all selectors/indices=0.
    fn build_scheduled_preprocessed_trace(&self, schedule: &[ScheduleEntry]) -> RowMajorMatrix<F> {
        let plw = PREP_LANE_WIDTH;
        let row_count = schedule.len().div_ceil(self.lanes);
        let row_width = self.preprocessed_width();

        let mut values = F::zero_vec(row_count.max(1) * row_width);

        for (pos, entry) in schedule.iter().enumerate() {
            let row = pos / self.lanes;
            let lane = pos % self.lanes;
            let base = row * row_width + lane * plw;

            match entry {
                ScheduleEntry::Op(i) => {
                    let src = &self.preprocessed[i * plw..(i + 1) * plw];
                    values[base..base + plw].copy_from_slice(src);
                }
                ScheduleEntry::PackedHorner(first_idx, actual_k) => {
                    if lane == 0 {
                        let k = *actual_k;
                        let k_max = self.horner_packed_steps;
                        let src0 = &self.preprocessed[*first_idx * plw..(*first_idx + 1) * plw];
                        let last = *first_idx + k - 1;
                        let src_last = &self.preprocessed[last * plw..(last + 1) * plw];

                        values[base..base + plw].copy_from_slice(src0);

                        let mult_a_lane = {
                            let lane_prep: &mut AluPrepLaneCols<F> =
                                values[base..base + plw].borrow_mut();
                            let src_last_prep: &AluPrepLaneCols<F> = src_last.borrow();
                            lane_prep.out_idx = src_last_prep.out_idx;
                            lane_prep.mult_out = src_last_prep.mult_out;

                            lane_prep.mult_b *= F::from_usize(k);
                            lane_prep.mult_a
                        };

                        let extra_base = row * row_width + self.lanes * plw;
                        values[extra_base + extra_prep_sel_k_idx(k)] = F::ONE;
                        for t in 1..k {
                            let src_t: &AluPrepLaneCols<F> = self.preprocessed
                                [(*first_idx + t) * plw..(*first_idx + t + 1) * plw]
                                .borrow();
                            let p = extra_base + extra_prep_a_idx_for_step(t, k_max);
                            let step: &mut AluPackedHornerStepPrepCols<F> =
                                values[p..p + PACKED_HORNER_STEP_PREP_WIDTH].borrow_mut();
                            step.a_idx = src_t.a_idx;
                            step.c_idx = src_t.c_idx;
                            step.a_reader = src_t.a_is_reader;
                            step.c_reader = src_t.c_is_reader;
                            let on = if t < k { F::ONE } else { F::ZERO };
                            step.horner_lookup_mult_a = mult_a_lane * src_t.a_is_reader * on;
                            step.horner_lookup_mult_c = mult_a_lane * src_t.c_is_reader * on;
                        }
                    }
                }
                ScheduleEntry::Separator => {}
            }
        }

        let mut mat = RowMajorMatrix::new(values, row_width);
        mat.pad_to_min_power_of_two_height(self.min_height, F::ZERO);
        mat
    }
}

impl<F: Field + Copy, const D: usize> BaseAir<F> for AluAir<F, D> {
    fn width(&self) -> usize {
        self.total_width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.schedule.as_ref().map_or_else(
            || {
                // No Horner scheduling: build the preprocessed trace at the
                // base width, then widen with zero columns for scheduling slots.
                let base_width = self.lanes * PREP_LANE_WIDTH;
                let mut mat = RowMajorMatrix::from_flat_padded(
                    self.preprocessed.to_vec(),
                    base_width,
                    F::ZERO,
                );
                mat.widen_right(horner_extra_prep_width(self.horner_packed_steps), F::ZERO);
                mat.pad_to_min_power_of_two_height(self.min_height, F::ZERO);
                Some(mat)
            },
            |schedule| Some(self.build_scheduled_preprocessed_trace(schedule)),
        )
    }
}

/// Compute `x * y` as a D-coefficient extension-field product, where
/// `w` is the binomial parameter (only used when `D > 1`).
///
/// When `D == 1` the `w` parameter is unused and all loops degenerate to a
/// single scalar multiply, so this is zero-cost for base-field AIRs.
#[inline]
fn ext_mul_binomial<AB: AirBuilder, const D: usize>(
    x: &[AB::Var],
    y: &[AB::Var],
    w: &Option<AB::Expr>,
) -> Vec<AB::Expr> {
    let mut acc = vec![AB::Expr::ZERO; D];
    for i in 0..D {
        for j in 0..D {
            let term = x[i] * y[j];
            let k = i + j;
            if k < D {
                acc[k] = acc[k].dup() + term;
            } else {
                acc[k - D] = acc[k - D].dup() + w.as_ref().unwrap().dup() * term;
            }
        }
    }
    acc
}

/// Extension multiply in `GF(p)[X]/(X^5 + X^2 - 1)` (KoalaBear-style), on `AB::Expr`.
#[inline]
fn ext_mul_quintic_trinomial<AB: AirBuilder>(x: &[AB::Var], y: &[AB::Var]) -> Vec<AB::Expr> {
    debug_assert_eq!(x.len(), 5);
    debug_assert_eq!(y.len(), 5);
    let xi = |i: usize| AB::Expr::from(x[i]);
    let yj = |j: usize| AB::Expr::from(y[j]);

    let c0 = xi(0) * yj(0);
    let c1 = xi(0) * yj(1) + xi(1) * yj(0);
    let c2 = xi(0) * yj(2) + xi(1) * yj(1) + xi(2) * yj(0);
    let c3 = xi(0) * yj(3) + xi(1) * yj(2) + xi(2) * yj(1) + xi(3) * yj(0);
    let c4 = xi(0) * yj(4) + xi(1) * yj(3) + xi(2) * yj(2) + xi(3) * yj(1) + xi(4) * yj(0);
    let c5 = xi(1) * yj(4) + xi(2) * yj(3) + xi(3) * yj(2) + xi(4) * yj(1);
    let c6 = xi(2) * yj(4) + xi(3) * yj(3) + xi(4) * yj(2);
    let c7 = xi(3) * yj(4) + xi(4) * yj(3);
    let c8 = xi(4) * yj(4);

    let c5_minus_c8 = c5 - c8.clone();

    vec![
        c0 + c5_minus_c8.clone(),
        c1 + c6.clone(),
        c2 - c5_minus_c8 + c7.clone(),
        c3 - c6 + c8,
        c4 - c7,
    ]
}

impl<AB: AirBuilder, const D: usize> Air<AB> for AluAir<AB::F, D>
where
    AB::F: Field + Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        debug_assert_eq!(
            main.current_slice().len(),
            self.total_width(),
            "column width mismatch"
        );

        let local = main.current_slice();
        let next = main.next_slice();
        let lane_width = Self::lane_width();

        let preprocessed = builder.preprocessed().clone();
        let prep_local = preprocessed.current_slice();
        let prep_next = preprocessed.next_slice();

        let kind = self.ext_mul_kind;
        #[cfg(debug_assertions)]
        {
            match kind {
                AluExtMulKind::Base => debug_assert_eq!(D, 1, "Base ext_mul_kind requires D == 1"),
                AluExtMulKind::Binomial { .. } => debug_assert!(D >= 2, "Binomial requires D >= 2"),
                AluExtMulKind::QuinticTrinomial => {
                    debug_assert_eq!(D, 5, "QuinticTrinomial requires D == 5");
                }
            }
        }

        let w: Option<AB::Expr> = match kind {
            AluExtMulKind::Binomial { w } => Some(AB::Expr::from(w)),
            _ => None,
        };
        let ext_mul_lane = |x: &[AB::Var], y: &[AB::Var]| -> Vec<AB::Expr> {
            match kind {
                AluExtMulKind::QuinticTrinomial => ext_mul_quintic_trinomial::<AB>(x, y),
                _ => ext_mul_binomial::<AB, D>(x, y, &w),
            }
        };

        for lane in 0..self.lanes {
            let m = lane * lane_width;
            let p = lane * PREP_LANE_WIDTH;

            let lane_local: &AluMainLaneCols<_, D> = local[m..m + lane_width].borrow();
            let lane_next: &AluMainLaneCols<_, D> = next[m..m + lane_width].borrow();

            let prep_cur: &AluPrepLaneCols<_> = prep_local[p..p + PREP_LANE_WIDTH].borrow();
            let prep_n: &AluPrepLaneCols<_> = prep_next[p..p + PREP_LANE_WIDTH].borrow();

            let a = &lane_local.a;
            let b = &lane_local.b;
            let c = &lane_local.c;
            let out = &lane_local.out;

            let mult_a = prep_cur.mult_a;
            let sel_add = prep_cur.sel_add;
            let sel_bool = prep_cur.sel_bool;
            let sel_muladd = prep_cur.sel_muladd;
            let sel_horner = prep_cur.sel_horner;

            let active = AB::Expr::ZERO - mult_a;
            let sel_mul = active - sel_bool - sel_muladd - sel_horner - sel_add;

            // ── ADD: a + b - out = 0 ────────────────────────────────────
            for i in 0..D {
                builder.assert_zero(sel_add * (a[i] + b[i] - out[i]));
            }

            // ── MUL: a * b - out = 0 ────────────────────────────────────
            let ab = ext_mul_lane(a, b);
            for i in 0..D {
                builder.assert_zero(sel_mul.dup() * (ab[i].dup() - out[i]));
            }

            // ── BOOL_CHECK: a[0]*(a[0]-1)=0, a[1..D]=0 ─────────────────
            let one = AB::Expr::ONE;
            builder.assert_zero(sel_bool * a[0] * (a[0] - one));
            for i in 1..D {
                builder.assert_zero(sel_bool * a[i]);
            }

            // ── MUL_ADD: a * b + c - out = 0 ────────────────────────────
            for i in 0..D {
                builder.assert_zero(sel_muladd * (ab[i].dup() + c[i] - out[i]));
            }

            // ── HORNER_ACC ───────────────────────────────────────────────
            let next_sel_horner = prep_n.sel_horner;

            let next_a = &lane_next.a;
            let next_b = &lane_next.b;
            let next_c = &lane_next.c;
            let next_out = &lane_next.out;

            let out_next_b = ext_mul_lane(out, next_b);

            let extra_main = self.lanes * lane_width;
            let extra_prep = self.lanes * PREP_LANE_WIDTH;
            let k_max = self.horner_packed_steps;
            let num_int = num_horner_intermediates(k_max);
            let extra_coeff_width = (num_int + 2 * (k_max - 1) + 1) * D;
            let has_extra_cols = extra_main + extra_coeff_width <= local.len()
                && extra_prep + horner_extra_prep_width(k_max) <= prep_local.len()
                && extra_prep + horner_extra_prep_width(k_max) <= prep_next.len();

            if lane == 0 && has_extra_cols {
                let next_int0 = &next[extra_main..extra_main + D];

                let mut any_packed_cur = AB::Expr::ZERO;
                for kk in 2..=k_max {
                    any_packed_cur += prep_local[extra_prep + extra_prep_sel_k_idx(kk)];
                }

                let mut any_packed_next = AB::Expr::ZERO;
                for kk in 2..=k_max {
                    any_packed_next += prep_next[extra_prep + extra_prep_sel_k_idx(kk)];
                }

                let next_sel_k2 = prep_next[extra_prep + extra_prep_sel_k_idx(2)];
                let mut sel_ge3_next = AB::Expr::ZERO;
                for kk in 3..=k_max {
                    sel_ge3_next += prep_next[extra_prep + extra_prep_sel_k_idx(kk)];
                }

                let ac_base = extra_main + num_int * D;
                let b_sq_base = ac_base + 2 * (k_max - 1) * D;
                let b_sq = &local[b_sq_base..b_sq_base + D];
                let b_sq_next = &next[b_sq_base..b_sq_base + D];
                let bb = ext_mul_lane(b, b);
                for i in 0..D {
                    builder.assert_zero(any_packed_cur.dup() * (b_sq[i] - bb[i].dup()));
                }

                let out_b_sq = ext_mul_lane(out, b_sq_next);
                let c0_b_next = ext_mul_lane(next_c, next_b);
                let a0_b_next = ext_mul_lane(next_a, next_b);

                let ac_base_next = extra_main + num_int * D;
                let off1 = ac_base_next;
                let a1_next = &next[off1..off1 + D];
                let c1_next = &next[off1 + D..off1 + 2 * D];

                // 1) Packed inter-row: fold first two steps of next row
                for i in 0..D {
                    let poly_i = out_b_sq[i].dup() + c0_b_next[i].dup() - a0_b_next[i].dup()
                        + c1_next[i]
                        - a1_next[i];
                    builder.assert_zero(next_sel_k2.dup() * (poly_i.dup() - next_out[i]));
                    builder.assert_zero(sel_ge3_next.dup() * (poly_i - next_int0[i]));
                }

                // 2) Single-step fallback: prev_out -> next_out
                let next_sel_single = next_sel_horner - any_packed_next;
                for i in 0..D {
                    builder.assert_zero(
                        next_sel_single.dup()
                            * (out_next_b[i].dup() + next_c[i] - next_a[i] - next_out[i]),
                    );
                }

                // 3) Intra-row packed legs (per arity selector sel_k, k >= 3)
                for kk in 3..=k_max {
                    let sel_kk = prep_local[extra_prep + extra_prep_sel_k_idx(kk)];
                    let mut s = 2usize;
                    let mut curr_int_slot = 0usize;
                    while s < kk {
                        let int_curr = &local
                            [extra_main + curr_int_slot * D..extra_main + (curr_int_slot + 1) * D];
                        let off_s = ac_base + 2 * (s - 1) * D;
                        let a_s = &local[off_s..off_s + D];
                        let c_s = &local[off_s + D..off_s + 2 * D];
                        if s + 1 < kk {
                            let c_s = &local[off_s + D..off_s + 2 * D];
                            let off_sp1 = ac_base + 2 * s * D;
                            let a_sp1 = &local[off_sp1..off_sp1 + D];
                            let c_sp1 = &local[off_sp1 + D..off_sp1 + 2 * D];

                            let int_b_sq = ext_mul_lane(int_curr, b_sq);
                            let c_s_b = ext_mul_lane(c_s, b);
                            let a_s_b = ext_mul_lane(a_s, b);

                            if s + 2 >= kk {
                                for i in 0..D {
                                    let prod = int_b_sq[i].dup() + c_s_b[i].dup() - a_s_b[i].dup()
                                        + c_sp1[i]
                                        - a_sp1[i];
                                    builder.assert_zero(sel_kk * (prod - out[i]));
                                }
                            } else {
                                let int_next = &local[extra_main + (curr_int_slot + 1) * D
                                    ..extra_main + (curr_int_slot + 2) * D];
                                for i in 0..D {
                                    let prod = int_b_sq[i].dup() + c_s_b[i].dup() - a_s_b[i].dup()
                                        + c_sp1[i]
                                        - a_sp1[i];
                                    builder.assert_zero(sel_kk * (prod - int_next[i]));
                                }
                                curr_int_slot += 1;
                            }
                            s += 2;
                        } else {
                            let int_b = ext_mul_lane(int_curr, b);
                            for i in 0..D {
                                builder.assert_zero(
                                    sel_kk * (int_b[i].dup() + c_s[i] - a_s[i] - out[i]),
                                );
                            }
                            s += 1;
                        }
                    }
                }
            } else {
                for i in 0..D {
                    builder.assert_zero(
                        next_sel_horner
                            * (out_next_b[i].dup() + next_c[i] - next_a[i] - next_out[i]),
                    );
                }
            }
        }
    }
}

impl<F: Field + Copy, const D: usize> LookupAir<F> for AluAir<F, D> {
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let new_idx = self.num_lookup_columns;
        self.num_lookup_columns += 1;
        vec![new_idx]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        let mut lookups = Vec::new();
        self.num_lookup_columns = 0;

        let (symbolic_main_local, preprocessed_local) = create_symbolic_variables::<F>(
            self.preprocessed_width(),
            BaseAir::<F>::width(self),
            0,
            0,
        );

        for lane in 0..self.lanes {
            let lane_offset = lane * Self::lane_width();
            let preprocessed_lane_offset = lane * Self::preprocessed_lane_width();

            // 4 lookups per lane: a, b, c, out (all Direction::Receive)
            let lane_lookup_inputs = get_alu_index_lookups::<F, D>(
                lane_offset,
                preprocessed_lane_offset,
                &symbolic_main_local,
                &preprocessed_local,
            );
            lookups.extend(lane_lookup_inputs.into_iter().map(|inps| {
                LookupAir::register_lookup(self, Kind::Global("WitnessChecks".to_string()), &[inps])
            }));
        }

        // Extra lookups for (a_t, c_t), t = 1..K_max-1, on packed Horner rows.
        let extra_main = self.lanes * Self::lane_width();
        let extra_prep = self.lanes * Self::preprocessed_lane_width();
        let k_max = self.horner_packed_steps;
        let num_int = num_horner_intermediates(k_max);
        let ac_base = extra_main + num_int * D;

        for t in 1..k_max {
            let p = extra_prep + extra_prep_a_idx_for_step(t, k_max);
            let step: &AluPackedHornerStepPrepCols<_> =
                preprocessed_local[p..p + PACKED_HORNER_STEP_PREP_WIDTH].borrow();
            let eff_mult_a = SymbolicExpression::from(step.horner_lookup_mult_a);
            let eff_mult_c = SymbolicExpression::from(step.horner_lookup_mult_c);

            let a_idx = SymbolicExpression::from(step.a_idx);
            let main_off = ac_base + 2 * (t - 1) * D;
            let mut a_inps = vec![a_idx];
            for j in 0..D {
                a_inps.push(SymbolicExpression::from(symbolic_main_local[main_off + j]));
            }
            lookups.push(LookupAir::register_lookup(
                self,
                Kind::Global("WitnessChecks".to_string()),
                &[(a_inps, eff_mult_a, Direction::Receive)],
            ));

            let c_idx = SymbolicExpression::from(step.c_idx);
            let mut c_inps = vec![c_idx];
            for j in 0..D {
                c_inps.push(SymbolicExpression::from(
                    symbolic_main_local[main_off + D + j],
                ));
            }
            lookups.push(LookupAir::register_lookup(
                self,
                Kind::Global("WitnessChecks".to_string()),
                &[(c_inps, eff_mult_c, Direction::Receive)],
            ));
        }

        lookups
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_circuit::WitnessId;
    use p3_circuit::ops::AluOpKind;
    use p3_field::BasedVectorSpace;
    use p3_matrix::Matrix;
    use p3_test_utils::baby_bear_params::{
        BabyBear as Val, BinomialExtensionField, PrimeCharacteristicRing,
    };
    use p3_uni_stark::{prove_with_preprocessed, setup_preprocessed, verify_with_preprocessed};
    use p3_util::log2_ceil_usize;

    use super::*;
    use crate::air::test_utils::build_test_config;

    /// Convert an `AluTrace` to preprocessed values (13 columns per op) for standalone tests.
    fn trace_to_preprocessed<F: Field, ExtF: BasedVectorSpace<F>, const D: usize>(
        trace: &AluTrace<ExtF>,
    ) -> Vec<F> {
        let total_len = trace.indices.len() * AluAir::<F, D>::preprocessed_lane_width();
        let mut preprocessed_values = Vec::with_capacity(total_len);
        let neg_one = F::ZERO - F::ONE;

        for (i, kind) in trace.op_kind.iter().enumerate() {
            let (sel_add_vs_mul, sel_bool, sel_muladd, sel_horner) = match kind {
                AluOpKind::Add => (F::ONE, F::ZERO, F::ZERO, F::ZERO),
                AluOpKind::Mul => (F::ZERO, F::ZERO, F::ZERO, F::ZERO),
                AluOpKind::BoolCheck => (F::ZERO, F::ONE, F::ZERO, F::ZERO),
                AluOpKind::MulAdd => (F::ZERO, F::ZERO, F::ONE, F::ZERO),
                AluOpKind::HornerAcc => (F::ZERO, F::ZERO, F::ZERO, F::ONE),
            };

            preprocessed_values.extend(&[
                neg_one, // mult_a (base; active = 1)
                sel_add_vs_mul,
                sel_bool,
                sel_muladd,
                sel_horner,
                F::from_u32(trace.indices[i][0].0 * D as u32),
                F::from_u32(trace.indices[i][1].0 * D as u32),
                F::from_u32(trace.indices[i][2].0 * D as u32),
                F::from_u32(trace.indices[i][3].0 * D as u32),
                neg_one, // mult_b (reader placeholder)
                F::ONE,  // mult_out (creator placeholder)
                F::ONE,  // a_is_reader (standalone: constrained)
                F::ONE,  // c_is_reader (standalone: constrained)
            ]);
        }

        preprocessed_values
    }

    #[test]
    fn prove_verify_alu_add_base_field() {
        let n = 8;
        let op_kind = vec![AluOpKind::Add; n];
        let values = vec![
            [
                Val::from_u64(3),
                Val::from_u64(5),
                Val::ZERO,
                Val::from_u64(8)
            ];
            n
        ];
        let indices = vec![[WitnessId(1), WitnessId(2), WitnessId(0), WitnessId(3)]; n];

        let trace = AluTrace {
            op_kind,
            values,
            indices,
        };

        let preprocessed_values = trace_to_preprocessed::<Val, _, 1>(&trace);
        let air = AluAir::<Val, 1>::new_with_preprocessed(n, 1, preprocessed_values, 2);
        let matrix: RowMajorMatrix<Val> = air.trace_to_matrix(&trace, 1);
        assert_eq!(matrix.width(), air.total_width());

        let config = build_test_config();
        let pis: Vec<Val> = vec![];
        let (prover_data, verifier_data) =
            setup_preprocessed(&config, &air, log2_ceil_usize(matrix.height())).unwrap();
        let proof = prove_with_preprocessed(&config, &air, matrix, &pis, Some(&prover_data));
        verify_with_preprocessed(&config, &air, &proof, &pis, Some(&verifier_data))
            .expect("verification failed");
    }

    #[test]
    fn prove_verify_alu_mul_base_field() {
        let n = 8;
        let op_kind = vec![AluOpKind::Mul; n];
        let values = vec![
            [
                Val::from_u64(3),
                Val::from_u64(5),
                Val::ZERO,
                Val::from_u64(15)
            ];
            n
        ];
        let indices = vec![[WitnessId(1), WitnessId(2), WitnessId(0), WitnessId(3)]; n];

        let trace = AluTrace {
            op_kind,
            values,
            indices,
        };

        let preprocessed_values = trace_to_preprocessed::<Val, _, 1>(&trace);
        let air = AluAir::<Val, 1>::new_with_preprocessed(n, 1, preprocessed_values, 2);
        let matrix: RowMajorMatrix<Val> = air.trace_to_matrix(&trace, 1);

        let config = build_test_config();
        let pis: Vec<Val> = vec![];

        let (prover_data, verifier_data) =
            setup_preprocessed(&config, &air, log2_ceil_usize(matrix.height())).unwrap();
        let proof = prove_with_preprocessed(&config, &air, matrix, &pis, Some(&prover_data));
        verify_with_preprocessed(&config, &air, &proof, &pis, Some(&verifier_data))
            .expect("verification failed");
    }

    #[test]
    fn prove_verify_alu_bool_check() {
        let n = 8;
        // Test with valid boolean values (0 and 1)
        let op_kind = vec![AluOpKind::BoolCheck; n];
        let values = (0..n)
            .map(|i| {
                [
                    Val::from_u64(i as u64 % 2),
                    Val::ZERO,
                    Val::ZERO,
                    Val::from_u64(i as u64 % 2),
                ]
            })
            .collect();
        let indices = vec![[WitnessId(1), WitnessId(0), WitnessId(0), WitnessId(1)]; n];

        let trace = AluTrace {
            op_kind,
            values,
            indices,
        };

        let preprocessed_values = trace_to_preprocessed::<Val, _, 1>(&trace);
        let air = AluAir::<Val, 1>::new_with_preprocessed(n, 1, preprocessed_values, 2);
        let matrix: RowMajorMatrix<Val> = air.trace_to_matrix(&trace, 1);

        let config = build_test_config();
        let pis: Vec<Val> = vec![];

        let (prover_data, verifier_data) =
            setup_preprocessed(&config, &air, log2_ceil_usize(matrix.height())).unwrap();
        let proof = prove_with_preprocessed(&config, &air, matrix, &pis, Some(&prover_data));
        verify_with_preprocessed(&config, &air, &proof, &pis, Some(&verifier_data))
            .expect("verification failed");
    }

    #[test]
    fn prove_verify_alu_muladd() {
        let n = 8;
        // a * b + c = out => 3 * 5 + 2 = 17
        let op_kind = vec![AluOpKind::MulAdd; n];
        let values = vec![
            [
                Val::from_u64(3),
                Val::from_u64(5),
                Val::from_u64(2),
                Val::from_u64(17)
            ];
            n
        ];
        let indices = vec![[WitnessId(1), WitnessId(2), WitnessId(3), WitnessId(4)]; n];

        let trace = AluTrace {
            op_kind,
            values,
            indices,
        };

        let preprocessed_values = trace_to_preprocessed::<Val, _, 1>(&trace);
        let air = AluAir::<Val, 1>::new_with_preprocessed(n, 1, preprocessed_values, 2);
        let matrix: RowMajorMatrix<Val> = air.trace_to_matrix(&trace, 1);

        let config = build_test_config();
        let pis: Vec<Val> = vec![];

        let (prover_data, verifier_data) =
            setup_preprocessed(&config, &air, log2_ceil_usize(matrix.height())).unwrap();
        let proof = prove_with_preprocessed(&config, &air, matrix, &pis, Some(&prover_data));
        verify_with_preprocessed(&config, &air, &proof, &pis, Some(&verifier_data))
            .expect("verification failed");
    }

    #[test]
    fn prove_verify_alu_mixed_ops() {
        // Mix of ADD and MUL operations
        let op_kind = vec![AluOpKind::Add, AluOpKind::Mul];
        let values = vec![
            [
                Val::from_u64(3),
                Val::from_u64(5),
                Val::ZERO,
                Val::from_u64(8),
            ],
            [
                Val::from_u64(4),
                Val::from_u64(6),
                Val::ZERO,
                Val::from_u64(24),
            ],
        ];
        let indices = vec![
            [WitnessId(1), WitnessId(2), WitnessId(0), WitnessId(3)],
            [WitnessId(1), WitnessId(2), WitnessId(0), WitnessId(3)],
        ];

        let trace = AluTrace {
            op_kind,
            values,
            indices,
        };

        let preprocessed_values = trace_to_preprocessed::<Val, _, 1>(&trace);
        let air = AluAir::<Val, 1>::new_with_preprocessed(2, 1, preprocessed_values, 2);
        let matrix: RowMajorMatrix<Val> = air.trace_to_matrix(&trace, 1);

        let config = build_test_config();
        let pis: Vec<Val> = vec![];
        let (prover_data, verifier_data) =
            setup_preprocessed(&config, &air, log2_ceil_usize(matrix.height())).unwrap();
        let proof = prove_with_preprocessed(&config, &air, matrix, &pis, Some(&prover_data));
        verify_with_preprocessed(&config, &air, &proof, &pis, Some(&verifier_data))
            .expect("verification failed");
    }

    #[test]
    fn prove_verify_alu_double_step_horner_base_field() {
        // Build a simple Horner chain of length 3 over the base field and
        // check that the scheduled ALU trace with double-step rows verifies.
        // Relation: out = prev_out * b + c - a; double-step shares b for two steps.
        let n = 3;
        let op_kind = vec![AluOpKind::HornerAcc; n];

        let prev_out = Val::ZERO;
        let a0 = Val::from_u64(1);
        let b0 = Val::from_u64(2);
        let c0 = Val::from_u64(5);
        let out0 = prev_out * b0 + c0 - a0;

        let a1 = Val::ZERO;
        let b1 = b0;
        let c1 = Val::from_u64(3);
        let out1 = out0 * b1 + c1 - a1;

        let a2 = Val::from_u64(1);
        let b2 = Val::from_u64(3);
        let c2 = Val::from_u64(2);
        let out2 = out1 * b2 + c2 - a2;

        let values = vec![[a0, b0, c0, out0], [a1, b1, c1, out1], [a2, b2, c2, out2]];
        let indices = vec![[WitnessId(1), WitnessId(2), WitnessId(3), WitnessId(4)]; n];

        let trace = AluTrace {
            op_kind,
            values,
            indices,
        };

        let preprocessed_values = trace_to_preprocessed::<Val, _, 1>(&trace);
        let air = AluAir::<Val, 1>::new_with_preprocessed(n, 1, preprocessed_values, 2);
        let matrix: RowMajorMatrix<Val> = air.trace_to_matrix(&trace, 1);

        let config = build_test_config();
        let pis: Vec<Val> = vec![];
        let (prover_data, verifier_data) =
            setup_preprocessed(&config, &air, log2_ceil_usize(matrix.height())).unwrap();
        let proof = prove_with_preprocessed(&config, &air, matrix, &pis, Some(&prover_data));
        verify_with_preprocessed(&config, &air, &proof, &pis, Some(&verifier_data))
            .expect("double-step horner base-field verification failed");
    }

    #[test]
    fn prove_verify_alu_k4_packed_horner_base_field() {
        const K: usize = 4;
        let n = 8;
        let op_kind = vec![AluOpKind::HornerAcc; n];
        let b = Val::from_u64(2);
        let mut acc = Val::ZERO;
        let mut values = Vec::with_capacity(n);
        for step in 0..n {
            let a = Val::from_u64((step + 1) as u64);
            let c = Val::from_u64(5);
            let out = acc * b + c - a;
            values.push([a, b, c, out]);
            acc = out;
        }
        let indices = vec![[WitnessId(1), WitnessId(2), WitnessId(3), WitnessId(4)]; n];

        let trace = AluTrace {
            op_kind,
            values,
            indices,
        };

        let preprocessed_values = trace_to_preprocessed::<Val, _, 1>(&trace);
        let air = AluAir::<Val, 1>::new_with_preprocessed(n, 1, preprocessed_values, K);
        assert_eq!(air.horner_packed_steps, K);
        let matrix: RowMajorMatrix<Val> = air.trace_to_matrix(&trace, 1);

        let config = build_test_config();
        let pis: Vec<Val> = vec![];
        let (prover_data, verifier_data) =
            setup_preprocessed(&config, &air, log2_ceil_usize(matrix.height())).unwrap();
        let proof = prove_with_preprocessed(&config, &air, matrix, &pis, Some(&prover_data));
        verify_with_preprocessed(&config, &air, &proof, &pis, Some(&verifier_data))
            .expect("K=4 packed horner base-field verification failed");
    }

    #[test]
    fn prove_verify_alu_extension_field_d4() {
        type ExtField = BinomialExtensionField<Val, 4>;
        let n = 4;

        let a = ExtField::from_basis_coefficients_slice(&[
            Val::from_u64(7),
            Val::from_u64(3),
            Val::from_u64(4),
            Val::from_u64(5),
        ])
        .unwrap();

        let b = ExtField::from_basis_coefficients_slice(&[
            Val::from_u64(11),
            Val::from_u64(2),
            Val::from_u64(9),
            Val::from_u64(6),
        ])
        .unwrap();

        let c = ExtField::ZERO;
        let out = a * b; // multiplication result

        let trace = AluTrace {
            op_kind: vec![AluOpKind::Mul; n],
            values: vec![[a, b, c, out]; n],
            indices: vec![[WitnessId(1), WitnessId(2), WitnessId(0), WitnessId(3)]; n],
        };

        let preprocessed_values = trace_to_preprocessed::<Val, _, 4>(&trace);

        let config = build_test_config();
        let pis: Vec<Val> = vec![];

        // Get w from the extension field
        let w = Val::from_u64(11); // BabyBear's binomial extension uses w=11

        let air = AluAir::<Val, 4>::new_binomial_with_preprocessed(n, 1, w, preprocessed_values, 2);
        let matrix: RowMajorMatrix<Val> = air.trace_to_matrix(&trace, 1);
        assert_eq!(matrix.width(), air.total_width());
        let (prover_data, verifier_data) =
            setup_preprocessed(&config, &air, log2_ceil_usize(matrix.height())).unwrap();
        let proof = prove_with_preprocessed(&config, &air, matrix, &pis, Some(&prover_data));
        verify_with_preprocessed(&config, &air, &proof, &pis, Some(&verifier_data))
            .expect("extension field verification failed");
    }

    #[test]
    fn prove_verify_alu_bool_check_extension_field_d4() {
        type ExtField = BinomialExtensionField<Val, 4>;
        let n = 4;
        let w = Val::from_u64(11);

        // Valid booleans: [0,0,0,0] and [1,0,0,0] interleaved
        let zero =
            ExtField::from_basis_coefficients_slice(&[Val::ZERO, Val::ZERO, Val::ZERO, Val::ZERO])
                .unwrap();
        let one =
            ExtField::from_basis_coefficients_slice(&[Val::ONE, Val::ZERO, Val::ZERO, Val::ZERO])
                .unwrap();

        let values: Vec<[ExtField; 4]> = (0..n)
            .map(|i| {
                let v = if i % 2 == 0 { zero } else { one };
                [v, zero, zero, v]
            })
            .collect();
        let indices = vec![[WitnessId(1), WitnessId(0), WitnessId(0), WitnessId(1)]; n];

        let trace = AluTrace {
            op_kind: vec![AluOpKind::BoolCheck; n],
            values,
            indices,
        };

        let preprocessed_values = trace_to_preprocessed::<Val, _, 4>(&trace);
        let air = AluAir::<Val, 4>::new_binomial_with_preprocessed(n, 1, w, preprocessed_values, 2);
        let matrix: RowMajorMatrix<Val> = air.trace_to_matrix(&trace, 1);

        let config = build_test_config();
        let pis: Vec<Val> = vec![];
        let (prover_data, verifier_data) =
            setup_preprocessed(&config, &air, log2_ceil_usize(matrix.height())).unwrap();
        let proof = prove_with_preprocessed(&config, &air, matrix, &pis, Some(&prover_data));
        verify_with_preprocessed(&config, &air, &proof, &pis, Some(&verifier_data))
            .expect("extension field bool_check verification failed");
    }

    /// The prover should panic with an unsatisfied as higher limbs constraints are not satisfied.
    #[test]
    #[should_panic]
    fn bool_check_extension_field_rejects_nonzero_higher_coefficients() {
        type ExtField = BinomialExtensionField<Val, 4>;
        let n = 4;
        let w = Val::from_u64(11);

        // a[0]=1 is a valid boolean, but a[1]=5 is non-zero — must be rejected.
        let bad = ExtField::from_basis_coefficients_slice(&[
            Val::ONE,
            Val::from_u64(5),
            Val::ZERO,
            Val::ZERO,
        ])
        .unwrap();
        let zero = ExtField::ZERO;

        let values: Vec<[ExtField; 4]> = vec![[bad, zero, zero, bad]; n];
        let indices = vec![[WitnessId(1), WitnessId(0), WitnessId(0), WitnessId(1)]; n];

        let trace = AluTrace {
            op_kind: vec![AluOpKind::BoolCheck; n],
            values,
            indices,
        };

        let preprocessed_values = trace_to_preprocessed::<Val, _, 4>(&trace);
        let air = AluAir::<Val, 4>::new_binomial_with_preprocessed(n, 1, w, preprocessed_values, 2);
        let matrix: RowMajorMatrix<Val> = air.trace_to_matrix(&trace, 1);

        let config = build_test_config();
        let pis: Vec<Val> = vec![];
        let (prover_data, verifier_data) =
            setup_preprocessed(&config, &air, log2_ceil_usize(matrix.height())).unwrap();
        let proof = prove_with_preprocessed(&config, &air, matrix, &pis, Some(&prover_data));
        // Verification must fail — the constraint a[1] = 0 is violated.
        verify_with_preprocessed(&config, &air, &proof, &pis, Some(&verifier_data))
            .expect("should have failed");
    }

    #[test]
    fn test_alu_air_constraint_degree() {
        let preprocessed = vec![Val::ZERO; 8 * 13]; // 8 ops * 13 preprocessed columns per op
        let air = AluAir::<Val, 1>::new_with_preprocessed(8, 2, preprocessed, 2);
        p3_test_utils::assert_air_constraint_degree!(air, "AluAir");
    }
}
