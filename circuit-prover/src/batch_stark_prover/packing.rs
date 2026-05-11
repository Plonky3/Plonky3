use alloc::vec::Vec;

use p3_circuit::ops::NpoTypeId;
use serde::{Deserialize, Serialize};

/// Configuration for packing multiple primitive operations into a single AIR row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TablePacking {
    /// Number of public-input operations packed per AIR row.
    public_lanes: usize,
    /// Number of ALU operations packed per AIR row.
    alu_lanes: usize,
    /// Per-NPO lane counts: `(op_type, lanes)`. Defaults to 1 for any op not listed.
    #[serde(default)]
    npo_lanes: Vec<(NpoTypeId, usize)>,
    /// Minimum trace height for all tables (must be power of two).
    /// This is required for FRI with higher `log_final_poly_len`.
    /// FRI requires: `log_trace_height > log_final_poly_len + log_blowup`
    /// So min_trace_height should be >= `2^(log_final_poly_len + log_blowup + 1)`
    min_trace_height: usize,
    /// Pack this many consecutive `HornerAcc` ops (same `b` witness) per ALU row on lane 0.
    /// Must be at least 2. Default 2 matches the previous double-step Horner layout.
    #[serde(default = "default_horner_pack_k")]
    horner_packed_steps: usize,
}

const fn default_horner_pack_k() -> usize {
    2
}

impl TablePacking {
    /// Create a new [`TablePacking`] with the given primitive lane counts (clamped to at least 1).
    ///
    /// NPO lanes default to 1. Use [`with_npo_lanes`](Self::with_npo_lanes) to override per op type.
    pub fn new(public_lanes: usize, alu_lanes: usize) -> Self {
        Self {
            public_lanes: public_lanes.max(1),
            alu_lanes: alu_lanes.max(1),
            npo_lanes: Vec::new(),
            min_trace_height: 1,
            horner_packed_steps: 2,
        }
    }

    /// Override packed Horner chain length (must be >= 2).
    #[must_use]
    pub fn with_horner_pack_k(mut self, k: usize) -> Self {
        assert!(k >= 2, "horner_packed_steps must be at least 2");
        self.horner_packed_steps = k;
        self
    }

    /// Override public and ALU lane counts after trace-driven clamping (e.g. dummy-only traces).
    ///
    /// Used when embedding the effective packing in [`super::BatchStarkProof`] so metadata matches
    /// proving while preserving [`Self::horner_packed_steps`] and NPO lane overrides.
    #[must_use]
    pub fn with_public_alu_lanes(mut self, public_lanes: usize, alu_lanes: usize) -> Self {
        self.public_lanes = public_lanes.max(1);
        self.alu_lanes = alu_lanes.max(1);
        self
    }

    /// Override the lane count for a specific non-primitive op type (builder-style).
    ///
    /// Any NPO not listed falls back to the lane count returned by its [`TableProver`].
    #[must_use]
    pub fn with_npo_lanes(mut self, op_type: impl Into<NpoTypeId>, lanes: usize) -> Self {
        let op_type = op_type.into();
        let lanes = lanes.max(1);
        if let Some(entry) = self.npo_lanes.iter_mut().find(|(k, _)| *k == op_type) {
            entry.1 = lanes;
        } else {
            self.npo_lanes.push((op_type, lanes));
        }
        self
    }

    /// Update the current [`TablePacking`] with a minimum trace height requirement.
    ///
    /// FRI requires: `log_trace_height > log_final_poly_len + log_blowup`
    /// So `min_trace_height` should be >= `2^(log_final_poly_len + log_blowup + 1)`
    ///
    /// For example, with `log_final_poly_len = 3` and `log_blowup = 1`:
    /// - Required: `min_trace_height > 2^(3+1) = 16`
    /// - So use `min_trace_height = 32` (next power of two)
    #[must_use]
    pub fn with_min_trace_height(mut self, min_trace_height: usize) -> Self {
        // Ensure min_trace_height is a power of two and at least 1
        self.min_trace_height = min_trace_height.next_power_of_two().max(1);
        self
    }

    /// Update the current [`TablePacking`] with minimum height derived from FRI parameters.
    ///
    /// This automatically calculates the minimum trace height from `log_final_poly_len` and `log_blowup`.
    #[must_use]
    pub const fn with_fri_params(mut self, log_final_poly_len: usize, log_blowup: usize) -> Self {
        // FRI requires: log_min_height > log_final_poly_len + log_blowup
        // So min_height must be >= 2^(log_final_poly_len + log_blowup + 1)
        let min_log_height = log_final_poly_len + log_blowup + 1;
        self.min_trace_height = 1usize << min_log_height;
        self
    }

    /// Return the number of public-input operations packed per AIR row.
    pub const fn public_lanes(&self) -> usize {
        self.public_lanes
    }

    /// Return the number of ALU operations packed per AIR row.
    pub const fn alu_lanes(&self) -> usize {
        self.alu_lanes
    }

    /// Return the lane count for a specific NPO type.
    ///
    /// Returns the overridden value if one was set via [`with_npo_lanes`](Self::with_npo_lanes),
    /// otherwise returns `None` (the caller should fall back to the prover's own default).
    pub fn npo_lanes(&self, op_type: &NpoTypeId) -> Option<usize> {
        self.npo_lanes
            .iter()
            .find(|(k, _)| k == op_type)
            .map(|(_, v)| *v)
    }

    /// Return the minimum trace height (always a power of two, at least 1).
    pub const fn min_trace_height(&self) -> usize {
        self.min_trace_height
    }

    /// Number of consecutive HornerAcc steps packed into one scheduled ALU row (lane 0).
    pub const fn horner_packed_steps(&self) -> usize {
        self.horner_packed_steps
    }
}

impl Default for TablePacking {
    fn default() -> Self {
        Self::new(1, 1)
    }
}

/// Main trace width, preprocessed row width, logical AIR row count, and lane packing for one table.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AirTableShape {
    pub main_cols: usize,
    pub prep_cols: usize,
    pub rows: usize,
    pub lanes: usize,
}

/// Layout of every table in a batch (for prover logging).
#[derive(Debug)]
pub(crate) struct TraceTablesLayout {
    pub const_: AirTableShape,
    pub public: AirTableShape,
    pub alu: AirTableShape,
    pub non_primitives: Vec<(NpoTypeId, AirTableShape)>,
}

impl TraceTablesLayout {
    /// Log each AIR’s main width, preprocessed width, row count, and lanes at info level.
    pub fn log(&self) {
        tracing::info!(
            table = "CONST",
            main_cols = self.const_.main_cols,
            prep_cols = self.const_.prep_cols,
            rows = self.const_.rows,
            "AIR shape"
        );
        tracing::info!(
            table = "PUBLIC",
            main_cols = self.public.main_cols,
            prep_cols = self.public.prep_cols,
            rows = self.public.rows,
            "AIR shape"
        );
        tracing::info!(
            table = "ALU",
            main_cols = self.alu.main_cols,
            prep_cols = self.alu.prep_cols,
            rows = self.alu.rows,
            lanes = self.alu.lanes,
            "AIR shape"
        );
        for (op, shape) in &self.non_primitives {
            tracing::info!(
                table = ?op,
                main_cols = shape.main_cols,
                prep_cols = shape.prep_cols,
                rows = shape.rows,
                lanes = shape.lanes,
                "AIR shape"
            );
        }
    }
}
