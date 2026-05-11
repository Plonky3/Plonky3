//! [`PublicAir`] stores public inputs either in the base field or the extension field (of extension degree `D`).
//!
//! # Column Layout
//!
//! For each logical public input (lane) we allocate `D` base-field columns for the value.
//! The runtime parameter `lanes` controls how many independent public inputs are packed
//! side-by-side in a single row of the trace.
//!
//! We also allocate 2 preprocessed base-field columns per lane:
//! - 1 column for the multiplicity (1 for active rows, 0 for padding),
//! - 1 column for the witness index.
//!
//! # Constraints
//!
//! The AIR has no constraints.
//!
//! # Global Interactions
//!
//! For each lane, there is one interaction with the witness bus:
//! - send (index, value)

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_circuit::tables::PublicTrace;
use p3_field::{BasedVectorSpace, Field};
use p3_lookup::LookupAir;
use p3_lookup::lookup_traits::{Direction, Kind, Lookup};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

use crate::air::column_layout::WITNESS_LOOKUP_PREP_LANE_WIDTH;
use crate::air::utils::{create_symbolic_variables, get_index_lookups};

/// PublicAir: vector-valued public input binding with generic extension degree D.
/// Layout per row: [value[0..D)] repeated `lanes` times.
#[derive(Debug, Clone)]
pub struct PublicAir<F, const D: usize = 1> {
    /// Total number of logical public input operations in the trace.
    pub num_ops: usize,
    /// Number of independent public inputs packed per trace row.
    pub lanes: usize,
    /// Preprocessed witness indices for the public inputs.
    pub preprocessed: Vec<F>,
    /// Number of lookup columns registered by this AIR so far.
    pub num_lookup_columns: usize,
    /// Minimum trace height (for FRI compatibility with higher log_final_poly_len).
    pub min_height: usize,
    _phantom: PhantomData<F>,
}

impl<F: Field, const D: usize> PublicAir<F, D> {
    /// Construct a new `PublicAir` instance.
    ///
    /// - `num_ops`: total number of public input operations to be proven,
    /// - `lanes`: how many operations are packed side-by-side in each row.
    ///
    /// Panics if `lanes == 0` because we always need at least one lane per row.
    pub const fn new(num_ops: usize, lanes: usize) -> Self {
        assert!(lanes > 0, "lane count must be non-zero");
        Self {
            num_ops,
            lanes,
            preprocessed: Vec::new(),
            num_lookup_columns: 0,
            min_height: 1,
            _phantom: PhantomData,
        }
    }

    /// Construct a new `PublicAir` instance with preprocessed data.
    ///
    /// - `num_ops`: total number of public input operations to be proven,
    /// - `lanes`: how many operations are packed side-by-side in each row.
    /// - `preprocessed`: flattened preprocessed values (indices without multiplicities).
    ///
    /// Panics if `lanes == 0` because we always need at least one lane per row.
    pub const fn new_with_preprocessed(num_ops: usize, lanes: usize, preprocessed: Vec<F>) -> Self {
        assert!(lanes > 0, "lane count must be non-zero");
        Self {
            num_ops,
            lanes,
            preprocessed,
            num_lookup_columns: 0,
            min_height: 1,
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

    /// Number of base-field columns occupied by a single lane.
    /// Each lane stores D coordinates for the value.
    pub const fn lane_width() -> usize {
        D
    }

    /// Total number of columns in the main trace for this AIR instance.
    pub const fn total_width(&self) -> usize {
        self.lanes * Self::lane_width()
    }

    /// Number of preprocessed base-field columns occupied by a single lane.
    /// Each lane stores multiplicity + index (see [`WitnessLookupPrepCols`](crate::air::column_layout::WitnessLookupPrepCols)).
    pub const fn preprocessed_lane_width() -> usize {
        WITNESS_LOOKUP_PREP_LANE_WIDTH
    }

    /// Total number of preprocessed columns for this AIR instance.
    pub const fn preprocessed_width(&self) -> usize {
        self.lanes * Self::preprocessed_lane_width()
    }

    /// Flatten a PublicTrace over an extension into a base-field matrix with lanes packing.
    #[inline]
    #[instrument(skip_all, name = "PublicAir::build_trace")]
    pub fn trace_to_matrix<ExtF: BasedVectorSpace<F>>(
        trace: &PublicTrace<ExtF>,
        lanes: usize,
        min_height: usize,
    ) -> RowMajorMatrix<F> {
        let num_ops = trace.values.len();
        assert_eq!(
            num_ops,
            trace.index.len(),
            "PublicTrace column length mismatch"
        );

        let lane_width = Self::lane_width();
        let row_width = lanes * lane_width;
        let num_rows = num_ops.div_ceil(lanes);

        let mut values = Vec::with_capacity(num_rows * row_width);
        for row_idx in 0..num_rows {
            for lane in 0..lanes {
                let op_idx = row_idx * lanes + lane;
                if op_idx < num_ops {
                    let coeffs = trace.values[op_idx].as_basis_coefficients_slice();
                    debug_assert_eq!(
                        coeffs.len(),
                        D,
                        "extension degree mismatch for PublicTrace value"
                    );
                    values.extend_from_slice(coeffs);
                } else {
                    // Padding: fill with zeros
                    values.extend(core::iter::repeat_n(F::ZERO, lane_width));
                }
            }
        }

        let mut mat = RowMajorMatrix::new(values, row_width);
        mat.pad_to_min_power_of_two_height(
            core::cmp::max(min_height, mat.height().next_power_of_two()),
            F::ZERO,
        );

        mat
    }
}

impl<F: Field, const D: usize> BaseAir<F> for PublicAir<F, D> {
    fn width(&self) -> usize {
        self.total_width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = self.lanes * Self::preprocessed_lane_width();
        let mut mat = RowMajorMatrix::from_flat_padded(self.preprocessed.to_vec(), width, F::ZERO);
        mat.pad_to_min_power_of_two_height(self.min_height, F::ZERO);
        Some(mat)
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}

impl<AB: AirBuilder, const D: usize> Air<AB> for PublicAir<AB::F, D>
where
    AB::F: Field,
{
    fn eval(&self, _builder: &mut AB) {
        // No constraints for public inputs in Stage 1
    }
}

impl<F: Field, const D: usize> LookupAir<F> for PublicAir<F, D> {
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
            self.lanes,
            0,
        );

        for lane in 0..self.lanes {
            let lane_offset = lane * Self::lane_width();
            let preprocessed_lane_offset = lane * Self::preprocessed_lane_width();

            let lane_lookup_inputs = get_index_lookups::<F, D>(
                lane_offset,
                preprocessed_lane_offset,
                1,
                &symbolic_main_local,
                &preprocessed_local,
                Direction::Receive,
            );

            lookups.extend(lane_lookup_inputs.into_iter().map(|inps| {
                LookupAir::register_lookup(self, Kind::Global("WitnessChecks".to_string()), &[inps])
            }));
        }
        lookups
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_circuit::WitnessId;
    use p3_matrix::Matrix;
    use p3_test_utils::baby_bear_params::{
        BabyBear as F, BinomialExtensionField, PrimeCharacteristicRing,
    };
    use p3_uni_stark::{prove_with_preprocessed, setup_preprocessed, verify_with_preprocessed};
    use p3_util::log2_ceil_usize;

    use super::*;
    use crate::air::test_utils::build_test_config;

    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_public_air_base_field() {
        let n = 8usize;
        let lanes = 1usize;
        let values: Vec<F> = (1..=n as u64).map(F::from_u64).collect();
        let indices: Vec<WitnessId> = (0..n as u32).map(WitnessId).collect();

        // Preprocessed values are [ext_mult, index] pairs.
        let preprocessed_values = indices
            .iter()
            .flat_map(|idx| [F::ONE, F::from_u64(idx.0 as u64)])
            .collect::<Vec<_>>();

        let trace = PublicTrace {
            values,
            index: indices,
        };

        let matrix = PublicAir::<F, 1>::trace_to_matrix(&trace, lanes, 1);

        // Verify matrix dimensions
        assert_eq!(matrix.width(), 1); // D = 1, lanes = 1

        // Check first row (scope the borrow)
        {
            let row0 = matrix.row_slice(0).unwrap();
            assert_eq!(row0[0], F::from_u64(1)); // value
        }

        // Check last original row (scope the borrow)
        {
            let last_original_row = n - 1;
            let row_last = matrix.row_slice(last_original_row).unwrap();
            assert_eq!(row_last[0], F::from_u64(n as u64)); // value
        }

        let config = build_test_config();
        let air = PublicAir::<F, 1>::new_with_preprocessed(n, lanes, preprocessed_values);
        let (prover_data, verifier_data) =
            setup_preprocessed(&config, &air, log2_ceil_usize(matrix.height())).unwrap();

        // Check the correctness of preprocessed values.
        let preprocessed = air.preprocessed_trace().unwrap();
        let row0 = preprocessed.row_slice(0).unwrap();
        let last_row = preprocessed.row_slice(n - 1).unwrap();
        // Layout: [ext_mult, index]
        assert_eq!(row0[0], F::ONE); // ext_mult
        assert_eq!(last_row[0], F::ONE); // ext_mult
        assert_eq!(row0[1], F::from_u64(0)); // first index
        assert_eq!(last_row[1], F::from_u64((n - 1) as u64)); // last index

        let pis: Vec<F> = vec![];

        let proof = prove_with_preprocessed(&config, &air, matrix, &pis, Some(&prover_data));
        verify_with_preprocessed(&config, &air, &proof, &pis, Some(&verifier_data))
            .expect("PublicAir base field verification failed");
    }

    #[test]
    fn test_public_air_padding() {
        let n = 5usize;
        let lanes = 1usize;
        let values: Vec<F> = (1..=n as u64).map(F::from_u64).collect();
        let indices: Vec<WitnessId> = (0..n as u32).map(WitnessId).collect();

        // Preprocessed values are [ext_mult, index] pairs.
        let preprocessed_values = indices
            .iter()
            .flat_map(|idx| [F::ONE, F::from_u64(idx.0 as u64)])
            .collect::<Vec<_>>();

        let trace = PublicTrace {
            values,
            index: indices,
        };

        let matrix = PublicAir::<F, 1>::trace_to_matrix(&trace, lanes, 1);

        // Verify matrix dimensions
        assert_eq!(matrix.width(), 1); // D = 1, lanes = 1
        assert_eq!(matrix.height(), 8); // Padded to next power of two

        // Check first row (scope the borrow)
        {
            let row0 = matrix.row_slice(0).unwrap();
            assert_eq!(row0[0], F::from_u64(1)); // value
        }

        // Check last original row (scope the borrow)
        {
            let last_original_row = n - 1;
            let row_last = matrix.row_slice(last_original_row).unwrap();
            assert_eq!(row_last[0], F::from_u64(n as u64)); // value
        }
        // Check padded rows (scope the borrow)
        {
            for i in n..matrix.height() {
                let row = matrix.row_slice(i).unwrap();
                assert_eq!(row[0], F::ZERO); // value
            }
        }

        let config = build_test_config();
        let air = PublicAir::<F, 1>::new_with_preprocessed(n, lanes, preprocessed_values);
        let (prover_data, verifier_data) =
            setup_preprocessed(&config, &air, log2_ceil_usize(matrix.height())).unwrap();

        // Check the correctness of preprocessed values.
        let preprocessed = air.preprocessed_trace().unwrap();
        assert!(preprocessed.height() == 8);
        for i in 0..n {
            let row = preprocessed.row_slice(i).unwrap();
            // Layout: [ext_mult, index]
            assert_eq!(row[0], F::ONE); // ext_mult
            assert_eq!(row[1], F::from_u64(i as u64)); // witness index
        }
        for i in n..preprocessed.height() {
            let row = preprocessed.row_slice(i).unwrap();
            assert_eq!(row[0], F::ZERO); // padding
            assert_eq!(row[1], F::ZERO); // padding
        }

        let pis: Vec<F> = vec![];

        let proof = prove_with_preprocessed(&config, &air, matrix, &pis, Some(&prover_data));
        verify_with_preprocessed(&config, &air, &proof, &pis, Some(&verifier_data))
            .expect("PublicAir base field verification failed");
    }

    #[test]
    fn test_public_air_extension_field() {
        let lanes = 1usize;
        let a = EF::from_basis_coefficients_slice(&[
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ])
        .unwrap();

        let b = EF::from_basis_coefficients_slice(&[
            F::from_u64(5),
            F::from_u64(6),
            F::from_u64(7),
            F::from_u64(8),
        ])
        .unwrap();

        let values = vec![a, b];
        let indices = vec![WitnessId(10), WitnessId(20)];
        // Preprocessed values are [ext_mult, index] pairs; indices are D-scaled.
        let preprocessed_values = indices
            .iter()
            .flat_map(|idx| [F::ONE, F::from_u64(idx.0 as u64 * 4)])
            .collect();

        let trace = PublicTrace {
            values,
            index: indices,
        };
        let matrix = PublicAir::<F, 4>::trace_to_matrix(&trace, lanes, 1);

        // Verify matrix dimensions
        assert_eq!(matrix.width(), 4); // D = 4, lanes = 1

        // Check first row - extension field coefficients (scope the borrow)
        {
            let row0 = matrix.row_slice(0).unwrap();
            let a_coeffs = a.as_basis_coefficients_slice();
            assert_eq!(&row0[0..4], a_coeffs);
        }

        // Check second row (scope the borrow)
        {
            let row1 = matrix.row_slice(1).unwrap();
            let b_coeffs = b.as_basis_coefficients_slice();
            assert_eq!(&row1[0..4], b_coeffs);
        }

        let config = build_test_config();
        let air = PublicAir::<F, 4>::new_with_preprocessed(2, lanes, preprocessed_values);
        let (prover_data, verifier_data) =
            setup_preprocessed(&config, &air, log2_ceil_usize(matrix.height())).unwrap();

        let prep = air.preprocessed_trace().unwrap();
        let row0 = prep.row_slice(0).unwrap();
        let last_row = prep.row_slice(1).unwrap();
        // Layout: [ext_mult, index]
        assert_eq!(row0[0], F::ONE); // ext_mult
        assert_eq!(last_row[0], F::ONE); // ext_mult
        // D-scaled: WitnessId(10) → 10 * 4 = 40, WitnessId(20) → 20 * 4 = 80
        assert_eq!(row0[1], F::from_u64(40));
        assert_eq!(last_row[1], F::from_u64(80));

        let pis: Vec<F> = vec![];

        let proof = prove_with_preprocessed(&config, &air, matrix, &pis, Some(&prover_data));
        verify_with_preprocessed(&config, &air, &proof, &pis, Some(&verifier_data))
            .expect("PublicAir extension field verification failed");
    }

    #[test]
    #[should_panic]
    fn test_public_air_mismatched_lengths() {
        let values = vec![F::from_u64(1), F::from_u64(2)];
        let indices = vec![WitnessId(0)]; // Wrong length

        let trace = PublicTrace {
            values,
            index: indices,
        };
        PublicAir::<F, 1>::trace_to_matrix(&trace, 1, 1);
    }

    #[test]
    fn test_air_constraint_degree() {
        // 8 ops * 2 columns per op ([ext_mult, index])
        let air = PublicAir::<F, 1>::new_with_preprocessed(8, 1, vec![F::ZERO; 16]);
        p3_test_utils::assert_air_constraint_degree!(air, "PublicAir");
    }
}
