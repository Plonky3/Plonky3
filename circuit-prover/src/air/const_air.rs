//! [`ConstAir`] stores constants either in the base field or the extension field (of extension degree `D`).
//!
//! # Column layout
//!
//! The AIR is generic over an extension degree `D`.
//! For each constant entry, we allocate `D + 1` base-field columns.
//!
//! - `D` columns for the constant value (basis coefficients),
//! - `1` column for the `index`: the witness-bus index of the constant.
//!
//! The layout for a single row is:
//!
//! ```text
//!     [value[0], value[1], ..., value[D-1], index]
//! ```
//!
//! # Constraints
//!
//! The AIR has no constraints.
//!
//! # Global Interactions
//!
//! One interaction with the global witness bus (WitnessChecks):
//!
//! - send `(index, value[0..D])` with multiplicity `ext_mult`

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_circuit::tables::ConstTrace;
use p3_field::{BasedVectorSpace, Field};
use p3_lookup::LookupAir;
use p3_lookup::lookup_traits::{Direction, Kind, Lookup};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

use crate::air::column_layout::WITNESS_LOOKUP_PREP_LANE_WIDTH;
use crate::air::utils::{create_symbolic_variables, get_index_lookups};

/// ConstAir: vector-valued constant binding with generic extension degree D.
///
/// This chip exposes preprocessed constants that don't need to be committed during proving.
/// It serves as the source of truth for constant values in the system, with each row
/// representing a (value, index) pair where the index corresponds to a WitnessId.
///
/// Layout per row: [value[0..D-1], index] → width = D + 1
/// - value[0..D-1]: Extension field value represented as D base field coefficients
/// - index: Preprocessed WitnessId that this constant binds to
#[derive(Debug, Clone)]
pub struct ConstAir<F, const D: usize = 1> {
    /// Total number of constants defined in this trace.
    pub height: usize,
    /// Preprocessed values, corresponding to the indices in the trace.
    pub preprocessed: Vec<F>,
    /// Minimum trace height (for FRI compatibility with higher log_final_poly_len).
    pub min_height: usize,
    /// Counter for unique auxiliary lookup column indices assigned by `add_lookup_columns`.
    pub num_lookup_columns: usize,
    /// Marker tying this AIR to its base field.
    _phantom: PhantomData<F>,
}

impl<F: Field, const D: usize> ConstAir<F, D> {
    /// Construct a new `ConstAir` instance.
    ///
    /// - `height`: The number of constant values to be exposed.
    pub const fn new(height: usize) -> Self {
        Self {
            height,
            preprocessed: Vec::new(),
            min_height: 1,
            num_lookup_columns: 0,
            _phantom: PhantomData,
        }
    }

    pub const fn new_with_preprocessed(height: usize, preprocessed: Vec<F>) -> Self {
        Self {
            height,
            preprocessed,
            min_height: 1,
            num_lookup_columns: 0,
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

    /// Number of preprocessed columns: multiplicity + index.
    pub const fn preprocessed_width() -> usize {
        WITNESS_LOOKUP_PREP_LANE_WIDTH
    }
    /// Convert a `ConstTrace` into a `RowMajorMatrix` suitable for the STARK prover.
    ///
    /// This function is responsible for:
    ///
    /// 1. Decomposing each extension element in the trace into `D` basis coordinates.
    /// 2. Padding the trace to have a power-of-two number of rows.
    #[inline]
    #[instrument(skip_all, name = "ConstAir::build_trace")]
    pub fn trace_to_matrix<ExtF: BasedVectorSpace<F>>(
        trace: &ConstTrace<ExtF>,
        min_height: usize,
    ) -> RowMajorMatrix<F> {
        let height = trace.values.len();
        assert_eq!(
            height,
            trace.index.len(),
            "ConstTrace column length mismatch: values vs indices"
        );
        let width = D;

        let mut values = Vec::with_capacity(height * width);

        // Iterate over values and indices, populating the flat vector.
        for i in 0..height {
            // Extract basis coefficients.
            let coeffs = trace.values[i].as_basis_coefficients_slice();
            debug_assert_eq!(
                coeffs.len(),
                D,
                "extension degree mismatch for ConstTrace value"
            );
            // Copy coefficients into the first D columns.
            values.extend_from_slice(coeffs);
        }

        // Pad to power of two by repeating last row
        let mut mat = RowMajorMatrix::new(values, width);
        mat.pad_to_min_power_of_two_height(
            core::cmp::max(min_height, mat.height().next_power_of_two()),
            F::ZERO,
        );

        mat
    }
}

impl<F: Field, const D: usize> BaseAir<F> for ConstAir<F, D> {
    fn width(&self) -> usize {
        D
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = Self::preprocessed_width();
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

impl<AB: AirBuilder, const D: usize> Air<AB> for ConstAir<AB::F, D>
where
    AB::F: Field,
{
    fn eval(&self, _builder: &mut AB) {
        // No constraints for constants in Stage 1
    }
}

impl<F: Field, const D: usize> LookupAir<F> for ConstAir<F, D> {
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let new_idx = self.num_lookup_columns;
        self.num_lookup_columns += 1;
        vec![new_idx]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        self.num_lookup_columns = 0;

        let (symbolic_main_local, preprocessed_local) = create_symbolic_variables::<F>(
            Self::preprocessed_width(),
            BaseAir::<F>::width(self),
            1,
            0,
        );

        let lookup_inps = get_index_lookups::<F, D>(
            0,
            0,
            1,
            &symbolic_main_local,
            &preprocessed_local,
            Direction::Receive,
        );

        assert!(lookup_inps.len() == 1);
        let lookup = LookupAir::register_lookup(
            self,
            Kind::Global("WitnessChecks".to_string()),
            &lookup_inps,
        );

        vec![lookup]
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
    fn test_const_air_base_field() {
        // Create a CONST trace with several constant values
        // Toy example used: assert(37 * x - 111 = 0)
        let const_values = vec![
            F::from_u64(37),  // CONST 1 37
            F::from_u64(111), // CONST 3 111
            F::from_u64(0),   // CONST 4 0
        ];
        // Witness IDs these constants bind to
        let const_indices = vec![WitnessId(1), WitnessId(3), WitnessId(4)];

        // Preprocessed values are [ext_mult, index] pairs.
        let preprocessed_values = const_indices
            .iter()
            .flat_map(|idx| [F::ONE, F::from_u64(idx.0 as u64)])
            .collect::<Vec<_>>();

        let trace = ConstTrace {
            index: const_indices.clone(),
            values: const_values,
        };

        // Convert to matrix using the ConstAir
        let matrix = ConstAir::<F, 1>::trace_to_matrix(&trace, 1);

        // Verify matrix dimensions
        assert_eq!(matrix.width(), 1);

        // Height should be next power of two >= 3
        let height = matrix.height();
        assert_eq!(height, 4);

        // Verify the data layout: [value] per row (no index in main trace)
        let data = &matrix.values;

        // First row: value=37
        assert_eq!(data[0], F::from_u64(37));

        // Second row: value=111
        assert_eq!(data[1], F::from_u64(111));

        // Third row: value=0
        assert_eq!(data[2], F::from_u64(0));

        // Test that we can prove and verify (should succeed since no constraints)
        let config = build_test_config();
        let pis: Vec<F> = vec![];

        let air = ConstAir::<F, 1>::new_with_preprocessed(height, preprocessed_values);

        let preprocessed_matrix = air.preprocessed_trace().unwrap();
        assert_eq!(preprocessed_matrix.height(), height);

        // Assert the preprocessed values were properly created.
        // Layout: [ext_mult, index] (width=2)
        const_indices.iter().enumerate().for_each(|(i, const_idx)| {
            let row = preprocessed_matrix.row_slice(i).unwrap();
            assert_eq!(row[0], F::ONE);
            assert_eq!(row[1], F::from_u32(const_idx.0));
        });
        // Check the padding row
        let last_row = preprocessed_matrix.row_slice(height - 1).unwrap();
        assert_eq!(last_row[0], F::ZERO);
        assert_eq!(last_row[1], F::ZERO);

        let (prover_data, verifier_data) =
            setup_preprocessed(&config, &air, log2_ceil_usize(height)).unwrap();
        let proof = prove_with_preprocessed(&config, &air, matrix, &pis, Some(&prover_data));
        verify_with_preprocessed(&config, &air, &proof, &pis, Some(&verifier_data))
            .expect("CONST chip verification failed");
    }

    #[test]
    fn test_const_air_extension_field() {
        // Create extension field constants with all non-zero coefficients
        let const1 = EF::from_basis_coefficients_slice(&[
            F::from_u64(1), // a0
            F::from_u64(2), // a1
            F::from_u64(3), // a2
            F::from_u64(4), // a3
        ])
        .unwrap();

        let const2 = EF::from_basis_coefficients_slice(&[
            F::from_u64(5), // b0
            F::from_u64(6), // b1
            F::from_u64(7), // b2
            F::from_u64(8), // b3
        ])
        .unwrap();

        let const_values = vec![const1, const2];
        let const_indices = vec![WitnessId(10), WitnessId(20)];
        // Preprocessed values are [ext_mult, index] pairs; indices are D-scaled.
        let preprocessed_values = const_indices
            .iter()
            .flat_map(|idx| [F::ONE, F::from_u64(idx.0 as u64 * 4)])
            .collect::<Vec<_>>();

        let trace = ConstTrace {
            index: const_indices,
            values: const_values,
        };

        // Convert to matrix for D=4 extension field
        let matrix: RowMajorMatrix<F> = ConstAir::<F, 4>::trace_to_matrix(&trace, 1);

        // Verify matrix dimensions: D = 4 (4 value coefficients)
        assert_eq!(matrix.width(), 4);
        let height = matrix.height();
        assert_eq!(height, 2);

        let data = &matrix.values;

        // First row: [a0, a1, a2, a3] = [1, 2, 3, 4]
        assert_eq!(data[0], F::from_u64(1));
        assert_eq!(data[1], F::from_u64(2));
        assert_eq!(data[2], F::from_u64(3));
        assert_eq!(data[3], F::from_u64(4));

        // Second row: [b0, b1, b2, b3] = [5, 6, 7, 8]
        assert_eq!(data[4], F::from_u64(5));
        assert_eq!(data[5], F::from_u64(6));
        assert_eq!(data[6], F::from_u64(7));
        assert_eq!(data[7], F::from_u64(8));

        // Test proving and verification for extension field
        let config = build_test_config();
        let pis: Vec<F> = vec![];

        let air = ConstAir::<F, 4>::new_with_preprocessed(height, preprocessed_values);
        let preprocessed_matrix = air.preprocessed_trace().unwrap();
        // Layout: [ext_mult, index] (width=2, D-scaled indices)
        let row0 = preprocessed_matrix.row_slice(0).unwrap();
        assert_eq!(row0[0], F::ONE); // ext_mult
        // D-scaled index: WitnessId(10) → 10 * 4 = 40
        assert_eq!(row0[1], F::from_u64(40));
        let last_row = preprocessed_matrix.row_slice(height - 1).unwrap();
        assert_eq!(last_row[0], F::ONE); // ext_mult
        // D-scaled index: WitnessId(20) → 20 * 4 = 80
        assert_eq!(last_row[1], F::from_u64(80));
        let (prover_data, verifier_data) =
            setup_preprocessed(&config, &air, log2_ceil_usize(height)).unwrap();
        let proof = prove_with_preprocessed(&config, &air, matrix, &pis, Some(&prover_data));
        verify_with_preprocessed(&config, &air, &proof, &pis, Some(&verifier_data))
            .expect("Extension field CONST verification failed");
    }

    #[test]
    fn test_air_constraint_degree() {
        // 8 ops * 2 columns per op ([ext_mult, index])
        let air = ConstAir::<F, 1>::new_with_preprocessed(8, vec![F::ZERO; 16]);
        p3_test_utils::assert_air_constraint_degree!(air, "ConstAir");
    }
}
