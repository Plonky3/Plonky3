#![allow(unused)]

use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_lookup::LookupAir;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_recursion::pcs::{
    FriProofTargets, InputProofTargets, RecExtensionValMmcs, RecValMmcs, Witness,
};
use p3_uni_stark::{StarkGenericConfig, Val};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

// Type of the `OpeningProof` used in the circuit for a `TwoAdicFriPcs`.
pub(crate) type InnerFriGeneric<MyConfig, MyHash, MyCompress, const DIGEST_ELEMS: usize> =
    FriProofTargets<
        Val<MyConfig>,
        <MyConfig as StarkGenericConfig>::Challenge,
        RecExtensionValMmcs<
            Val<MyConfig>,
            <MyConfig as StarkGenericConfig>::Challenge,
            DIGEST_ELEMS,
            RecValMmcs<Val<MyConfig>, DIGEST_ELEMS, MyHash, MyCompress>,
        >,
        InputProofTargets<
            Val<MyConfig>,
            <MyConfig as StarkGenericConfig>::Challenge,
            RecValMmcs<Val<MyConfig>, DIGEST_ELEMS, MyHash, MyCompress>,
        >,
        Witness<Val<MyConfig>>,
    >;

/// A test AIR that enforces multiplication constraints: `a^(degree-1) * b = c`
///
/// # Constraints
/// For each of REPETITIONS triples `(a, b, c)`:
/// 1. Multiplication: `a^(degree-1) * b = c`
/// 2. First row: `a^2 + 1 = b`
/// 3. Transition: `a' = a + REPETITIONS` (where `a'` is next row's `a`)
///
/// # Trace Layout
/// The trace has TRACE_WIDTH = REPETITIONS * 3 columns:
/// `[a_0, b_0, c_0, a_1, b_1, c_1, ..., a_19, b_19, c_19]`
#[derive(Clone, Copy)]
pub(crate) struct MulAir {
    /// Degree of the polynomial constraint `(a^(degree-1) * b = c)`
    pub(crate) degree: u64,
    pub(crate) rows: usize,
}

impl Default for MulAir {
    fn default() -> Self {
        Self {
            degree: 3,
            rows: 1 << 3,
        }
    }
}

/// Number of repetitions of the multiplication constraint (must be < 255 to fit in u8)
pub(crate) const REPETITIONS: usize = 20;

/// Total trace width: 3 columns per repetition (a, b, c)
pub(crate) const MAIN_TRACE_WIDTH: usize = REPETITIONS; // For c values
pub(crate) const PREP_WIDTH: usize = REPETITIONS * 2; // For a and b values

impl MulAir {
    /// Generate a random valid (or invalid) trace for testing. The trace consists of a main trace and a preprocessed trace.
    ///
    /// # Parameters
    /// - `rows`: Number of rows in the trace
    /// - `valid`: If true, generates a valid trace; if false, makes it invalid
    pub fn random_valid_trace<Val: Field>(
        &self,
        valid: bool,
    ) -> (RowMajorMatrix<Val>, RowMajorMatrix<Val>)
    where
        StandardUniform: Distribution<Val>,
    {
        let mut rng = SmallRng::seed_from_u64(1);
        let mut main_trace_values = Val::zero_vec(self.rows * MAIN_TRACE_WIDTH);
        let mut prep_trace_values = Val::zero_vec(self.rows * PREP_WIDTH);

        for (i, (a, b)) in prep_trace_values.iter_mut().tuples().enumerate() {
            let row = i / REPETITIONS;
            *a = Val::from_usize(i);

            // First row: b = a^2 + 1
            // Other rows: random b
            *b = if row == 0 {
                a.square() + Val::ONE
            } else {
                rng.random()
            };

            // Compute c = a^(degree-1) * b
            main_trace_values[i] = a.exp_u64(self.degree - 1) * *b;

            if !valid {
                // Make the trace invalid by corrupting c
                main_trace_values[i] *= Val::TWO;
            }
        }

        (
            RowMajorMatrix::new(main_trace_values, MAIN_TRACE_WIDTH),
            RowMajorMatrix::new(prep_trace_values, PREP_WIDTH),
        )
    }
}

impl<Val: Field> BaseAir<Val> for MulAir
where
    StandardUniform: Distribution<Val>,
{
    fn width(&self) -> usize {
        MAIN_TRACE_WIDTH
    }
    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<Val>> {
        Some(self.random_valid_trace(true).1)
    }
}

impl<AB: AirBuilder> Air<AB> for MulAir
where
    AB::F: Field,
    StandardUniform: Distribution<AB::F>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let main_local = main.current_slice();

        let preprocessed = builder.preprocessed().clone();
        let preprocessed_local = preprocessed.current_slice();
        let preprocessed_next = preprocessed.next_slice();

        for (i, c) in main_local.iter().enumerate() {
            let prep_start = i * 2;
            let a = preprocessed_local[prep_start];
            let b = preprocessed_local[prep_start + 1];

            // Constraint 1: a^(degree-1) * b = c
            builder.assert_zero(a.into().exp_u64(self.degree - 1) * b - *c);

            // Constraint 2: On first row, b = a^2 + 1
            builder.when_first_row().assert_eq(a * a + AB::Expr::ONE, b);

            // Constraint 3: On transition rows, a' = a + REPETITIONS
            let next_a = preprocessed_next[prep_start];
            builder
                .when_transition()
                .assert_eq(a + AB::Expr::from_u8(REPETITIONS as u8), next_a);
        }
    }
}

impl<Val: Field> LookupAir<Val> for MulAir where StandardUniform: Distribution<Val> {}
