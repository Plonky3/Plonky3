//! Turn an execution trace into the committed multilinear witness.

use alloc::vec::Vec;

use p3_commit::MultilinearPcs;
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::poly::Poly;

use crate::config::{Commitment, MultiStarkConfig, ProverData};

/// Split a trace matrix into one multilinear polynomial per column.
///
/// A column of height `2^k` becomes the evaluations of a `k`-variable multilinear polynomial.
/// The evaluations are taken in row order over the Boolean hypercube.
///
/// # Panics
///
/// Panics if the trace height is not a power of two.
pub(crate) fn trace_to_columns<F: Field>(trace: &RowMajorMatrix<F>) -> Vec<Poly<F>> {
    let width = trace.width;
    let height = trace.height();

    // Build one polynomial per column.
    (0..width)
        .map(|col| {
            // Row-major storage places entry (row, col) at index `row * width + col`.
            //
            //     column `col` evaluations = trace[col], trace[width + col], trace[2*width + col], ...
            let evals = (0..height)
                .map(|row| trace.values[row * width + col])
                .collect();
            Poly::new(evals)
        })
        .collect()
}

/// Commit to a trace through the configured commitment scheme.
///
/// - The trace becomes one multilinear polynomial per column.
/// - The scheme stacks those columns into a single polynomial and commits once.
/// - One commitment therefore covers every column.
/// - The scheme absorbs that commitment into the transcript, advancing the challenger.
///
/// Columns stay separate up to this point for two reasons:
/// - each carries its own opening claim,
/// - each occupies its own slot in the stacked polynomial.
///
/// # Arguments
///
/// - `config`: the proof configuration selecting the commitment scheme.
/// - `trace`: the execution trace, one column per AIR column.
/// - `challenger`: the Fiat-Shamir transcript, advanced by the commitment.
///
/// # Returns
///
/// - The succinct commitment to all columns.
/// - The prover-only data needed later to open the committed columns.
pub fn commit_trace<C: MultiStarkConfig>(
    config: &C,
    trace: &RowMajorMatrix<C::Val>,
    challenger: &mut C::Challenger,
) -> (Commitment<C>, ProverData<C>) {
    // One multilinear polynomial per trace column.
    let columns = trace_to_columns(trace);

    // The witness builder lays each column into a slot sized by its arity.
    // Every column must therefore share the same number of variables.
    debug_assert!(
        columns
            .windows(2)
            .all(|pair| pair[0].num_variables() == pair[1].num_variables()),
        "all trace columns must share the same arity"
    );

    // Pack the columns into the scheme's witness form (slot layout and folding live in the config).
    let witness = config.build_witness(columns);

    // Commit; the scheme observes its commitment into the transcript.
    config.pcs().commit(witness, challenger)
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_commit::MultilinearPcs;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_multilinear_util::poly::Poly;

    use super::*;
    use crate::config::MultiStarkConfig;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    /// Transcript stand-in that records every field element absorbed into it.
    #[derive(Default)]
    struct RecordingChallenger {
        observed: Vec<F>,
    }

    /// Commitment scheme stand-in for the wiring tests.
    ///
    /// - The witness is the column list.
    /// - The commitment is one checksum per column.
    struct MockPcs {
        num_vars: usize,
    }

    impl MultilinearPcs<EF, RecordingChallenger> for MockPcs {
        type Val = F;
        type Commitment = Vec<F>;
        type ProverData = Vec<Poly<F>>;
        type Proof = ();
        type Error = ();
        type Witness = Vec<Poly<F>>;
        type OpeningProtocol = ();

        fn num_vars(&self) -> usize {
            self.num_vars
        }

        fn commit(
            &self,
            witness: Self::Witness,
            challenger: &mut RecordingChallenger,
        ) -> (Self::Commitment, Self::ProverData) {
            // One checksum per column: the sum of that column's evaluations.
            let commitment: Vec<F> = witness
                .iter()
                .map(|column| column.as_slice().iter().copied().sum())
                .collect();
            // Absorb the commitment so any later transcript draw depends on it.
            challenger.observed.extend_from_slice(&commitment);
            (commitment, witness)
        }

        fn open(
            &self,
            _prover_data: Self::ProverData,
            _protocol: Self::OpeningProtocol,
            _challenger: &mut RecordingChallenger,
        ) {
        }

        fn verify(
            &self,
            _commitment: &Self::Commitment,
            _proof: &Self::Proof,
            _challenger: &mut RecordingChallenger,
            _protocol: Self::OpeningProtocol,
        ) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    /// Configuration backed by the mock scheme; its witness is the bare column list.
    struct MockConfig {
        pcs: MockPcs,
    }

    impl MultiStarkConfig for MockConfig {
        type Val = F;
        type Challenge = EF;
        type Challenger = RecordingChallenger;
        type Pcs = MockPcs;

        fn pcs(&self) -> &MockPcs {
            &self.pcs
        }

        fn build_witness(&self, columns: Vec<Poly<F>>) -> Vec<Poly<F>> {
            // The mock scheme commits columns directly, with no stacking or folding.
            columns
        }
    }

    #[test]
    fn trace_to_columns_extracts_each_column() {
        // Fixture state: a width-2, height-4 row-major trace.
        //
        //     row 0: [1, 5]
        //     row 1: [2, 6]
        //     row 2: [3, 7]
        //     row 3: [4, 8]
        //
        // Column 0 is [1, 2, 3, 4] and column 1 is [5, 6, 7, 8].
        let values = [1, 5, 2, 6, 3, 7, 4, 8].map(F::from_u64).to_vec();
        let trace = RowMajorMatrix::new(values, 2);

        let columns = trace_to_columns(&trace);

        // One polynomial per column.
        assert_eq!(columns.len(), 2);
        // Each column has height-4 evaluations, so two variables.
        assert_eq!(columns[0].num_variables(), 2);
        // Column 0 read in row order.
        assert_eq!(columns[0].as_slice(), &[1, 2, 3, 4].map(F::from_u64));
        // Column 1 read in row order.
        assert_eq!(columns[1].as_slice(), &[5, 6, 7, 8].map(F::from_u64));
    }

    #[test]
    fn commit_trace_returns_data_and_observes_commitment() {
        // Fixture state: the same width-2, height-4 trace.
        //
        // Per-column checksums:
        //
        //     column 0 sum = 1 + 2 + 3 + 4 = 10
        //     column 1 sum = 5 + 6 + 7 + 8 = 26
        let values = [1, 5, 2, 6, 3, 7, 4, 8].map(F::from_u64).to_vec();
        let trace = RowMajorMatrix::new(values, 2);
        let config = MockConfig {
            pcs: MockPcs { num_vars: 2 },
        };

        let mut challenger = RecordingChallenger::default();
        let (commitment, prover_data) = commit_trace(&config, &trace, &mut challenger);

        // The commitment carries one checksum per column.
        assert_eq!(commitment, vec![F::from_u64(10), F::from_u64(26)]);
        // Prover data retains all committed columns.
        assert_eq!(prover_data.len(), 2);
        // The commitment was absorbed into the transcript.
        assert_eq!(challenger.observed, vec![F::from_u64(10), F::from_u64(26)]);
    }
}
