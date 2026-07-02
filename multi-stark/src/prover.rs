//! Prove that an AIR is satisfied by a committed trace.

use p3_air::{Air, BaseAir, SymbolicAirBuilder};
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_sumcheck::PrescribedPointPcs;
use p3_util::log2_strict_usize;

use crate::commit::commit_trace;
use crate::config::{Commitment, MultiStarkConfig};
use crate::folder::MultilinearFolder;
use crate::packed_ext::PackedExt;
use crate::proof::{MultiStarkProof, single_table_protocol};
use crate::zerocheck::AirZerocheck;

/// Prove that an AIR is satisfied by a committed execution trace.
///
/// The three phases share one transcript:
///
/// ```text
///     1. commit(trace)        -> absorb commitment
///     2. zerocheck reduction  -> bound point r, sumcheck transcript
///     3. open columns at r    -> opening proof bound to the commitment
/// ```
///
/// # Arguments
///
/// - Proof configuration selecting the commitment scheme.
/// - AIR whose constraints are proved.
/// - Execution trace with one column per AIR column.
/// - Public inputs forwarded to the AIR.
/// - Grinding difficulty per sumcheck round.
/// - Fiat-Shamir transcript.
///
/// # Panics
///
/// - The trace width must match the AIR width.
/// - The trace arity must meet the commitment scheme's padding floor.
/// - This keeps the committed successor view in the same frame as zerocheck.
/// - The AIR must declare no preprocessed columns.
/// - The AIR must declare no periodic columns.
pub fn prove<C, A>(
    config: &C,
    air: &A,
    trace: &RowMajorMatrix<C::Val>,
    public_values: &[C::Val],
    pow_bits: usize,
    challenger: &mut C::Challenger,
) -> MultiStarkProof<C>
where
    C: MultiStarkConfig,
    C::Pcs: PrescribedPointPcs<C::Challenge, C::Challenger>,
    C::Challenger: FieldChallenger<C::Val>
        + GrindingChallenger<Witness = C::Val>
        + CanSampleUniformBits<C::Val>
        + CanObserve<Commitment<C>>,
    A: for<'b> Air<MultilinearFolder<'b, C::Val, C::Val, C::Challenge>>
        + for<'b> Air<
            MultilinearFolder<
                'b,
                C::Val,
                <C::Val as Field>::Packing,
                <C::Challenge as ExtensionField<C::Val>>::ExtensionPacking,
            >,
        > + for<'b> Air<MultilinearFolder<'b, C::Val, C::Challenge, C::Challenge>>
        + for<'b> Air<
            MultilinearFolder<
                'b,
                C::Val,
                PackedExt<C::Val, <C::Challenge as ExtensionField<C::Val>>::ExtensionPacking>,
                PackedExt<C::Val, <C::Challenge as ExtensionField<C::Val>>::ExtensionPacking>,
            >,
        > + Air<SymbolicAirBuilder<C::Val, C::Challenge>>
        + BaseAir<C::Val>,
    <C::Challenge as ExtensionField<C::Val>>::ExtensionPacking:
        From<C::Challenge> + From<<C::Val as Field>::Packing>,
{
    let log_height = log2_strict_usize(trace.height());
    let width = trace.width;
    let next_columns = air.main_next_row_columns();

    // The trace columns are the AIR columns, so their counts must agree.
    assert_eq!(width, air.width(), "trace width must match the AIR width");

    // A padded table would read a pad row as the last row's successor.
    // The zerocheck's successor repeats the last row, so padding would desync the two.
    assert!(
        log_height >= config.min_num_variables(),
        "trace arity must be at least the commitment scheme's padding floor"
    );

    // 1. Commit to the trace columns; the scheme absorbs its commitment into the transcript.
    let (commitment, prover_data) = commit_trace(config, trace, challenger);

    // 2. Reduce the AIR constraint to a single bound point.
    // The committed prover opens the columns through the commitment scheme,
    // so the zerocheck's own opened values are not used here.
    let zerocheck = AirZerocheck::new(air, pow_bits);
    let (zerocheck_proof, point) =
        zerocheck.prove::<C::Val, C::Challenge, _>(trace, public_values, challenger);
    let sumcheck = zerocheck_proof.sumcheck;

    // 3. Open the trace columns at the bound point, binding the opened values to the commitment.
    let protocol = single_table_protocol(log_height, width, &next_columns);
    let opening = config
        .pcs()
        .open_at(prover_data, &protocol, &[point], challenger);

    MultiStarkProof {
        commitment,
        sumcheck,
        opening,
    }
}
