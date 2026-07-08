//! Prove that an AIR is satisfied by a committed trace.

use p3_air::{Air, BaseAir, SymbolicAirBuilder};
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_sumcheck::PrescribedPointPcs;
use p3_util::log2_strict_usize;

use crate::commit::commit_trace;
use crate::config::{Commitment, MultiStarkConfig, ProverData};
use crate::folder::MultilinearFolder;
use crate::keys::ProvingKey;
use crate::packed_ext::PackedExt;
use crate::proof::{MultiStarkProof, single_table_protocol};
use crate::zerocheck::AirZerocheck;

/// Prove that an AIR is satisfied by a committed execution trace.
///
/// The phases share one transcript:
///
/// ```text
///     1. absorb preprocessed commitment (if any)
///     2. commit(main trace)   -> absorb commitment
///     3. zerocheck reduction  -> bound point r, sumcheck transcript
///     4. open main at r       -> main opening bound to the main commitment
///     5. open preprocessed at r (if any) -> opening bound to the preprocessed commitment
/// ```
///
/// The preprocessed commitment lives in the proving key, committed once at setup.
/// Each proof reuses it.
/// The committed data is cloned to open at this proof's point.
///
/// # Arguments
///
/// - Proof configuration selecting the commitment schemes.
/// - Proving key carrying the reusable preprocessed commitment.
/// - AIR whose constraints are proved.
/// - Execution trace with one column per main AIR column.
/// - Public inputs forwarded to the AIR.
/// - Grinding difficulty per sumcheck round.
/// - Fiat-Shamir transcript.
///
/// # Panics
///
/// - The trace width must match the AIR width.
/// - The trace arity must meet the commitment scheme's padding floor.
/// - This keeps the committed successor view in the same frame as zerocheck.
/// - The preprocessed key width must match the AIR's declared preprocessed width.
/// - A preprocessed key, when present, must have the same height as the main trace.
/// - A periodic column's period must be a power of two dividing the trace height.
pub fn prove<C, A>(
    config: &C,
    proving_key: &ProvingKey<C>,
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
    Commitment<C>: Clone,
    ProverData<C>: Clone,
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
    let preprocessed = proving_key.preprocessed.as_ref();

    // The trace columns are the AIR columns, so their counts must agree.
    assert_eq!(width, air.width(), "trace width must match the AIR width");

    // A padded table would read a pad row as the last row's successor.
    // The zerocheck's successor repeats the last row, so padding would desync the two.
    assert!(
        log_height >= config.min_num_variables(),
        "trace arity must be at least the commitment scheme's padding floor"
    );

    // Invariant: committed preprocessed column count == AIR's declared preprocessed width.
    // A key with no preprocessed data pairs with an AIR that declares none (width 0).
    assert_eq!(
        preprocessed.map_or(0, |p| p.width),
        air.preprocessed_width(),
        "preprocessed key width must match the AIR's declared preprocessed width"
    );

    // Preprocessed and main columns share the bound point, so they must share a height.
    if let Some(preprocessed) = preprocessed {
        assert_eq!(
            preprocessed.log_height, log_height,
            "preprocessed trace height must match the main trace height"
        );
    }

    // 1. Absorb the reusable preprocessed commitment before any challenge depends on it.
    if let Some(preprocessed) = preprocessed {
        challenger.observe(preprocessed.commitment.clone());
    }

    // 2. Commit to the main trace columns.
    // The scheme absorbs its commitment into the transcript.
    let (commitment, prover_data) = commit_trace(config, trace, challenger);

    // 3. Reduce the AIR constraint to a single bound point.
    // The committed prover opens the columns through the commitment scheme,
    // so the zerocheck's own opened values are not used here.
    let zerocheck = AirZerocheck::new(air, pow_bits);
    let (zerocheck_proof, point) = zerocheck.prove::<C::Val, C::Challenge, _>(
        trace,
        preprocessed.map(|p| &p.trace),
        public_values,
        challenger,
    );
    let sumcheck = zerocheck_proof.sumcheck;

    // 4. Open the main trace columns at the bound point, binding them to the main commitment.
    let protocol = single_table_protocol(log_height, width, &next_columns);
    let opening = config.pcs().open_at(
        prover_data,
        &protocol,
        core::slice::from_ref(&point),
        challenger,
    );

    // 5. Open the preprocessed columns at the same point against the reused commitment.
    // Cloning the committed data reuses the setup encoding and Merkle tree, skipping a rebuild.
    let preprocessed_opening = preprocessed.map(|preprocessed| {
        let protocol = single_table_protocol(
            preprocessed.log_height,
            preprocessed.width,
            &preprocessed.next_columns,
        );
        config.preprocessed_pcs().open_at(
            preprocessed.prover_data.clone(),
            &protocol,
            core::slice::from_ref(&point),
            challenger,
        )
    });

    MultiStarkProof {
        commitment,
        sumcheck,
        opening,
        preprocessed_opening,
    }
}
