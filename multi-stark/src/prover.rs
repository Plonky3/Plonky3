//! Prove that AIR instances are satisfied by committed traces.

use alloc::vec::Vec;

use p3_air::{Air, BaseAir, SymbolicAirBuilder};
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::MultilinearPcs;
use p3_field::{ExtensionField, Field};
use p3_sumcheck::PrescribedPointPcs;

use crate::ProverInstances;
use crate::boundary::{self, BoundaryIo};
use crate::config::{Commitment, MultiStarkConfig, ProverData};
use crate::folder::MultilinearFolder;
use crate::instance::ProverParts;
use crate::packed_ext::PackedExt;
use crate::proof::MultiStarkProof;
use crate::zerocheck::AirZerocheck;

/// Prove that a batch of AIR instances is satisfied by committed execution traces.
///
/// The phases share one transcript:
///
/// ```text
///     1. absorb batched preprocessed commitment (if any)
///     2. commit(main trace tables) -> absorb main commitment
///     3. zerocheck reduction       -> bound point r, sumcheck transcript
///     4. open main tables at r     -> openings bound to the main commitment
///     5. open preprocessed tables at r (if any)
///                                  -> openings bound to the preprocessed commitment
/// ```
///
/// Main trace tables are committed together in input-instance order. Each table
/// is still opened at the suffix of the common zerocheck point matching that
/// instance's height.
///
/// The preprocessed commitment lives in the proving key, committed once at setup.
/// All non-empty preprocessed traces are stacked in AIR-instance order, skipping
/// AIRs with no preprocessed columns. Each proof clones the committed data to open
/// it at this proof's point without rebuilding the preprocessed commitment.
///
/// # Arguments
///
/// - `config`: proof configuration selecting the commitment schemes.
/// - `instances`: AIRs, transposed main trace tables, shared proving key, and public inputs.
/// - `pow_bits`: grinding difficulty per sumcheck round.
/// - `challenger`: Fiat-Shamir transcript.
///
/// # Panics
///
/// Panics if the instance list is empty.
/// Panics if any trace arity is below the commitment scheme's padding floor.
/// Panics if the prover instances do not all use the same proving key.
/// Panics if an AIR declares preprocessed columns but the proving key has none.
/// Panics if an AIR's public boundary declaration names a cell or value it does not have.
#[tracing::instrument(skip_all)]
pub fn prove<'a, C, A>(
    config: &C,
    instances: ProverInstances<'a, C, A>,
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
    assert!(!instances.is_empty());

    let ProverParts {
        proving_key,
        mut tables,
        instances,
    } = instances.into_parts();

    // Every committed table must meet the scheme's padding floor.
    //
    // A table below the floor is zero-padded before commitment.
    // Padding moves the repeated boundary row into the pad.
    // The committed successor view then reads a pad row instead of the last row.
    // That disagrees with the zerocheck's repeat-last successor convention.
    assert!(
        tables
            .iter()
            .all(|table| table.num_variables() >= config.min_num_variables()),
        "every trace arity must be at least the commitment scheme's padding floor"
    );

    // 1. Absorb the reusable batched preprocessed commitment before any challenge depends on it.
    if let Some(preprocessed) = &proving_key.preprocessed {
        challenger.observe(preprocessed.commitment.clone());
    }

    // 2. Commit all main trace tables in instance order. The scheme absorbs its
    // commitment into the transcript.
    let num_variables = instances.num_variables();

    // Public boundary cells per instance: committed as zero, restored on the verifier.
    //
    // Invariant: every cell names a real column and a real public value.
    //   Blanking and folding index by those numbers with no further check.
    let airs = instances.airs();
    let boundary_io = airs
        .iter()
        .map(|air| {
            boundary::validate::<C::Val, _>(*air).expect("invalid boundary-IO declaration");
            BoundaryIo::new(air.public_boundary_io())
        })
        .collect::<Vec<_>>();

    // Blank each declared cell in place, keeping its true value aside.
    // A table with no declared cells is left exactly as the caller supplied it.
    let cell_values = tables
        .iter_mut()
        .zip(boundary_io.iter())
        .map(|(table, boundary)| boundary.take_cells(table))
        .collect::<Vec<_>>();

    // The commitment therefore never carries data the verifier already holds.
    let witness = config.build_witness(tables);
    let (commitment, prover_data) = config.pcs().commit(witness, challenger);

    // Tables the zerocheck folds, every one derived from a committed view.
    //
    //     no declared cells : folded as committed, nothing copied
    //     declared cells    : folded as committed with the true values written back
    //
    // Writing the values back leaves the fold one Lagrange bump above the commitment:
    //
    //     folded   = committed + sum_cells eq_cell * true_value
    //     restored = committed + sum_cells eq_cell * public       (the verifier's side)
    //
    // The pin the folder asserts forces those two lines to agree.
    let restored = boundary_io
        .iter()
        .zip(cell_values.iter())
        .enumerate()
        .map(|(table_index, (boundary, values))| {
            (!boundary.is_empty()).then(|| {
                let mut table = config.committed_table(&prover_data, table_index).clone();

                // Cells are addressed by row index.
                // The padding-floor assert above keeps the committed arity unchanged.
                assert_eq!(table.num_variables(), num_variables[table_index]);

                // Writing back asserts each target cell is still blank.
                // The commit path is thereby held to leaving the cells in place.
                boundary.restore_cells(&mut table, values);
                table
            })
        })
        .collect::<Vec<_>>();
    let tables = restored
        .iter()
        .enumerate()
        .map(|(table_index, restored)| {
            // Fall back to the committed view for any table that needed no edit.
            restored
                .as_ref()
                .unwrap_or_else(|| config.committed_table(&prover_data, table_index))
        })
        .collect::<Vec<_>>();

    // One entry per instance, in instance order.
    // An AIR with preprocessed columns takes the next committed table in setup order.
    // An AIR without them takes `None`.
    // A missing preprocessed key is valid only when no AIR declares preprocessed columns.
    let preprocessed_data = proving_key.preprocessed.as_ref().map(|p| &p.prover_data);
    let mut next_table = 0;
    let preprocessed_tables = instances
        .iter()
        .map(|instance| {
            (instance.air.preprocessed_width() != 0).then(|| {
                let data = preprocessed_data.expect(
                    "preprocessed proving key is missing for an AIR with preprocessed columns",
                );
                let table = config.committed_table(data, next_table);
                next_table += 1;
                table
            })
        })
        .collect::<Vec<_>>();

    // 3. Reduce all AIR constraints to one batched sumcheck and one bound point.
    // The committed prover opens columns through the commitment schemes below, so
    // the zerocheck's own opened values are not used as the final proof openings.
    let zerocheck = AirZerocheck::new(&airs, pow_bits);
    let (zerocheck_proof, point) = zerocheck.prove::<C::Val, C::Challenge, _>(
        &preprocessed_tables,
        &tables,
        &instances.public_values(),
        challenger,
    );
    let sumcheck = zerocheck_proof.sumcheck;

    drop(tables);
    drop(restored);
    drop(preprocessed_tables);

    // 4. Open each main trace table at its suffix of the common bound point.
    let opening = config.pcs().open_at(
        prover_data,
        &instances.opening_protocol(),
        &instances.main_points(&point),
        challenger,
    );

    // 5. Open each non-empty preprocessed table at the corresponding suffix of
    // the same bound point, reusing the setup commitment data.
    let preprocessed_opening = proving_key.preprocessed.as_ref().map(|preprocessed| {
        config.preprocessed_pcs().open_at(
            preprocessed.prover_data.clone(),
            &instances.preprocessed_opening_protocol(),
            &instances.preprocessed_points(&point),
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
