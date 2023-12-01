use alloc::vec::Vec;

use itertools::Itertools;
use p3_air::{Air, TwoRowMatrixView};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, UnivariatePcs, UnivariatePcsWithLde};
use p3_field::{
    cyclic_subgroup_coset_known_order, AbstractExtensionField, AbstractField, Field, PackedField,
    TwoAdicField,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Matrix, MatrixGet};
use p3_maybe_rayon::{IndexedParallelIterator, MaybeIntoParIter, ParallelIterator};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use tracing::{info_span, instrument};

use crate::symbolic_builder::SymbolicAirBuilder;
use crate::{
    decompose_and_flatten, Commitments, OpenedValues, Proof, ProverConstraintFolder, StarkConfig,
    ZerofierOnCoset,
};

#[instrument(skip_all)]
pub fn prove<SC, A>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    trace: RowMajorMatrix<SC::Val>,
) -> Proof<SC>
where
    SC: StarkConfig,
    A: Air<SymbolicAirBuilder<SC::Val>> + for<'a> Air<ProverConstraintFolder<'a, SC>>,
{
    let degree = trace.height();
    let log_degree = log2_strict_usize(degree);

    let max_constraint_degree = get_max_constraint_degree::<SC, A>(air);
    // The quotient's actual degree is approximately (max_constraint_degree - 1) n,
    // where subtracting 1 comes from division by the zerofier.
    // But we pad it to a power of two so that we can efficiently decompose the quotient.
    let log_quotient_degree = log2_ceil_usize(max_constraint_degree - 1);

    let g_subgroup = SC::Val::two_adic_generator(log_degree);

    let pcs = config.pcs();
    let (trace_commit, trace_data) =
        info_span!("commit to trace data").in_scope(|| pcs.commit_batch(trace));

    challenger.observe(trace_commit.clone());
    let alpha: SC::Challenge = challenger.sample_ext_element::<SC::Challenge>();

    let mut trace_ldes = pcs.get_ldes(&trace_data);
    assert_eq!(trace_ldes.len(), 1);
    let trace_lde = trace_ldes.pop().unwrap();
    let quotient_values = quotient_values(
        config,
        air,
        log_degree,
        log_quotient_degree,
        trace_lde,
        alpha,
    );

    let quotient_chunks_flattened = info_span!("decompose quotient polynomial").in_scope(|| {
        decompose_and_flatten::<SC>(
            quotient_values,
            SC::Challenge::from_base(pcs.coset_shift()),
            log_quotient_degree,
        )
    });
    let (quotient_commit, quotient_data) =
        info_span!("commit to quotient poly chunks").in_scope(|| {
            pcs.commit_shifted_batch(
                quotient_chunks_flattened,
                pcs.coset_shift().exp_power_of_2(log_quotient_degree),
            )
        });
    challenger.observe(quotient_commit.clone());

    let commitments = Commitments {
        trace: trace_commit,
        quotient_chunks: quotient_commit,
    };

    let zeta: SC::Challenge = challenger.sample_ext_element();
    let (opened_values, opening_proof) = pcs.open_multi_batches(
        &[
            (&trace_data, &[zeta, zeta * g_subgroup]),
            (&quotient_data, &[zeta.exp_power_of_2(log_quotient_degree)]),
        ],
        challenger,
    );
    let trace_local = opened_values[0][0][0].clone();
    let trace_next = opened_values[0][1][0].clone();
    let quotient_chunks = opened_values[1][0][0].clone();
    let opened_values = OpenedValues {
        trace_local,
        trace_next,
        quotient_chunks,
    };
    Proof {
        commitments,
        opened_values,
        opening_proof,
        degree_bits: log_degree,
    }
}

#[instrument(name = "compute quotient polynomial", skip_all)]
fn quotient_values<SC, A, Mat>(
    config: &SC,
    air: &A,
    degree_bits: usize,
    quotient_degree_bits: usize,
    trace_lde: Mat,
    alpha: SC::Challenge,
) -> Vec<SC::Challenge>
where
    SC: StarkConfig,
    A: for<'a> Air<ProverConstraintFolder<'a, SC>>,
    Mat: MatrixGet<SC::Val> + Sync,
{
    let degree = 1 << degree_bits;
    let quotient_size_bits = degree_bits + quotient_degree_bits;
    let quotient_size = 1 << quotient_size_bits;
    let g_subgroup = SC::Val::two_adic_generator(degree_bits);
    let g_extended = SC::Val::two_adic_generator(quotient_size_bits);
    let subgroup_last = g_subgroup.inverse();
    let coset_shift = config.pcs().coset_shift();
    let next_step = 1 << quotient_degree_bits;

    let coset: Vec<_> =
        cyclic_subgroup_coset_known_order(g_extended, coset_shift, quotient_size).collect();

    let zerofier_on_coset = ZerofierOnCoset::new(degree_bits, quotient_degree_bits, coset_shift);

    // Evaluations of L_first(x) = Z_H(x) / (x - 1) on our coset s H.
    let lagrange_first_evals = zerofier_on_coset.lagrange_basis_unnormalized(0);
    let lagrange_last_evals = zerofier_on_coset.lagrange_basis_unnormalized(degree - 1);

    (0..quotient_size)
        .into_par_iter()
        .step_by(SC::PackedVal::WIDTH)
        .flat_map_iter(|i_local_start| {
            let wrap = |i| i % quotient_size;
            let i_next_start = wrap(i_local_start + next_step);
            let i_range = i_local_start..i_local_start + SC::PackedVal::WIDTH;

            let x = *SC::PackedVal::from_slice(&coset[i_range.clone()]);
            let is_transition = x - subgroup_last;
            let is_first_row = *SC::PackedVal::from_slice(&lagrange_first_evals[i_range.clone()]);
            let is_last_row = *SC::PackedVal::from_slice(&lagrange_last_evals[i_range]);

            let local: Vec<_> = (0..trace_lde.width())
                .map(|col| {
                    SC::PackedVal::from_fn(|offset| {
                        let row = wrap(i_local_start + offset);
                        trace_lde.get(row, col)
                    })
                })
                .collect();
            let next: Vec<_> = (0..trace_lde.width())
                .map(|col| {
                    SC::PackedVal::from_fn(|offset| {
                        let row = wrap(i_next_start + offset);
                        trace_lde.get(row, col)
                    })
                })
                .collect();

            let accumulator = SC::PackedChallenge::zero();
            let mut folder = ProverConstraintFolder {
                main: TwoRowMatrixView {
                    local: &local,
                    next: &next,
                },
                is_first_row,
                is_last_row,
                is_transition,
                alpha,
                accumulator,
            };
            air.eval(&mut folder);

            // quotient(x) = constraints(x) / Z_H(x)
            let zerofier_inv: SC::PackedVal = zerofier_on_coset.eval_inverse_packed(i_local_start);
            let quotient = folder.accumulator * zerofier_inv;

            // "Transpose" D packed base coefficients into WIDTH scalar extension coefficients.
            (0..SC::PackedVal::WIDTH).map(move |idx_in_packing| {
                let quotient_value = (0..<SC::Challenge as AbstractExtensionField<SC::Val>>::D)
                    .map(|coeff_idx| quotient.as_base_slice()[coeff_idx].as_slice()[idx_in_packing])
                    .collect_vec();
                SC::Challenge::from_base_slice(&quotient_value)
            })
        })
        .collect()
}

fn get_max_constraint_degree<SC, A>(air: &A) -> usize
where
    SC: StarkConfig,
    A: Air<SymbolicAirBuilder<SC::Val>>,
{
    let mut builder = SymbolicAirBuilder::new(air.width());
    air.eval(&mut builder);
    builder.max_degree_multiple()
}
