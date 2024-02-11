use alloc::vec;
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
use p3_matrix::{Matrix, MatrixGet, MatrixRows};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::symbolic_builder::{get_log_quotient_degree, SymbolicAirBuilder};
use crate::{
    decompose_and_flatten, Commitments, OpenedValues, Proof, ProverConstraintFolder,
    StarkGenericConfig, ZerofierOnCoset,
};

#[instrument(skip_all)]
pub fn prove<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<crate::check_constraints::DebugConstraintBuilder<'a, SC::Val>>,
    #[cfg(not(debug_assertions))] A,
>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    trace: RowMajorMatrix<SC::Val>,
) -> Proof<SC>
where
    SC: StarkGenericConfig,
    A: Air<SymbolicAirBuilder<SC::Val>> + for<'a> Air<ProverConstraintFolder<'a, SC>>,
{
    #[cfg(debug_assertions)]
    crate::check_constraints::check_constraints(air, &trace);

    let degree = trace.height();
    let log_degree = log2_strict_usize(degree);

    let log_quotient_degree = get_log_quotient_degree::<SC::Val, A>(air);

    let g_subgroup = SC::Val::two_adic_generator(log_degree);

    let pcs = config.pcs();
    let (trace_commit, trace_data) =
        info_span!("commit to trace data").in_scope(|| pcs.commit_batch(trace));

    challenger.observe(trace_commit.clone());
    let alpha: SC::Challenge = challenger.sample_ext_element();

    let mut trace_ldes = pcs.get_ldes(&trace_data);
    assert_eq!(trace_ldes.len(), 1);
    let trace_lde = trace_ldes.pop().unwrap();

    let log_stride_for_quotient = pcs.log_blowup() - log_quotient_degree;
    let trace_lde_for_quotient = trace_lde.vertically_strided(1 << log_stride_for_quotient, 0);

    let quotient_values = quotient_values(
        config,
        air,
        log_degree,
        log_quotient_degree,
        trace_lde_for_quotient,
        alpha,
    );
    let quotient_chunks_flattened = decompose_and_flatten(
        quotient_values,
        SC::Challenge::from_base(pcs.coset_shift()),
        log_quotient_degree,
    );
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
            (&trace_data, &[vec![zeta, zeta * g_subgroup]]),
            (
                &quotient_data,
                &[vec![zeta.exp_power_of_2(log_quotient_degree)]],
            ),
        ],
        challenger,
    );
    let trace_local = opened_values[0][0][0].clone();
    let trace_next = opened_values[0][0][1].clone();
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
    SC: StarkGenericConfig,
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

    let mut coset: Vec<_> =
        cyclic_subgroup_coset_known_order(g_extended, coset_shift, quotient_size).collect();

    let zerofier_on_coset = ZerofierOnCoset::new(degree_bits, quotient_degree_bits, coset_shift);

    // Evaluations of L_first(x) = Z_H(x) / (x - 1) on our coset s H.
    let mut lagrange_first_evals = zerofier_on_coset.lagrange_basis_unnormalized(0);
    let mut lagrange_last_evals = zerofier_on_coset.lagrange_basis_unnormalized(degree - 1);

    // We have a few vectors of length `quotient_size`, and we're going to take slices therein of
    // length `WIDTH`. In the edge case where `quotient_size < WIDTH`, we need to pad those vectors
    // in order for the slices to exist. The entries beyond quotient_size will be ignored, so we can
    // just use default values.
    for _ in quotient_size..SC::PackedVal::WIDTH {
        coset.push(SC::Val::default());
        lagrange_first_evals.push(SC::Val::default());
        lagrange_last_evals.push(SC::Val::default());
    }

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
            let limit = SC::PackedVal::WIDTH.min(quotient_size);
            (0..limit).map(move |idx_in_packing| {
                let quotient_value = (0..<SC::Challenge as AbstractExtensionField<SC::Val>>::D)
                    .map(|coeff_idx| quotient.as_base_slice()[coeff_idx].as_slice()[idx_in_packing])
                    .collect_vec();
                SC::Challenge::from_base_slice(&quotient_value)
            })
        })
        .collect()
}
