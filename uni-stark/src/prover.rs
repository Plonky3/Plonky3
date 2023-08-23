use alloc::vec;
use alloc::vec::Vec;

use p3_air::{Air, TwoRowMatrixView};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, UnivariatePcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{
    cyclic_subgroup_coset_known_order, AbstractExtensionField, AbstractField, Field, PackedField,
    TwoAdicField,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Matrix, MatrixGet};
use p3_maybe_rayon::{IndexedParallelIterator, MaybeIntoParIter, ParallelIterator};
use p3_util::log2_strict_usize;

use crate::{
    decompose, Commitments, ConstraintFolder, OpenedValues, Proof, StarkConfig, ZerofierOnCoset,
};

pub fn prove<SC, A>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    trace: RowMajorMatrix<SC::Val>,
) -> Proof<SC>
where
    SC: StarkConfig,
    A: for<'a> Air<ConstraintFolder<'a, SC>>,
{
    let degree = trace.height();
    let degree_bits = log2_strict_usize(degree);
    let quotient_degree_bits = 1; // TODO
    let quotient_size_bits = degree_bits + quotient_degree_bits;
    let quotient_size = 1 << quotient_size_bits;

    let g_subgroup = SC::Domain::primitive_root_of_unity(degree_bits);
    let g_extended = SC::Domain::primitive_root_of_unity(quotient_size_bits);
    let subgroup_last = g_subgroup.inverse();
    let next_step = 1 << quotient_degree_bits;

    let coset_shift = SC::Domain::multiplicative_group_generator();
    let coset: Vec<_> =
        cyclic_subgroup_coset_known_order(g_extended, coset_shift, quotient_size).collect();

    let zerofier_on_coset = ZerofierOnCoset::new(degree_bits, quotient_degree_bits, coset_shift);

    // Evaluations of L_first(x) = Z_H(x) / (x - 1) on our coset s H.
    let mut lagrange_first_evals = vec![SC::Domain::ZERO; degree];
    lagrange_first_evals[0] = SC::Domain::ONE;
    lagrange_first_evals = config.dft().lde(lagrange_first_evals, quotient_degree_bits);

    // Evaluations of L_last(x) = Z_H(x) / (x - g^-1) on our coset s H.
    let mut lagrange_last_evals = vec![SC::Domain::ZERO; degree];
    lagrange_last_evals[degree - 1] = SC::Domain::ONE;
    lagrange_last_evals = config.dft().lde(lagrange_last_evals, quotient_degree_bits);

    // TODO: Skip this if using FriBasedPcs, in which case we already computed the trace LDE.
    let trace_lde = config
        .dft()
        .coset_lde_batch(trace.to_ext(), quotient_degree_bits, coset_shift);

    let (trace_commit, trace_data) = config.pcs().commit_batch(trace);

    challenger.observe(trace_commit.clone());
    let alpha: SC::Challenge = challenger.sample_ext_element::<SC::Challenge>();

    let quotient_values = (0..quotient_size)
        .into_par_iter()
        .step_by(SC::PackedDomain::WIDTH)
        .flat_map_iter(|i_local_start| {
            let wrap = |i| i % quotient_size;
            let i_next_start = wrap(i_local_start + next_step);
            let i_range = i_local_start..i_local_start + SC::PackedDomain::WIDTH;

            let x = *SC::PackedDomain::from_slice(&coset[i_range.clone()]);
            let is_transition = x - subgroup_last;
            let is_first_row =
                *SC::PackedDomain::from_slice(&lagrange_first_evals[i_range.clone()]);
            let is_last_row = *SC::PackedDomain::from_slice(&lagrange_last_evals[i_range]);

            let local: Vec<_> = (0..trace_lde.width())
                .map(|col| {
                    SC::PackedDomain::from_fn(|offset| {
                        let row = wrap(i_local_start + offset);
                        trace_lde.get(row, col)
                    })
                })
                .collect();
            let next: Vec<_> = (0..trace_lde.width())
                .map(|col| {
                    SC::PackedDomain::from_fn(|offset| {
                        let row = wrap(i_next_start + offset);
                        trace_lde.get(row, col)
                    })
                })
                .collect();

            let accumulator = SC::PackedChallenge::ZERO;
            let mut builder = ConstraintFolder {
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
            air.eval(&mut builder);

            // quotient(x) = constraints(x) / Z_H(x)
            let zerofier_inv: SC::PackedDomain =
                zerofier_on_coset.eval_inverse_packed(i_local_start);
            let quotient = builder.accumulator * zerofier_inv;
            quotient.as_slice().to_vec()
        })
        .collect::<Vec<SC::Challenge>>();

    let num_quotient_chunks = 1 << quotient_degree_bits;
    let quotient_value_chunks = decompose(quotient_values, quotient_degree_bits);
    let quotient_chunks_flattened: Vec<SC::Val> = (0..degree)
        .into_par_iter()
        .flat_map_iter(|row| {
            quotient_value_chunks
                .iter()
                .flat_map(move |chunk| chunk[row].as_base_slice().iter().copied())
        })
        .collect();
    let challenge_ext_degree = <SC::Challenge as AbstractExtensionField<SC::Val>>::D;
    let quotient_chunks_flattened = RowMajorMatrix::new(
        quotient_chunks_flattened,
        num_quotient_chunks * challenge_ext_degree,
    );

    let (quotient_commit, quotient_data) = config.pcs().commit_batch(quotient_chunks_flattened);
    challenger.observe(quotient_commit.clone());

    let commitments = Commitments {
        trace: trace_commit,
        quotient_chunks: quotient_commit,
    };

    let zeta: SC::Challenge = challenger.sample_ext_element();
    let (opened_values, opening_proof) = config.pcs().open_multi_batches(
        &[
            (&trace_data, &[zeta, zeta * g_subgroup]),
            (&quotient_data, &[zeta.square()]),
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
    }
}
