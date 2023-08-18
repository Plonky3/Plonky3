use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_air::{Air, TwoRowMatrixView};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::Pcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{
    cyclic_subgroup_coset_known_order, AbstractField, Field, PackedField, TwoAdicField,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Matrix, MatrixGet};
use p3_maybe_rayon::{IndexedParallelIterator, MaybeIntoParIter, ParallelIterator};
use p3_util::log2_strict_usize;

use crate::{ConstraintFolder, StarkConfig};

pub fn prove<SC, A>(
    air: &A,
    config: &SC,
    challenger: &mut SC::Challenger,
    trace: RowMajorMatrix<SC::Val>,
) where
    SC: StarkConfig,
    A: for<'a> Air<ConstraintFolder<'a, SC::Domain, SC::Challenge, SC::PackedChallenge>>,
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

    // Evaluations of L_first(x) = Z_H(x) / (x - 1) on our coset s H.
    let mut lagrange_first_evals = vec![SC::Domain::ZERO; degree];
    lagrange_first_evals[0] = SC::Domain::ONE;
    lagrange_first_evals = config.dft().lde(lagrange_first_evals, quotient_degree_bits);

    // Evaluations of L_last(x) = Z_H(x) / (x - g^-1) on our coset s H.
    let mut lagrange_last_evals = vec![SC::Domain::ZERO; degree];
    lagrange_last_evals[degree - 1] = SC::Domain::ONE;
    lagrange_last_evals = config.dft().lde(lagrange_last_evals, quotient_degree_bits);

    let (trace_commit, _trace_data) = config.pcs().commit_batch(trace.as_view());

    // TODO: Skip this if using FriBasedPcs, in which case we already computed the trace LDE.
    let trace_lde = config
        .dft()
        .coset_lde_batch(trace.to_ext(), quotient_degree_bits, coset_shift);

    challenger.observe(trace_commit);
    let alpha = challenger.sample_ext_element::<SC::Challenge>();

    let _quotient_values = (0..quotient_size)
        .into_par_iter()
        .step_by(<SC::Val as Field>::Packing::WIDTH)
        .flat_map_iter(|i_local_start| {
            let wrap = |i| i % quotient_size;
            let i_next_start = wrap(i_local_start + next_step);
            let i_range = i_local_start..i_local_start + <SC::Val as Field>::Packing::WIDTH;

            let x = *<SC::Domain as Field>::Packing::from_slice(&coset[i_range.clone()]);
            let is_transition = x - subgroup_last;
            let is_first_row =
                *<SC::Domain as Field>::Packing::from_slice(&lagrange_first_evals[i_range.clone()]);
            let is_last_row =
                *<SC::Domain as Field>::Packing::from_slice(&lagrange_last_evals[i_range]);

            let local: Vec<_> = (0..trace_lde.width())
                .map(|col| {
                    <SC::Domain as Field>::Packing::from_fn(|offset| {
                        let row = wrap(i_local_start + offset);
                        trace_lde.get(row, col)
                    })
                })
                .collect();
            let next: Vec<_> = (0..trace_lde.width())
                .map(|col| {
                    <SC::Domain as Field>::Packing::from_fn(|offset| {
                        let row = wrap(i_next_start + offset);
                        trace_lde.get(row, col)
                    })
                })
                .collect();

            let accumulator = SC::PackedChallenge::ZEROS;
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
                _phantom_f: PhantomData,
            };
            air.eval(&mut builder);

            // TODO: divide the constraints evaluations by `Z_H(x) = x^n - 1`.

            builder.accumulator.as_slice().to_vec()
        })
        .collect::<Vec<SC::Challenge>>();
}
