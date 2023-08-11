use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_air::{Air, TwoRowMatrixView};
use p3_challenger::FieldChallenger;
use p3_commit::PCS;
use p3_field::{
    cyclic_subgroup_coset_known_order, AbstractField, Field, PackedField, TwoAdicField,
};
use p3_lde::{TwoAdicCosetLDE, TwoAdicLDE};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Matrix, MatrixGet};
use p3_maybe_rayon::{IndexedParallelIterator, MaybeIntoParIter, ParallelIterator};
use p3_util::log2_strict_usize;

use crate::{ConstraintFolder, StarkConfig};

pub fn prove<SC, A, Chal>(
    air: &A,
    config: &SC,
    challenger: &mut Chal,
    trace: RowMajorMatrix<SC::Val>,
) where
    SC: StarkConfig,
    A: for<'a> Air<ConstraintFolder<'a, SC::Domain, SC::Challenge, SC::PackedChallenge>>,
    Chal: FieldChallenger<SC::Domain>,
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

    let coset_shift = config.lde().shift(quotient_size_bits);
    let coset: Vec<_> =
        cyclic_subgroup_coset_known_order(g_extended, coset_shift, quotient_size).collect();

    // Evaluations of x^n on our coset s H. Note that
    //     (s g^i)^n = s^n (g^n)^i,
    // so this is the coset of <g^n> shifted by s^n.
    let x_pow_n_evals = cyclic_subgroup_coset_known_order(
        g_extended.exp_power_of_2(degree_bits),
        coset_shift.exp_power_of_2(degree_bits),
        quotient_size,
    );

    // Evaluations of Z_H(x) = (x^n - 1) on our coset s H.
    let zerofier_evals = x_pow_n_evals.map(|y| y - SC::Val::ONE);

    // Evaluations of L_first(x) = Z_H(x) / (x - 1) on our coset s H.
    let lagrange_first_evals: Vec<_> = g_subgroup
        .powers()
        .zip(zerofier_evals.clone())
        .map(|(x, z)| z / (x - SC::Val::ONE))
        .collect();

    // Evaluations of L_last(x) = Z_H(x) / (x - g^-1) on our coset s H.
    let lagrange_last_evals: Vec<_> = g_subgroup
        .powers()
        .zip(zerofier_evals)
        .map(|(x, z)| z / (x - subgroup_last))
        .collect();

    let trace_lde = config.lde().lde_batch(trace.clone(), quotient_degree_bits);

    let (_trace_commit, _trace_data) = config.pcs().commit_batch(trace.as_view());

    // challenger.observe_ext_element(trace_commit); // TODO
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

            // TODO: divide the constraints evaluations by `Z_H(x)`.

            builder.accumulator.as_slice().to_vec()
        })
        .collect::<Vec<SC::Challenge>>();
}
