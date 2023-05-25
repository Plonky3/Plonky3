use crate::{BasicFoldingAirBuilder, BasicSymVar, StarkConfig};
use alloc::vec::Vec;
use p3_air::{Air, TwoRowMatrixView};
use p3_commit::PCS;
use p3_field::{
    cyclic_subgroup_coset_known_order, AbstractField, Field, PackedField, SymbolicField,
    TwoAdicField,
};
use p3_lde::TwoAdicLDE;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::{IndexedParallelIterator, MaybeIntoParIter, ParallelIterator};
use p3_util::log2_strict_usize;

pub fn prove<SC, A>(air: &A, config: SC, trace: RowMajorMatrix<SC::Val>)
where
    SC: StarkConfig,
    A: for<'a> Air<
        BasicFoldingAirBuilder<
            'a,
            SC::Domain,
            <SC::Domain as Field>::Packing,
            <SC::Domain as Field>::Packing,
        >,
    >,
    A: for<'a> Air<
        BasicFoldingAirBuilder<
            'a,
            SC::Domain,
            SymbolicField<SC::Domain, BasicSymVar<SC::Domain>>,
            BasicSymVar<SC::Domain>,
        >,
    >,
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

    let (trace_commit, trace_data) = config.pcs().commit_batch(trace);

    let quotient_values = (0..quotient_size)
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
                        trace_lde.get(row, col).into()
                    })
                })
                .collect();
            let next: Vec<_> = (0..trace_lde.width())
                .map(|col| {
                    <SC::Domain as Field>::Packing::from_fn(|offset| {
                        let row = wrap(i_next_start + offset);
                        trace_lde.get(row, col).into()
                    })
                })
                .collect();

            let mut builder = BasicFoldingAirBuilder {
                main: TwoRowMatrixView {
                    local: &local,
                    next: &next,
                },
                is_first_row,
                is_last_row,
                is_transition,
                _phantom_f: Default::default(),
            };
            air.eval(&mut builder);

            // let mut constraints_evals = consumer.accumulators();
            // // We divide the constraints evaluations by `Z_H(x)`.
            // let denominator_inv: P = z_h_on_coset.eval_inverse_packed(i_start);
            //
            // for eval in &mut constraints_evals {
            //     *eval *= denominator_inv;
            // }

            (0..<SC::Val as Field>::Packing::WIDTH).map(move |i| {
                let x: SC::Val = todo!();
                x
                // (0..num_challenges)
                //     .map(|j| constraints_evals[j].as_slice()[i])
                //     .collect()
            })
        })
        .collect::<Vec<SC::Val>>();
}
