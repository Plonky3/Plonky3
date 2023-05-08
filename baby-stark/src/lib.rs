//! A minimal STARK framework.

#![no_std]

extern crate alloc;

mod sym_var;

use crate::sym_var::BasicSymVar;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::{Add, Mul, Sub};
use p3_air::two_row_matrix::TwoRowMatrixView;
use p3_air::{Air, AirBuilder};
use p3_commit::pcs::PCS;
use p3_field::field::{cyclic_subgroup_coset_known_order, AbstractField, Field, FieldExtension, TwoAdicField, PrimeField};
use p3_field::packed::PackedField;
use p3_field::symbolic::SymbolicField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::IndexedParallelIterator;
use p3_maybe_rayon::MaybeIntoParIter;
use p3_maybe_rayon::ParallelIterator;
use p3_util::log2_strict_usize;

pub trait StarkConfig {
    type F: PrimeField + TwoAdicField;
    type Challenge: FieldExtension<Self::F>;
    type PCS: PCS<Self::F>;
}

pub struct BasicFoldingAirBuilder<'a, F, Exp, Var>
where
    F: Field,
    Exp: AbstractField<F>,
    Var:
        Into<Exp> + Copy + Add<Var, Output = Exp> + Sub<Var, Output = Exp> + Mul<Var, Output = Exp>,
{
    main: TwoRowMatrixView<'a, Var>,
    is_first_row: Exp,
    is_last_row: Exp,
    is_transition: Exp,
    _phantom_f: PhantomData<F>,
}

impl<'a, F, Exp, Var> AirBuilder for BasicFoldingAirBuilder<'a, F, Exp, Var>
where
    F: Field,
    Exp:
        AbstractField<F> + Add<Var, Output = Exp> + Sub<Var, Output = Exp> + Mul<Var, Output = Exp>,
    Var: Into<Exp>
        + Copy
        + Add<Var, Output = Exp>
        + Add<F, Output = Exp>
        + Add<Exp, Output = Exp>
        + Sub<Var, Output = Exp>
        + Sub<F, Output = Exp>
        + Sub<Exp, Output = Exp>
        + Mul<Var, Output = Exp>
        + Mul<F, Output = Exp>
        + Mul<Exp, Output = Exp>,
{
    type F = F;
    type Exp = Exp;
    type Var = Var;
    type M = TwoRowMatrixView<'a, Var>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Exp {
        self.is_first_row.clone()
    }

    fn is_last_row(&self) -> Self::Exp {
        self.is_last_row.clone()
    }

    fn is_transition_window(&self, size: usize) -> Self::Exp {
        if size == 2 {
            self.is_transition.clone()
        } else {
            todo!()
        }
    }

    fn assert_zero<I: Into<Self::Exp>>(&mut self, x: I) {
        todo!()
    }
}

pub fn prove<SC, A>(air: &A, trace: RowMajorMatrix<SC::F>)
where
    SC: StarkConfig,
    A: for<'a> Air<
        BasicFoldingAirBuilder<'a, SC::F, <SC::F as Field>::Packing, <SC::F as Field>::Packing>,
    >,
    A: for<'a> Air<
        BasicFoldingAirBuilder<
            'a,
            SC::F,
            SymbolicField<SC::F, BasicSymVar<SC::F>>,
            BasicSymVar<SC::F>,
        >,
    >,
{
    let degree = trace.height();
    let degree_bits = log2_strict_usize(degree);
    let quotient_degree_bits = 2; // TODO
    let quotient_size_bits = degree_bits + quotient_degree_bits;
    let quotient_size = 1 << quotient_size_bits;

    let g_subgroup = SC::F::primitive_root_of_unity(degree_bits);
    let g_extended = SC::F::primitive_root_of_unity(quotient_size_bits);
    let subgroup_last = g_subgroup.inverse();
    let next_step = 1 << quotient_degree_bits;

    let coset_shift = SC::F::MULTIPLICATIVE_GROUP_GENERATOR;
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
    let zerofier_evals = x_pow_n_evals.map(|y| y - SC::F::ONE);

    // Evaluations of L_first(x) = Z_H(x) / (x - 1) on our coset s H.
    let lagrange_first_evals: Vec<_> = g_subgroup
        .powers()
        .zip(zerofier_evals.clone())
        .map(|(x, z)| z / (x - SC::F::ONE))
        .collect();

    // Evaluations of L_last(x) = Z_H(x) / (x - g^-1) on our coset s H.
    let lagrange_last_evals: Vec<_> = g_subgroup
        .powers()
        .zip(zerofier_evals)
        .map(|(x, z)| z / (x - subgroup_last))
        .collect();

    let quotient_values = (0..quotient_size)
        .into_par_iter()
        .step_by(<SC::F as Field>::Packing::WIDTH)
        .flat_map_iter(|i_local_start| {
            let i_next_start = (i_local_start + next_step) % quotient_size;
            let i_range = i_local_start..i_local_start + <SC::F as Field>::Packing::WIDTH;

            let x = *<SC::F as Field>::Packing::from_slice(&coset[i_range.clone()]);
            let is_transition = x - subgroup_last;
            let is_first_row =
                *<SC::F as Field>::Packing::from_slice(&lagrange_first_evals[i_range.clone()]);
            let is_last_row = *<SC::F as Field>::Packing::from_slice(&lagrange_last_evals[i_range]);

            let mut builder = BasicFoldingAirBuilder {
                main: TwoRowMatrixView {
                    local: todo!(), // &get_trace_values_packed(i_local_start),
                    next: todo!(),  // &get_trace_values_packed(i_next_start),
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

            (0..<SC::F as Field>::Packing::WIDTH).map(move |i| {
                let x: SC::F = todo!();
                x
                // (0..num_challenges)
                //     .map(|j| constraints_evals[j].as_slice()[i])
                //     .collect()
            })
        })
        .collect::<Vec<SC::F>>();
}

#[cfg(test)]
mod tests {
    use crate::{prove, StarkConfig};
    use p3_air::{Air, AirBuilder};
    use p3_fri::FRIBasedPCS;
    use p3_goldilocks::Goldilocks;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::Matrix;
    use p3_merkle_tree::MerkleTreeMMCS;
    use p3_poseidon::Poseidon;
    use p3_symmetric::compression::TruncatedPermutation;
    use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation, MDSPermutation};
    use p3_symmetric::sponge::PaddingFreeSponge;
    use rand::thread_rng;

    struct MyConfig;

    type F = Goldilocks;
    struct MyMds;
    impl CryptographicPermutation<[F; 8]> for MyMds {
        fn permute(&self, input: [F; 8]) -> [F; 8] {
            input // TODO
        }
    }
    impl ArrayPermutation<F, 8> for MyMds {}
    impl MDSPermutation<F, 8> for MyMds {}

    type MDS = MyMds;
    type Perm = Poseidon<F, MDS, 8, 7>;
    type H4 = PaddingFreeSponge<F, Perm, { 4 + 4 }>;
    type C = TruncatedPermutation<F, Perm, 2, 4, { 2 * 4 }>;
    type MMCS = MerkleTreeMMCS<F, [F; 4], H4, C>;
    impl StarkConfig for MyConfig {
        type F = F;
        type Challenge = Self::F; // TODO: Use an extension.
        type PCS = FRIBasedPCS<Self::F, Self::Challenge, MMCS, MMCS>;
    }

    struct MulAir;

    impl<AB: AirBuilder> Air<AB> for MulAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let main_local = main.row(0);
            let diff = main_local[0] * main_local[1] - main_local[2];
            builder.assert_zero(diff);
        }
    }

    #[test]
    fn test_prove() {
        let mut rng = thread_rng();
        let trace = RowMajorMatrix::rand(&mut rng, 256, 10);
        prove::<MyConfig, MulAir>(&MulAir, trace);
    }
}
