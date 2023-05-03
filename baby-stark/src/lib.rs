//! A minimal STARK framework.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::{Add, Mul, Sub};
use p3_air::two_row_matrix::TwoRowMatrixView;
use p3_air::{Air, AirBuilder};
use p3_commit::pcs::PCS;
use p3_field::field::{Field, FieldExtension, FieldLike, TwoAdicField};
use p3_field::packed::PackedField;
use p3_field::symbolic::SymbolicField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::IndexedParallelIterator;
use p3_maybe_rayon::MaybeIntoParIter;
use p3_maybe_rayon::ParallelIterator;
use p3_util::log2_strict_usize;

pub trait StarkConfig {
    type F: TwoAdicField;
    type Challenge: FieldExtension<Self::F>;
    type PCS: PCS<Self::F>;
}

pub struct BasicFoldingAirBuilder<'a, F, FL, V>
where
    F: Field,
    FL: FieldLike<F>,
    V: Into<FL> + Copy + Add<V, Output = FL> + Sub<V, Output = FL> + Mul<V, Output = FL>,
{
    main: TwoRowMatrixView<'a, V>,
    is_first_row: FL,
    is_last_row: FL,
    is_transition: FL,
    _phantom_f: PhantomData<F>,
}

impl<'a, F, FL, V> AirBuilder for BasicFoldingAirBuilder<'a, F, FL, V>
where
    F: Field,
    FL: FieldLike<F> + Add<V, Output = FL> + Sub<V, Output = FL> + Mul<V, Output = FL>,
    V: Into<FL>
        + Copy
        + Add<V, Output = FL>
        + Add<F, Output = FL>
        + Add<FL, Output = FL>
        + Sub<V, Output = FL>
        + Sub<F, Output = FL>
        + Sub<FL, Output = FL>
        + Mul<V, Output = FL>
        + Mul<F, Output = FL>
        + Mul<FL, Output = FL>,
{
    type F = F;
    type FL = FL;
    type Var = V;
    type M = TwoRowMatrixView<'a, V>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::FL {
        self.is_first_row.clone()
    }

    fn is_last_row(&self) -> Self::FL {
        self.is_last_row.clone()
    }

    fn is_transition_window(&self, size: usize) -> Self::FL {
        if size == 2 {
            self.is_transition.clone()
        } else {
            todo!()
        }
    }

    fn assert_zero<I: Into<Self::FL>>(&mut self, x: I) {
        todo!()
    }
}

pub struct BasicSymVar {
    pub row_offset: usize,
    pub column: usize,
}

pub fn prove<SC, A>(air: &A, trace: RowMajorMatrix<SC::F>)
where
    SC: StarkConfig,
    A: for<'a> Air<
        BasicFoldingAirBuilder<'a, SC::F, <SC::F as Field>::Packing, <SC::F as Field>::Packing>,
    >,
    A: for<'a> Air<
        BasicFoldingAirBuilder<'a, SC::F, SymbolicField<SC::F, BasicSymVar>, BasicSymVar>,
    >,
{
    let degree = trace.height();
    let degree_bits = log2_strict_usize(degree);
    let quotient_degree_bits = 2; // TODO
    let last = SC::F::primitive_root_of_unity(degree_bits).inverse();
    let quotient_size = degree << quotient_degree_bits;
    let next_step = 1 << quotient_degree_bits;
    let quotient_values = (0..quotient_size)
        .into_par_iter()
        .step_by(<SC::F as Field>::Packing::WIDTH)
        .flat_map_iter(|i_local_start| {
            let i_next_start = (i_local_start + next_step) % quotient_size;
            let i_range = i_local_start..i_local_start + <SC::F as Field>::Packing::WIDTH;

            let x: <SC::F as Field>::Packing = todo!(); // *P::from_slice(&coset[i_range.clone()]);
            let is_transition = x - last;
            let is_first_row = todo!(); // *P::from_slice(&lagrange_first.values[i_range.clone()]);
            let is_last_row = todo!(); // *P::from_slice(&lagrange_last.values[i_range]);

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
    use crate::StarkConfig;
    use p3_fri::FRIBasedPCS;
    use p3_goldilocks::Goldilocks;
    use p3_merkle_tree::MerkleTreeMMCS;
    use p3_poseidon::Poseidon;
    use p3_symmetric::compression::TruncatedPermutation;
    use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation, MDSPermutation};
    use p3_symmetric::sponge::PaddingFreeSponge;

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
    type H4 = PaddingFreeSponge<F, Perm, 4, 4>;
    type C = TruncatedPermutation<F, Perm, 2, 4>;
    type MMCS = MerkleTreeMMCS<F, [F; 4], H4, C>;
    impl StarkConfig for MyConfig {
        type F = F;
        type Challenge = Self::F; // TODO: Use an extension.
        type PCS = FRIBasedPCS<Self::F, Self::Challenge, MMCS>;
    }

    struct MulAir;

    // impl<T, W> Air<T, W> for MulAir
    // where
    //     T: AirTypes,
    //     W: AirWindow<T>,
    // {
    //     fn eval<CC>(&self, constraints: &mut CC)
    //     where
    //         CC: ConstraintConsumer<T, W>,
    //     {
    //         let main = constraints.window().main();
    //         let main_local = main.row(0);
    //         let diff = main_local[0] * main_local[1] - main_local[2];
    //         constraints.assert_zero(diff);
    //     }
    // }
    //
    // #[test]
    // fn test_prove() {
    //     let mut rng = thread_rng();
    //     let trace = RowMajorMatrix::rand(&mut rng, 256, 10);
    //     prove::<MyConfig, MulAir>(&MulAir, trace);
    // }
}
