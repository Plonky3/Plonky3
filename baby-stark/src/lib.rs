//! A minimal STARK framework.

#![no_std]

use p3_air::constraint_consumer::ConstraintConsumer;
use p3_air::symbolic::Symbolic;
use p3_air::window::{BasicAirWindow, TwoRowMatrixView};
use p3_air::Air;
use p3_commit::pcs::PCS;
use p3_field::field::{Field, FieldExtension};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

pub trait StarkConfig {
    type F: Field;
    type Challenge: FieldExtension<Self::F>;
    type PCS: PCS<Self::F>;
}

pub struct FoldingConstraintConsumer<'a, F: Field> {
    window: BasicAirWindow<'a, F>,
}

impl<'a, F: Field> ConstraintConsumer<F, BasicAirWindow<'a, F>>
    for FoldingConstraintConsumer<'a, F>
{
    fn window(&self) -> &BasicAirWindow<'a, F> {
        &self.window
    }

    fn assert_zero<I: Into<F>>(&mut self, _constraint: I) {
        todo!()
    }
}

pub fn prove<SC, A>(air: &A, trace: RowMajorMatrix<SC::F>)
where
    SC: StarkConfig,
    for<'a> A: Air<SC::F, BasicAirWindow<'a, SC::F>>
        + Air<Symbolic<SC::F>, BasicAirWindow<'a, Symbolic<SC::F>>>,
{
    for i_local in 0..trace.height() {
        let i_next = (i_local + 1) % trace.height();
        let main_local = trace.row(i_local);
        let main_next = trace.row(i_next);
        let main = TwoRowMatrixView::new(main_local, main_next);
        let window = BasicAirWindow::<SC::F> {
            main,
            is_first_row: SC::F::ZERO,  // TODO
            is_last_row: SC::F::ZERO,   // TODO
            is_transition: SC::F::ZERO, // TODO
        };
        let mut consumer = FoldingConstraintConsumer { window };
        air.eval(&mut consumer);
    }
}

#[cfg(test)]
mod tests {
    use crate::{prove, StarkConfig};
    use p3_air::constraint_consumer::ConstraintConsumer;
    use p3_air::types::AirTypes;
    use p3_air::window::AirWindow;
    use p3_air::Air;
    use p3_fri::FRIBasedPCS;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::Matrix;
    use p3_merkle_tree::MerkleTreeMMCS;
    use p3_mersenne_31::Mersenne31;
    use p3_poseidon::Poseidon;
    use p3_symmetric::compression::TruncatedPermutation;
    use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation, MDSPermutation};
    use p3_symmetric::sponge::PaddingFreeSponge;
    use rand::thread_rng;

    struct MyConfig;

    type F = Mersenne31;
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
        type PCS = FRIBasedPCS<Self::F, Self::Challenge, MMCS>;
    }

    struct MulAir;

    impl<T, W> Air<T, W> for MulAir
    where
        T: AirTypes,
        W: AirWindow<T>,
    {
        fn eval<CC>(&self, constraints: &mut CC)
        where
            CC: ConstraintConsumer<T, W>,
        {
            let main = constraints.window().main();
            let main_local = main.row(0);
            let diff = main_local[0] * main_local[1] - main_local[2];
            constraints.assert_zero(diff);
        }
    }

    #[test]
    fn test_prove() {
        let mut rng = thread_rng();
        let trace = RowMajorMatrix::rand(&mut rng, 256, 10);
        prove::<MyConfig, MulAir>(&MulAir, trace);
    }
}
