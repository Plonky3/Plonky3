use core::borrow::{Borrow, BorrowMut};

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;
use p3_poseidon2::{DiffusionPermutation, MdsLightPermutation};

use crate::air::eval;
use crate::constants::RoundConstants;
use crate::{Poseidon2Air, Poseidon2Cols};

/// A "vectorized" version of Poseidon2Cols, for computing multiple Poseidon2 permutations per row.
#[repr(C)]
pub struct VectorizedPoseidon2Cols<
    T,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> {
    pub(crate) cols:
        [Poseidon2Cols<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>;
            VECTOR_LEN],
}

impl<
        T,
        const WIDTH: usize,
        const SBOX_DEGREE: usize,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
        const VECTOR_LEN: usize,
    >
    Borrow<
        VectorizedPoseidon2Cols<
            T,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            VECTOR_LEN,
        >,
    > for [T]
{
    fn borrow(
        &self,
    ) -> &VectorizedPoseidon2Cols<
        T,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    > {
        // debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe {
            self.align_to::<VectorizedPoseidon2Cols<
                T,
                WIDTH,
                SBOX_DEGREE,
                SBOX_REGISTERS,
                HALF_FULL_ROUNDS,
                PARTIAL_ROUNDS,
                VECTOR_LEN,
            >>()
        };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<
        T,
        const WIDTH: usize,
        const SBOX_DEGREE: usize,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
        const VECTOR_LEN: usize,
    >
    BorrowMut<
        VectorizedPoseidon2Cols<
            T,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            VECTOR_LEN,
        >,
    > for [T]
{
    fn borrow_mut(
        &mut self,
    ) -> &mut VectorizedPoseidon2Cols<
        T,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    > {
        // debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe {
            self.align_to_mut::<VectorizedPoseidon2Cols<
                T,
                WIDTH,
                SBOX_DEGREE,
                SBOX_REGISTERS,
                HALF_FULL_ROUNDS,
                PARTIAL_ROUNDS,
                VECTOR_LEN,
            >>()
        };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

/// A "vectorized" version of Poseidon2Air, for computing multiple Poseidon2 permutations per row.
pub struct VectorizedPoseidon2Air<
    F: Field,
    MdsLight,
    Diffusion,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> {
    air: Poseidon2Air<
        F,
        MdsLight,
        Diffusion,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
}

impl<
        F: Field,
        MdsLight,
        Diffusion,
        const WIDTH: usize,
        const SBOX_DEGREE: usize,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
        const VECTOR_LEN: usize,
    >
    VectorizedPoseidon2Air<
        F,
        MdsLight,
        Diffusion,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >
{
    pub fn new(
        constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
        external_linear_layer: MdsLight,
        internal_linear_layer: Diffusion,
    ) -> Self {
        Self {
            air: Poseidon2Air::new(constants, external_linear_layer, internal_linear_layer),
        }
    }
}

impl<
        F: Field,
        MdsLight: Sync,
        Diffusion: Sync,
        const WIDTH: usize,
        const SBOX_DEGREE: usize,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
        const VECTOR_LEN: usize,
    > BaseAir<F>
    for VectorizedPoseidon2Air<
        F,
        MdsLight,
        Diffusion,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >
{
    fn width(&self) -> usize {
        self.air.width() * VECTOR_LEN
    }
}

impl<
        AB: AirBuilder,
        MdsLight: MdsLightPermutation<AB::Expr, WIDTH>,
        Diffusion: DiffusionPermutation<AB::Expr, WIDTH>,
        const WIDTH: usize,
        const SBOX_DEGREE: usize,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
        const VECTOR_LEN: usize,
    > Air<AB>
    for VectorizedPoseidon2Air<
        AB::F,
        MdsLight,
        Diffusion,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >
{
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &VectorizedPoseidon2Cols<
            AB::Var,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            VECTOR_LEN,
        > = (*local).borrow();
        for perm in &local.cols {
            eval(&self.air, builder, perm);
        }
    }
}
