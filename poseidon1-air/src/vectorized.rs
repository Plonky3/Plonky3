//! Vectorized Poseidon1 AIR for computing multiple permutations per row.
//!
//! # Overview
//!
//! When the per-permutation column count is small relative to the desired trace
//! width, packing multiple permutations into a single row improves prover
//! performance by increasing the utilization of each NTT.
//!
//! # Layout
//!
//! Each row contains `VECTOR_LEN` independent `Poseidon1Cols` structures
//! laid out contiguously:
//!
//! ```text
//!   Row:  [ Poseidon1Cols_0 | Poseidon1Cols_1 | ... | Poseidon1Cols_{VECTOR_LEN-1} ]
//!         |<----- ncols = VECTOR_LEN × cols_per_perm ----->|
//! ```
//!
//! Each `Poseidon1Cols` block is constrained independently using the same
//! round constants.
//!
//! # Usage
//!
//! ```text
//!   let air = VectorizedPoseidon1Air::<F, 16, 7, 1, 4, 13, 8>::new(constants);
//!   let trace = air.generate_vectorized_trace_rows(num_perms, log_blowup);
//! ```

use alloc::vec;
use alloc::vec::Vec;
use core::borrow::{Borrow, BorrowMut};

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{PrimeCharacteristicRing, PrimeField};
use p3_matrix::dense::RowMajorMatrix;
use rand::distr::StandardUniform;
use rand::prelude::Distribution;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::air::eval;
use crate::{
    FullRoundConstants, PartialRoundConstants, Poseidon1Air, Poseidon1Cols,
    generate_vectorized_trace_rows,
};

/// Column layout for a vectorized Poseidon1 row.
///
/// Contains `VECTOR_LEN` independent `Poseidon1Cols` blocks laid out contiguously.
/// The `#[repr(C)]` attribute ensures predictable layout for `Borrow` reinterpretation.
#[repr(C)]
pub struct VectorizedPoseidon1Cols<
    T,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> {
    /// Array of `VECTOR_LEN` independent permutation column blocks.
    ///
    /// Each block is constrained independently during AIR evaluation.
    pub(crate) cols:
        [Poseidon1Cols<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>;
            VECTOR_LEN],
}

impl<
    T,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
>
    Borrow<
        VectorizedPoseidon1Cols<
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
    ) -> &VectorizedPoseidon1Cols<
        T,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    > {
        let (prefix, shorts, suffix) = unsafe {
            self.align_to::<VectorizedPoseidon1Cols<
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
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
>
    BorrowMut<
        VectorizedPoseidon1Cols<
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
    ) -> &mut VectorizedPoseidon1Cols<
        T,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    > {
        let (prefix, shorts, suffix) = unsafe {
            self.align_to_mut::<VectorizedPoseidon1Cols<
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

/// Vectorized Poseidon1 AIR.
///
/// Wraps a standard `Poseidon1Air` and applies it `VECTOR_LEN` times per row.
/// All permutations within a row share the same round constants but operate
/// on independent state inputs.
pub struct VectorizedPoseidon1Air<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> {
    /// The underlying single-permutation AIR (holds the round constants).
    pub(crate) air:
        Poseidon1Air<F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
}

impl<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
>
    VectorizedPoseidon1Air<
        F,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >
{
    /// Construct a vectorized AIR from pre-computed round constants.
    pub fn new(
        full_constants: FullRoundConstants<F, WIDTH>,
        partial_constants: PartialRoundConstants<F, WIDTH>,
    ) -> Self {
        Self {
            air: Poseidon1Air::new(full_constants, partial_constants),
        }
    }

    /// Generate a vectorized trace with `num_hashes` random permutations.
    ///
    /// Uses a deterministic PRNG seeded with `1` for reproducible traces.
    /// The permutations are packed `VECTOR_LEN` per row.
    pub fn generate_vectorized_trace_rows(
        &self,
        num_hashes: usize,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F>
    where
        F: PrimeField,
        StandardUniform: Distribution<[F; WIDTH]>,
    {
        let mut rng = SmallRng::seed_from_u64(1);
        let inputs = (0..num_hashes).map(|_| rng.random()).collect();
        generate_vectorized_trace_rows::<
            _,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            VECTOR_LEN,
        >(
            inputs,
            &self.air.full_constants,
            &self.air.partial_constants,
            extra_capacity_bits,
        )
    }
}

impl<
    F: PrimeCharacteristicRing + Sync,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> BaseAir<F>
    for VectorizedPoseidon1Air<
        F,
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

    /// No next-row columns. All permutations are fully constrained within one row.
    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}

impl<
    AB: AirBuilder,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> Air<AB>
    for VectorizedPoseidon1Air<
        AB::F,
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
        // Read the current row as a flat slice.
        let main = builder.main();

        // Reinterpret as the vectorized column struct.
        let local: &VectorizedPoseidon1Cols<
            _,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            VECTOR_LEN,
        > = main.current_slice().borrow();

        // Evaluate constraints independently for each permutation in the row.
        for perm in &local.cols {
            eval(&self.air, builder, perm);
        }
    }
}
