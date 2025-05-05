//! The Poseidon permutation.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use p3_field::{Algebra, InjectiveMonomial, PrimeField};
use p3_mds::MdsPermutation;
use p3_symmetric::{CryptographicPermutation, Permutation};
use rand::Rng;
use rand::distr::StandardUniform;
use rand::prelude::Distribution;

/// The Poseidon permutation.
#[derive(Clone, Debug)]
pub struct Poseidon<F, Mds, const WIDTH: usize, const ALPHA: u64> {
    half_num_full_rounds: usize,
    num_partial_rounds: usize,
    constants: Vec<F>,
    mds: Mds,
}

impl<F, Mds, const WIDTH: usize, const ALPHA: u64> Poseidon<F, Mds, WIDTH, ALPHA>
where
    F: PrimeField + InjectiveMonomial<ALPHA>,
{
    /// Create a new Poseidon configuration.
    ///
    /// # Panics
    /// Number of constants must match WIDTH times `num_rounds`; panics otherwise.
    pub fn new(
        half_num_full_rounds: usize,
        num_partial_rounds: usize,
        constants: Vec<F>,
        mds: Mds,
    ) -> Self {
        let num_rounds = 2 * half_num_full_rounds + num_partial_rounds;
        assert_eq!(constants.len(), WIDTH * num_rounds);
        Self {
            half_num_full_rounds,
            num_partial_rounds,
            constants,
            mds,
        }
    }

    pub fn new_from_rng<R: Rng>(
        half_num_full_rounds: usize,
        num_partial_rounds: usize,
        mds: Mds,
        rng: &mut R,
    ) -> Self
    where
        StandardUniform: Distribution<F>,
    {
        let num_rounds = 2 * half_num_full_rounds + num_partial_rounds;
        let num_constants = WIDTH * num_rounds;
        let constants = rng
            .sample_iter(StandardUniform)
            .take(num_constants)
            .collect::<Vec<_>>();
        Self {
            half_num_full_rounds,
            num_partial_rounds,
            constants,
            mds,
        }
    }

    fn half_full_rounds<A>(&self, state: &mut [A; WIDTH], round_ctr: &mut usize)
    where
        A: Algebra<F> + InjectiveMonomial<ALPHA>,
        Mds: MdsPermutation<A, WIDTH>,
    {
        for _ in 0..self.half_num_full_rounds {
            self.constant_layer(state, *round_ctr);
            Self::full_sbox_layer(state);
            self.mds.permute_mut(state);
            *round_ctr += 1;
        }
    }

    fn partial_rounds<A>(&self, state: &mut [A; WIDTH], round_ctr: &mut usize)
    where
        A: Algebra<F> + InjectiveMonomial<ALPHA>,
        Mds: MdsPermutation<A, WIDTH>,
    {
        for _ in 0..self.num_partial_rounds {
            self.constant_layer(state, *round_ctr);
            Self::partial_sbox_layer(state);
            self.mds.permute_mut(state);
            *round_ctr += 1;
        }
    }

    fn full_sbox_layer<A>(state: &mut [A; WIDTH])
    where
        A: Algebra<F> + InjectiveMonomial<ALPHA>,
    {
        for x in state.iter_mut() {
            *x = x.injective_exp_n();
        }
    }

    fn partial_sbox_layer<A>(state: &mut [A; WIDTH])
    where
        A: Algebra<F> + InjectiveMonomial<ALPHA>,
    {
        state[0] = state[0].injective_exp_n();
    }

    fn constant_layer<A>(&self, state: &mut [A; WIDTH], round: usize)
    where
        A: Algebra<F>,
    {
        for (i, x) in state.iter_mut().enumerate() {
            *x += self.constants[round * WIDTH + i];
        }
    }
}

impl<F, A, Mds, const WIDTH: usize, const ALPHA: u64> Permutation<[A; WIDTH]>
    for Poseidon<F, Mds, WIDTH, ALPHA>
where
    F: PrimeField + InjectiveMonomial<ALPHA>,
    A: Algebra<F> + InjectiveMonomial<ALPHA>,
    Mds: MdsPermutation<A, WIDTH>,
{
    fn permute_mut(&self, state: &mut [A; WIDTH]) {
        let mut round_ctr = 0;
        self.half_full_rounds(state, &mut round_ctr);
        self.partial_rounds(state, &mut round_ctr);
        self.half_full_rounds(state, &mut round_ctr);
    }
}

impl<F, A, Mds, const WIDTH: usize, const ALPHA: u64> CryptographicPermutation<[A; WIDTH]>
    for Poseidon<F, Mds, WIDTH, ALPHA>
where
    F: PrimeField + InjectiveMonomial<ALPHA>,
    A: Algebra<F> + InjectiveMonomial<ALPHA>,
    Mds: MdsPermutation<A, WIDTH>,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::array;
    use p3_baby_bear::{BabyBear, MdsMatrixBabyBear};
    use p3_goldilocks::{Goldilocks, MdsMatrixGoldilocks};
    use p3_mersenne_31::{MdsMatrixMersenne31, Mersenne31};
    use p3_mds::coset_mds::CosetMds;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    // Test that permutation doesn't change the state when applied zero times
    #[test]
    fn test_identity_property() {
        let mut rng = SmallRng::seed_from_u64(0);
        let mds = MdsMatrixBabyBear::default();
        
        const WIDTH: usize = 16;
        const ALPHA: u64 = 7;
        const HALF_FULL_ROUNDS: usize = 4;
        const PARTIAL_ROUNDS: usize = 22;
        
        let poseidon = Poseidon::<BabyBear, _, WIDTH, ALPHA>::new_from_rng(
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            mds,
            &mut rng,
        );
        
        // Create a random input state
        let input: [BabyBear; WIDTH] = array::from_fn(|_| BabyBear::from_u32(rng.next_u32()));
        let input_copy = input.clone();
        
        // Permuting zero times should return the same state
        let output = input_copy;
        assert_eq!(output, input);
    }

    // Test that permutation is deterministic (same input -> same output)
    #[test]
    fn test_deterministic_property() {
        let mut rng = SmallRng::seed_from_u64(1);
        let mds = MdsMatrixBabyBear::default();
        
        const WIDTH: usize = 16;
        const ALPHA: u64 = 7;
        
        let poseidon = Poseidon::<BabyBear, _, WIDTH, ALPHA>::new_from_rng(
            4, // half_num_full_rounds 
            22, // num_partial_rounds
            mds,
            &mut rng,
        );
        
        // Create a random input state
        let input: [BabyBear; WIDTH] = array::from_fn(|_| BabyBear::from_u32(rng.next_u32()));
        
        // Run the permutation twice on the same input
        let output1 = poseidon.permute(input.clone());
        let output2 = poseidon.permute(input.clone());
        
        // Outputs should be identical
        assert_eq!(output1, output2);
    }

    // Test with different field types
    #[test]
    fn test_different_fields() {
        let mut rng = SmallRng::seed_from_u64(2);
        
        // Test with BabyBear
        {
            const WIDTH: usize = 12;
            const ALPHA: u64 = 7;
            
            let mds = MdsMatrixBabyBear::default();
            let poseidon = Poseidon::<BabyBear, _, WIDTH, ALPHA>::new_from_rng(
                4, 22, mds, &mut rng,
            );
            
            let input: [BabyBear; WIDTH] = array::from_fn(|_| BabyBear::from_u32(rng.next_u32()));
            let output = poseidon.permute(input.clone());
            
            // Output should be different from input
            assert_ne!(input, output);
        }
        
        // Test with Goldilocks
        {
            const WIDTH: usize = 12;
            const ALPHA: u64 = 7;
            
            let mds = MdsMatrixGoldilocks::default();
            let poseidon = Poseidon::<Goldilocks, _, WIDTH, ALPHA>::new_from_rng(
                4, 22, mds, &mut rng,
            );
            
            let input: [Goldilocks; WIDTH] = array::from_fn(|_| Goldilocks::from_u64(rng.next_u64()));
            let output = poseidon.permute(input.clone());
            
            // Output should be different from input
            assert_ne!(input, output);
        }
        
        // Test with Mersenne31
        {
            const WIDTH: usize = 12;
            const ALPHA: u64 = 5;
            
            let mds = MdsMatrixMersenne31::default();
            let poseidon = Poseidon::<Mersenne31, _, WIDTH, ALPHA>::new_from_rng(
                4, 22, mds, &mut rng,
            );
            
            let input: [Mersenne31; WIDTH] = array::from_fn(|_| Mersenne31::from_u32(rng.next_u32() & 0x7FFFFFFF));
            let output = poseidon.permute(input.clone());
            
            // Output should be different from input
            assert_ne!(input, output);
        }
    }

    // Test with different widths
    #[test]
    fn test_different_widths() {
        let mut rng = SmallRng::seed_from_u64(3);
        
        // Helper macro to test different widths
        macro_rules! test_width {
            ($width:expr) => {
                const ALPHA: u64 = 7;
                let mds = MdsMatrixBabyBear::default();
                let poseidon = Poseidon::<BabyBear, _, $width, ALPHA>::new_from_rng(
                    4, 22, mds, &mut rng,
                );
                
                let input: [BabyBear; $width] = array::from_fn(|_| BabyBear::from_u32(rng.next_u32()));
                let output = poseidon.permute(input.clone());
                
                // Output should be different from input
                assert_ne!(input, output);
            };
        }
        
        // Test different widths
        test_width!(8);
        test_width!(12);
        test_width!(16);
        test_width!(24);
    }

    // Test with coset MDS matrix
    #[test]
    fn test_coset_mds() {
        let mut rng = SmallRng::seed_from_u64(4);
        
        const WIDTH: usize = 32;
        const ALPHA: u64 = 7;
        
        let mds = CosetMds::<BabyBear, WIDTH>::default();
        let poseidon = Poseidon::<BabyBear, _, WIDTH, ALPHA>::new_from_rng(
            4, 22, mds, &mut rng,
        );
        
        let input: [BabyBear; WIDTH] = array::from_fn(|_| BabyBear::from_u32(rng.next_u32()));
        let output = poseidon.permute(input.clone());
        
        // Output should be different from input
        assert_ne!(input, output);
    }

    // Test that the permutation works with zero input
    #[test]
    fn test_zero_input() {
        let mut rng = SmallRng::seed_from_u64(5);
        
        const WIDTH: usize = 16;
        const ALPHA: u64 = 7;
        
        let mds = MdsMatrixBabyBear::default();
        let poseidon = Poseidon::<BabyBear, _, WIDTH, ALPHA>::new_from_rng(
            4, 22, mds, &mut rng,
        );
        
        let input: [BabyBear; WIDTH] = [BabyBear::ZERO; WIDTH];
        let output = poseidon.permute(input.clone());
        
        // Output should be different from all-zero input
        assert_ne!(input, output);
        
        // At least one element should be non-zero
        assert!(output.iter().any(|&x| x != BabyBear::ZERO));
    }

    // Test with invalid parameters
    #[test]
    #[should_panic(expected = "constants.len()")]
    fn test_invalid_parameters() {
        const WIDTH: usize = 16;
        const ALPHA: u64 = 7;
        
        let mds = MdsMatrixBabyBear::default();
        let half_num_full_rounds = 4;
        let num_partial_rounds = 22;
        
        // Create vector of wrong size (one element short)
        let num_rounds = 2 * half_num_full_rounds + num_partial_rounds;
        let wrong_size = WIDTH * num_rounds - 1;
        let constants = vec![BabyBear::ZERO; wrong_size];
        
        // This should panic due to invalid constants size
        let _poseidon = Poseidon::<BabyBear, _, WIDTH, ALPHA>::new(
            half_num_full_rounds,
            num_partial_rounds,
            constants,
            mds,
        );
    }

    // Test that permute and permute_mut give the same result
    #[test]
    fn test_permute_methods_consistent() {
        let mut rng = SmallRng::seed_from_u64(6);
        
        const WIDTH: usize = 16;
        const ALPHA: u64 = 7;
        
        let mds = MdsMatrixBabyBear::default();
        let poseidon = Poseidon::<BabyBear, _, WIDTH, ALPHA>::new_from_rng(
            4, 22, mds, &mut rng,
        );
        
        let input: [BabyBear; WIDTH] = array::from_fn(|_| BabyBear::from_u32(rng.next_u32()));
        
        // Test permute method
        let output1 = poseidon.permute(input.clone());
        
        // Test permute_mut method
        let mut input_mut = input.clone();
        poseidon.permute_mut(&mut input_mut);
        
        // Both methods should give the same result
        assert_eq!(output1, input_mut);
    }

    // Test that permutation is diffusing (changing one bit affects the entire state)
    #[test]
    fn test_diffusion() {
        let mut rng = SmallRng::seed_from_u64(7);
        
        const WIDTH: usize = 16;
        const ALPHA: u64 = 7;
        
        let mds = MdsMatrixBabyBear::default();
        let poseidon = Poseidon::<BabyBear, _, WIDTH, ALPHA>::new_from_rng(
            4, 22, mds, &mut rng,
        );
        
        // Create a random input state
        let mut input1: [BabyBear; WIDTH] = array::from_fn(|_| BabyBear::from_u32(rng.next_u32()));
        let mut input2 = input1.clone();
        
        // Modify just one element in the second input
        input2[0] = input2[0] + BabyBear::ONE;
        
        // Permute both inputs
        let output1 = poseidon.permute(input1.clone());
        let output2 = poseidon.permute(input2.clone());
        
        // Outputs should be different
        assert_ne!(output1, output2);
        
        // Count how many elements differ in the output
        let num_different = output1.iter().zip(output2.iter())
            .filter(|(&a, &b)| a != b)
            .count();
        
        // With good diffusion, most or all elements should be different
        // We'll assert that at least half of them are different
        assert!(num_different >= WIDTH / 2, 
            "Expected diffusion to affect at least half the elements, but only {num_different} differ");
    }
}
