use alloc::string::String;
use itertools::Itertools;
use core::marker::PhantomData;

use p3_field::{
    PrimeField, PrimeField32, SpongePaddingValue, absorb_radix_bits,
    max_shifted_absorb_injective_limbs, reduce_packed_shifted,
};

use crate::hasher::CryptographicHasher;
use crate::permutation::CryptographicPermutation;

/// A padded, overwrite-mode sponge function.
///
/// `WIDTH` is the sponge's rate plus the sponge's capacity.
#[derive(Copy, Clone, Debug)]
pub struct PaddingFreeSponge<P, const WIDTH: usize, const RATE: usize, const OUT: usize> {
    permutation: P,
}

impl<P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    PaddingFreeSponge<P, WIDTH, RATE, OUT>
{
    pub const fn new(permutation: P) -> Self {
        const {
            assert!(RATE > 0);
            assert!(RATE < WIDTH);
            assert!(OUT <= WIDTH);
        }

        Self { permutation }
    }
}

impl<T, P, const WIDTH: usize, const RATE: usize, const OUT: usize> CryptographicHasher<T, [T; OUT]>
    for PaddingFreeSponge<P, WIDTH, RATE, OUT>
where
    T: Default + SpongePaddingValue,
    P: CryptographicPermutation<[T; WIDTH]>,
{
    fn hash_iter<I>(&self, input: I) -> [T; OUT]
    where
        I: IntoIterator<Item = T>,
    {
        const {
            assert!(RATE > 0);
            assert!(RATE < WIDTH);
            assert!(OUT <= WIDTH);
        }
        // Start from the all-zero state.
        let mut state = [T::default(); WIDTH];
        let mut input = input.into_iter();

        'outer: loop {
            // Absorb one block: overwrite state[0..RATE] with input elements one at a time.
            for i in 0..RATE {
                if let Some(x) = input.next() {
                    // Overwrite the i-th rate position.
                    state[i] = x;
                } else {
                    // Input exhausted mid-block. Permute only if at least
                    // one element was absorbed in this block (i > 0).
                    // If i == 0 the state already reflects the previous
                    // permutation output and needs no extra call.
                    if i != 0 {
                        self.permutation.permute_mut(&mut state);
                    }
                    break 'outer;
                }
            }

            // Full block absorbed. Permute before the next block.
            self.permutation.permute_mut(&mut state);
        }

        // Squeeze: return the first OUT elements of the final state.
        state[..OUT].try_into().unwrap()
    }
}

/// Padding-free sponge over a large prime field, accepting 32-bit field elements as input.
///
/// # Security
///
/// **Not** collision-resistant for variable-length inputs.
#[derive(Clone, Debug)]
pub struct MultiField32PaddingFreeSponge<
    F,
    PF,
    P,
    const WIDTH: usize,
    const RATE: usize,
    const OUT: usize,
> {
    /// The cryptographic permutation applied after each absorbed block.
    permutation: P,
    /// How many small-field elements fit inside one large-field element.
    num_f_elms: usize,
    /// Radix used for shifted packing into the large field.
    radix_bits: u32,
    _phantom: PhantomData<(F, PF)>,
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    MultiField32PaddingFreeSponge<F, PF, P, WIDTH, RATE, OUT>
where
    F: PrimeField32,
    PF: PrimeField,
{
    pub fn new(permutation: P) -> Result<Self, String> {
        const {
            assert!(RATE > 0);
            assert!(RATE < WIDTH);
            assert!(OUT <= WIDTH);
        }

        if F::order() >= PF::order() {
            return Err(String::from("F::order() must be less than PF::order()"));
        }

        // Use shifted-radix injective packing for robust absorb encoding.
        let num_f_elms = max_shifted_absorb_injective_limbs::<F, PF>();
        let radix_bits = absorb_radix_bits::<F>();
        Ok(Self {
            permutation,
            num_f_elms,
            radix_bits,
            _phantom: PhantomData,
        })
    }
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    CryptographicHasher<F, [PF; OUT]> for MultiField32PaddingFreeSponge<F, PF, P, WIDTH, RATE, OUT>
where
    F: PrimeField32,
    PF: PrimeField + Default + Copy,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn hash_iter<I>(&self, input: I) -> [PF; OUT]
    where
        I: IntoIterator<Item = F>,
    {
        const {
            assert!(RATE > 0);
            assert!(RATE < WIDTH);
            assert!(OUT <= WIDTH);
        }
        let mut state = [PF::default(); WIDTH];

        // Example: RATE = 3, num_f_elms = 2, input = [f0..f7]
        //
        //   block_chunk = [f0, f1, f2, f3, f4, f5]  (RATE * 2 = 6 small elems)
        //     chunk 0: [f0, f1] -> pack into PF -> state[0]
        //     chunk 1: [f2, f3] -> pack into PF -> state[1]
        //     chunk 2: [f4, f5] -> pack into PF -> state[2]
        //   -> permute
        //
        //   block_chunk = [f6, f7]  (partial)
        //     chunk 0: [f6, f7] -> pack into PF -> state[0]
        //   -> permute
        for block_chunk in &input.into_iter().chunks(RATE * self.num_f_elms) {
            for (chunk_id, chunk) in (&block_chunk.chunks(self.num_f_elms))
                .into_iter()
                .enumerate()
            {
                // Pack num_f_elms small-field elements into one large-field
                // element via shifted-radix reduction.
                state[chunk_id] = reduce_packed_shifted(&chunk.collect_vec(), self.radix_bits);
            }
            state = self.permutation.permute(state);
        }

        state[..OUT].try_into().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use p3_field::{PrimeCharacteristicRing, absorb_radix_bits, reduce_32, reduce_packed_shifted};
    use p3_goldilocks::Goldilocks;
    use p3_koala_bear::KoalaBear;

    use super::*;
    use crate::Permutation;

    #[derive(Clone)]
    struct MockPermutation;

    impl<T, const WIDTH: usize> Permutation<[T; WIDTH]> for MockPermutation
    where
        T: Copy + core::ops::Add<Output = T> + Default,
    {
        fn permute_mut(&self, input: &mut [T; WIDTH]) {
            let sum: T = input.iter().copied().fold(T::default(), |acc, x| acc + x);
            // Set every element to the sum
            *input = [sum; WIDTH];
        }
    }

    impl<T, const WIDTH: usize> CryptographicPermutation<[T; WIDTH]> for MockPermutation where
        T: Copy + core::ops::Add<Output = T> + Default
    {
    }

    #[derive(Clone)]
    struct IdentityPermutation;

    impl<T: Clone, const WIDTH: usize> Permutation<[T; WIDTH]> for IdentityPermutation {
        fn permute_mut(&self, _input: &mut [T; WIDTH]) {}
    }

    impl<T: Clone, const WIDTH: usize> CryptographicPermutation<[T; WIDTH]> for IdentityPermutation {}

    #[test]
    fn test_padding_free_sponge_basic() {
        const WIDTH: usize = 4;
        const RATE: usize = 2;
        const OUT: usize = 2;

        let permutation = MockPermutation;
        let sponge = PaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(permutation);

        let input = [1, 2, 3, 4, 5];
        let output = sponge.hash_iter(input);

        // Explanation of why the final state results in [44, 44, 44, 44]:
        // Initial state: [0, 0, 0, 0]
        // First input chunk [1, 2] overwrites first two positions: [1, 2, 0, 0]
        // Apply permutation (sum all elements and overwrite): [3, 3, 3, 3]
        // Second input chunk [3, 4] overwrites first two positions: [3, 4, 3, 3]
        // Apply permutation: [13, 13, 13, 13] (3 + 4 + 3 + 3 = 13)
        // Third input chunk [5] overwrites first position: [5, 13, 13, 13]
        // Apply permutation: [44, 44, 44, 44] (5 + 13 + 13 + 13 = 44)

        assert_eq!(output, [44; OUT]);
    }

    #[test]
    fn test_padding_free_sponge_empty_input() {
        // Empty input: no elements absorbed, no permutation called.
        //
        // The initial all-zero state is returned directly.
        const WIDTH: usize = 4;
        const RATE: usize = 2;
        const OUT: usize = 2;

        let sponge = PaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);

        let input: [u64; 0] = [];
        let output = sponge.hash_iter(input);

        // Squeeze from the untouched zero state.
        assert_eq!(output, [0; OUT]);
    }

    #[test]
    fn test_padding_free_sponge_exact_block_size() {
        const WIDTH: usize = 6;
        const RATE: usize = 3;
        const OUT: usize = 2;

        let permutation = MockPermutation;
        let sponge = PaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(permutation);

        let input = [10, 20, 30];
        let output = sponge.hash_iter(input);

        assert_eq!(output, [60; OUT]);
    }

    #[test]
    fn test_multi_field32_padding_free_sponge_uses_absorb_radix() {
        const WIDTH: usize = 5;
        const RATE: usize = 4;
        const OUT: usize = 1;

        type F = KoalaBear;
        type PF = Goldilocks;

        let sponge =
            MultiField32PaddingFreeSponge::<F, PF, _, WIDTH, RATE, OUT>::new(IdentityPermutation)
                .unwrap();

        let input = [F::from_u32(1 << 30), F::ONE];
        let output = sponge.hash_iter(input);
        let expected = [reduce_packed_shifted::<F, PF>(
            &input,
            absorb_radix_bits::<F>(),
        )];

        assert_eq!(output, expected);
        assert_ne!(output[0], reduce_32::<F, PF>(&input));
    }

    #[test]
    fn test_multi_field32_padding_free_sponge_fills_full_pf_rate_rows() {
        const WIDTH: usize = 6;
        const RATE: usize = 5;
        const OUT: usize = 4;

        type F = KoalaBear;
        type PF = Goldilocks;

        let sponge =
            MultiField32PaddingFreeSponge::<F, PF, _, WIDTH, RATE, OUT>::new(IdentityPermutation)
                .unwrap();

        let input = core::array::from_fn::<_, 8, _>(|i| F::from_u32((i + 1) as u32));
        let radix_bits = absorb_radix_bits::<F>();
        let packed = [
            reduce_packed_shifted::<F, PF>(&input[0..2], radix_bits),
            reduce_packed_shifted::<F, PF>(&input[2..4], radix_bits),
            reduce_packed_shifted::<F, PF>(&input[4..6], radix_bits),
            reduce_packed_shifted::<F, PF>(&input[6..8], radix_bits),
        ];

        assert_eq!(sponge.num_f_elms, 2);
        assert_eq!(sponge.hash_iter(input), packed);
    }

    #[test]
    fn test_multi_field32_padding_free_sponge_distinguishes_trailing_zero_in_slot() {
        const WIDTH: usize = 2;
        const RATE: usize = 1;
        const OUT: usize = 1;

        type F = KoalaBear;
        type PF = Goldilocks;

        let sponge =
            MultiField32PaddingFreeSponge::<F, PF, _, WIDTH, RATE, OUT>::new(MockPermutation)
                .unwrap();

        assert_eq!(sponge.num_f_elms, 2);
        assert_ne!(
            sponge.hash_iter([F::ONE]),
            sponge.hash_iter([F::ONE, F::ZERO])
        );
    }
}
