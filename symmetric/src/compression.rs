use crate::hasher::CryptographicHasher;
use crate::permutation::CryptographicPermutation;

/// An `N`-to-1 compression function collision-resistant in a hash tree setting.
///
/// Unlike `CompressionFunction`, it may not be collision-resistant in general.
/// Instead it is only collision-resistant in hash-tree like settings where
/// the preimage of a non-leaf node must consist of compression outputs.
pub trait PseudoCompressionFunction<T, const N: usize>: Clone {
    /// Number of independent compressions this implementation performs most efficiently in one call.
    ///
    /// One means every group is compressed on its own, which is the behaviour of a plain scalar permutation.
    /// A vectorized implementation reports how many independent states its permutation advances at once.
    ///
    /// Callers read this only to decide whether grouping compressions is worth the bookkeeping.
    const LANES: usize = 1;

    fn compress(&self, input: [T; N]) -> T;

    /// Compress a batch of input groups, one output per group.
    ///
    /// ```text
    ///     inputs: [ grp_0 | grp_1 | ... | grp_{m-1} ]   m groups of N inputs
    ///     out:    [ dig_0 | dig_1 | ... | dig_{m-1} ]   m outputs
    /// ```
    ///
    /// The default compresses the groups one at a time.
    /// An override exists purely to exploit vector hardware and must return the very same outputs.
    ///
    /// Groups beyond the shorter of the two slices are ignored, so the caller controls the count.
    fn compress_many(&self, inputs: &[[T; N]], out: &mut [T])
    where
        T: Clone,
    {
        // Walk the groups in order so the outputs land in the caller's order.
        for (output, group) in out.iter_mut().zip(inputs) {
            *output = self.compress(group.clone());
        }
    }
}

/// An `N`-to-1 compression function.
pub trait CompressionFunction<T, const N: usize>: PseudoCompressionFunction<T, N> {}

#[derive(Clone, Debug)]
pub struct TruncatedPermutation<InnerP, const N: usize, const CHUNK: usize, const WIDTH: usize> {
    inner_permutation: InnerP,
}

impl<InnerP, const N: usize, const CHUNK: usize, const WIDTH: usize>
    TruncatedPermutation<InnerP, N, CHUNK, WIDTH>
{
    pub const fn new(inner_permutation: InnerP) -> Self {
        const {
            assert!(N > 0);
            assert!(CHUNK > 0);
            assert!(CHUNK * N <= WIDTH);
        }
        Self { inner_permutation }
    }
}

impl<T, InnerP, const N: usize, const CHUNK: usize, const WIDTH: usize>
    PseudoCompressionFunction<[T; CHUNK], N> for TruncatedPermutation<InnerP, N, CHUNK, WIDTH>
where
    T: Copy + Default,
    InnerP: CryptographicPermutation<[T; WIDTH]>,
{
    fn compress(&self, input: [[T; CHUNK]; N]) -> [T; CHUNK] {
        let mut pre = [T::default(); WIDTH];
        for i in 0..N {
            pre[i * CHUNK..(i + 1) * CHUNK].copy_from_slice(&input[i]);
        }
        let post = self.inner_permutation.permute(pre);
        post[..CHUNK].try_into().unwrap()
    }
}

#[derive(Clone, Debug)]
pub struct CompressionFunctionFromHasher<H, const N: usize, const CHUNK: usize> {
    hasher: H,
}

impl<H, const N: usize, const CHUNK: usize> CompressionFunctionFromHasher<H, N, CHUNK> {
    pub const fn new(hasher: H) -> Self {
        Self { hasher }
    }
}

impl<T, H, const N: usize, const CHUNK: usize> PseudoCompressionFunction<[T; CHUNK], N>
    for CompressionFunctionFromHasher<H, N, CHUNK>
where
    T: Clone,
    H: CryptographicHasher<T, [T; CHUNK]>,
{
    const LANES: usize = <H as CryptographicHasher<T, [T; CHUNK]>>::LANES;

    fn compress(&self, input: [[T; CHUNK]; N]) -> [T; CHUNK] {
        self.hasher.hash_iter(input.into_iter().flatten())
    }

    fn compress_many(&self, inputs: &[[[T; CHUNK]; N]], out: &mut [[T; CHUNK]]) {
        // A group is `N` adjacent chunks of `CHUNK` items, so a run of groups is already one
        // flat run of items with nothing between the groups:
        //
        //     inputs: [[c0 c1] [c2 c3] ...]  ->  flat: [c0 c1 c2 c3 ...]
        //
        // Flattening twice therefore hands the hasher exactly the concatenated preimages that
        // the single-group path builds one group at a time, with no copying.
        let messages = inputs.as_flattened().as_flattened();

        // Each message is `N * CHUNK` items long, so the batch hasher can split them itself.
        // Trim the input to the number of requested outputs to keep that split exact.
        let requested = out.len().min(inputs.len());
        self.hasher
            .hash_many(&messages[..requested * N * CHUNK], &mut out[..requested]);
    }
}

impl<T, H, const N: usize, const CHUNK: usize> CompressionFunction<[T; CHUNK], N>
    for CompressionFunctionFromHasher<H, N, CHUNK>
where
    T: Clone,
    H: CryptographicHasher<T, [T; CHUNK]>,
{
}

#[cfg(test)]
mod tests {
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
            // Simplest impl: set every element to the sum
            *input = [sum; WIDTH];
        }
    }

    impl<T, const WIDTH: usize> CryptographicPermutation<[T; WIDTH]> for MockPermutation where
        T: Copy + core::ops::Add<Output = T> + Default
    {
    }

    #[derive(Clone)]
    struct MockHasher;

    impl<const CHUNK: usize> CryptographicHasher<u64, [u64; CHUNK]> for MockHasher {
        fn hash_iter<I: IntoIterator<Item = u64>>(&self, iter: I) -> [u64; CHUNK] {
            let sum: u64 = iter.into_iter().sum();
            // Simplest impl: set every element to the sum
            [sum; CHUNK]
        }
    }

    #[test]
    fn test_truncated_permutation_compress() {
        const N: usize = 2;
        const CHUNK: usize = 4;
        const WIDTH: usize = 8;

        let permutation = MockPermutation;
        let compressor = TruncatedPermutation::<MockPermutation, N, CHUNK, WIDTH>::new(permutation);

        let input: [[u64; CHUNK]; N] = [[1, 2, 3, 4], [5, 6, 7, 8]];
        let output = compressor.compress(input);
        let expected_sum = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8;

        assert_eq!(output, [expected_sum; CHUNK]);
    }

    #[test]
    fn test_compression_function_from_hasher_compress() {
        const N: usize = 2;
        const CHUNK: usize = 4;

        let hasher = MockHasher;
        let compressor = CompressionFunctionFromHasher::<MockHasher, N, CHUNK>::new(hasher);

        let input = [[10, 20, 30, 40], [50, 60, 70, 80]];
        let output = compressor.compress(input);
        let expected_sum = 10 + 20 + 30 + 40 + 50 + 60 + 70 + 80;

        assert_eq!(output, [expected_sum; CHUNK]);
    }

    #[test]
    fn test_truncated_permutation_with_zeros() {
        const N: usize = 2;
        const CHUNK: usize = 4;
        const WIDTH: usize = 8;

        let permutation = MockPermutation;
        let compressor = TruncatedPermutation::<MockPermutation, N, CHUNK, WIDTH>::new(permutation);

        let input: [[u64; CHUNK]; N] = [[0, 0, 0, 0], [0, 0, 0, 0]];
        let output = compressor.compress(input);

        assert_eq!(output, [0; CHUNK]);
    }

    #[test]
    fn test_truncated_permutation_with_extra_width() {
        const N: usize = 2;
        const CHUNK: usize = 3;
        const WIDTH: usize = 10; // More than `CHUNK * N` (6 < 10)

        let permutation = MockPermutation;
        let compressor = TruncatedPermutation::<MockPermutation, N, CHUNK, WIDTH>::new(permutation);

        let input: [[u64; CHUNK]; N] = [[1, 2, 3], [4, 5, 6]];
        let output = compressor.compress(input);

        let expected_sum = 1 + 2 + 3 + 4 + 5 + 6;

        assert_eq!(
            output, [expected_sum; CHUNK],
            "Compression should correctly handle extra WIDTH space."
        );
    }
}
