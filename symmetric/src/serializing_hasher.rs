use p3_field::Field;

use crate::CryptographicHasher;

/// Size of the stack buffer that holds the serialized bytes of one group of rows.
///
/// The batched inner hash needs its messages back to back in memory, and a field element only
/// becomes bytes through a serializing iterator, so the bytes have to be materialized somewhere.
/// 8 KiB stays inside a typical 32 KiB L1 data cache and still holds eight messages for rows of
/// up to 1 KiB, which keeps every lane of a vector sponge busy at the widths a Merkle leaf uses.
const ROW_BYTES_SCRATCH: usize = 8 * 1024;

/// Converts a hasher which can hash bytes, u32's or u64's into a hasher which can hash field elements.
///
/// Supports two types of hashing.
/// - Hashing a sequence of field elements.
/// - Hashing a sequence of arrays of `N` field elements as if we are hashing `N` sequences of field elements in parallel.
///   This is useful when the inner hash is able to use vectorized instructions to compute multiple hashes at once.
#[derive(Copy, Clone, Debug)]
pub struct SerializingHasher<Inner> {
    inner: Inner,
}

impl<Inner> SerializingHasher<Inner> {
    pub const fn new(inner: Inner) -> Self {
        Self { inner }
    }
}

impl<F, Inner, const N: usize> CryptographicHasher<F, [u8; N]> for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<u8, [u8; N]>,
{
    const LANES: usize = <Inner as CryptographicHasher<u8, [u8; N]>>::LANES;

    fn hash_iter<I>(&self, input: I) -> [u8; N]
    where
        I: IntoIterator<Item = F>,
    {
        self.inner.hash_iter(F::into_byte_stream(input))
    }

    fn hash_many(&self, input: &[F], out: &mut [[u8; N]]) {
        // No digests requested means there is nothing to read from the input.
        if out.is_empty() {
            return;
        }

        // Every message has the same length, so the split into rows is exact by contract.
        assert!(
            input.len().is_multiple_of(out.len()),
            "input length ({}) must be a whole multiple of the digest count ({})",
            input.len(),
            out.len()
        );
        let row_len = input.len() / out.len();
        let row_bytes = row_len * F::NUM_BYTES;

        // Width-zero rows all hash to the same digest of the empty message.
        if row_len == 0 {
            for digest in out.iter_mut() {
                *digest = self.hash_iter(core::iter::empty::<F>());
            }
            return;
        }

        // A row too wide for the stack buffer gains nothing from grouping, and the unbatched
        // path serializes straight into the sponge with no buffer at all.
        if row_bytes > ROW_BYTES_SCRATCH {
            for (digest, row) in out.iter_mut().zip(input.chunks_exact(row_len)) {
                *digest = self.hash_iter(row.iter().copied());
            }
            return;
        }

        // Serialize as many whole rows at a time as the buffer holds, so the inner hasher sees a
        // long run of equal-length byte messages and can fill every lane of its permutation:
        //
        //     rows:    [ r_0 | r_1 | ... ]              row_len field elements each
        //     scratch: [ b_0 | b_1 | ... ]              row_bytes bytes each
        let rows_per_group = ROW_BYTES_SCRATCH / row_bytes;
        let mut scratch = [0u8; ROW_BYTES_SCRATCH];

        for (rows, digests) in input
            .chunks(rows_per_group * row_len)
            .zip(out.chunks_mut(rows_per_group))
        {
            // Only the leading part of the buffer is used when the final group is short.
            let bytes = &mut scratch[..digests.len() * row_bytes];

            // Serialize row by row so each message starts on its own row boundary.
            //
            // The row boundaries are computed from the declared serialized width of one field
            // element, so a stream of any other length would shift every later message and
            // silently change its digest; both directions are therefore checked.
            for (slot, row) in bytes.chunks_mut(row_bytes).zip(rows.chunks(row_len)) {
                let mut stream = F::into_byte_stream(row.iter().copied()).into_iter();
                for dst in slot.iter_mut() {
                    *dst = stream
                        .next()
                        .expect("field serialization is shorter than its declared width");
                }
                assert!(
                    stream.next().is_none(),
                    "field serialization is longer than its declared width"
                );
            }

            self.inner.hash_many(bytes, digests);
        }
    }
}

impl<F, Inner, const N: usize> CryptographicHasher<F, [u32; N]> for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<u32, [u32; N]>,
{
    fn hash_iter<I>(&self, input: I) -> [u32; N]
    where
        I: IntoIterator<Item = F>,
    {
        self.inner.hash_iter(F::into_u32_stream(input))
    }
}

impl<F, Inner, const N: usize> CryptographicHasher<F, [u64; N]> for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<u64, [u64; N]>,
{
    fn hash_iter<I>(&self, input: I) -> [u64; N]
    where
        I: IntoIterator<Item = F>,
    {
        self.inner.hash_iter(F::into_u64_stream(input))
    }
}

impl<F, Inner, const N: usize, const M: usize> CryptographicHasher<[F; M], [[u8; M]; N]>
    for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<[u8; M], [[u8; M]; N]>,
{
    fn hash_iter<I>(&self, input: I) -> [[u8; M]; N]
    where
        I: IntoIterator<Item = [F; M]>,
    {
        self.inner.hash_iter(F::into_parallel_byte_streams(input))
    }
}

impl<F, Inner, const N: usize, const M: usize> CryptographicHasher<[F; M], [[u32; M]; N]>
    for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<[u32; M], [[u32; M]; N]>,
{
    fn hash_iter<I>(&self, input: I) -> [[u32; M]; N]
    where
        I: IntoIterator<Item = [F; M]>,
    {
        self.inner.hash_iter(F::into_parallel_u32_streams(input))
    }
}

impl<F, Inner, const N: usize, const M: usize> CryptographicHasher<[F; M], [[u64; M]; N]>
    for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<[u64; M], [[u64; M]; N]>,
{
    fn hash_iter<I>(&self, input: I) -> [[u64; M]; N]
    where
        I: IntoIterator<Item = [F; M]>,
    {
        self.inner.hash_iter(F::into_parallel_u64_streams(input))
    }
}

#[cfg(test)]
mod tests {
    use core::array;

    use p3_koala_bear::KoalaBear;

    use crate::{CryptographicHasher, SerializingHasher};

    #[derive(Clone)]
    struct MockHasher;

    impl CryptographicHasher<u8, [u8; 4]> for MockHasher {
        fn hash_iter<I: IntoIterator<Item = u8>>(&self, iter: I) -> [u8; 4] {
            let sum: u8 = iter.into_iter().fold(0, |acc, x| acc.wrapping_add(x));
            // Simplest impl: set every element to the sum
            [sum; 4]
        }
    }

    impl CryptographicHasher<[u8; 4], [[u8; 4]; 4]> for MockHasher {
        fn hash_iter<I: IntoIterator<Item = [u8; 4]>>(&self, iter: I) -> [[u8; 4]; 4] {
            let sum: [u8; 4] = iter.into_iter().fold([0, 0, 0, 0], |acc, x| {
                [
                    acc[0].wrapping_add(x[0]),
                    acc[1].wrapping_add(x[1]),
                    acc[2].wrapping_add(x[2]),
                    acc[3].wrapping_add(x[3]),
                ]
            });
            // Simplest impl: set every element to the sum
            [sum; 4]
        }
    }

    impl CryptographicHasher<u32, [u32; 4]> for MockHasher {
        fn hash_iter<I: IntoIterator<Item = u32>>(&self, iter: I) -> [u32; 4] {
            let sum: u32 = iter.into_iter().fold(0, |acc, x| acc.wrapping_add(x));
            // Simplest impl: set every element to the sum
            [sum; 4]
        }
    }

    impl CryptographicHasher<[u32; 4], [[u32; 4]; 4]> for MockHasher {
        fn hash_iter<I: IntoIterator<Item = [u32; 4]>>(&self, iter: I) -> [[u32; 4]; 4] {
            let sum: [u32; 4] = iter.into_iter().fold([0, 0, 0, 0], |acc, x| {
                [
                    acc[0].wrapping_add(x[0]),
                    acc[1].wrapping_add(x[1]),
                    acc[2].wrapping_add(x[2]),
                    acc[3].wrapping_add(x[3]),
                ]
            });
            // Simplest impl: set every element to the sum
            [sum; 4]
        }
    }

    impl CryptographicHasher<u64, [u64; 4]> for MockHasher {
        fn hash_iter<I: IntoIterator<Item = u64>>(&self, iter: I) -> [u64; 4] {
            let sum: u64 = iter.into_iter().fold(0, |acc, x| acc.wrapping_add(x));
            // Simplest impl: set every element to the sum
            [sum; 4]
        }
    }

    impl CryptographicHasher<[u64; 4], [[u64; 4]; 4]> for MockHasher {
        fn hash_iter<I: IntoIterator<Item = [u64; 4]>>(&self, iter: I) -> [[u64; 4]; 4] {
            let sum: [u64; 4] = iter.into_iter().fold([0, 0, 0, 0], |acc, x| {
                [
                    acc[0].wrapping_add(x[0]),
                    acc[1].wrapping_add(x[1]),
                    acc[2].wrapping_add(x[2]),
                    acc[3].wrapping_add(x[3]),
                ]
            });
            // Simplest impl: set every element to the sum
            [sum; 4]
        }
    }

    #[test]
    fn test_parallel_hashers() {
        let mock_hash = MockHasher {};
        let hasher = SerializingHasher::new(mock_hash);
        let input: [KoalaBear; 256] = KoalaBear::new_array(array::from_fn(|x| x as u32));

        let parallel_input: [[KoalaBear; 4]; 64] = unsafe { core::mem::transmute(input) };
        let unzipped_input: [[KoalaBear; 64]; 4] = array::from_fn(|i| parallel_input.map(|x| x[i]));

        let u8_output_parallel: [[u8; 4]; 4] = hasher.hash_iter(parallel_input);
        let u8_output_individual: [[u8; 4]; 4] = unzipped_input.map(|x| hasher.hash_iter(x));
        let u8_output_individual_transposed =
            array::from_fn(|i| u8_output_individual.map(|x| x[i]));

        let u32_output_parallel: [[u32; 4]; 4] = hasher.hash_iter(parallel_input);
        let u32_output_individual: [[u32; 4]; 4] = unzipped_input.map(|x| hasher.hash_iter(x));
        let u32_output_individual_transposed =
            array::from_fn(|i| u32_output_individual.map(|x| x[i]));

        let u64_output_parallel: [[u64; 4]; 4] = hasher.hash_iter(parallel_input);
        let u64_output_individual: [[u64; 4]; 4] = unzipped_input.map(|x| hasher.hash_iter(x));
        let u64_output_individual_transposed =
            array::from_fn(|i| u64_output_individual.map(|x| x[i]));

        assert_eq!(u8_output_parallel, u8_output_individual_transposed);
        assert_eq!(u32_output_parallel, u32_output_individual_transposed);
        assert_eq!(u64_output_parallel, u64_output_individual_transposed);
    }
}
