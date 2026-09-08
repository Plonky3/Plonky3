use alloc::vec::Vec;

use p3_field::Field;

use crate::CryptographicHasher;

/// Byte budget for the buffer holding the serialized bytes of one group of rows.
///
/// The batched inner hash needs its messages back to back in memory.
/// Field elements reach byte form only through a serializing iterator.
///
/// 8 KiB stays inside a typical 32 KiB L1 data cache.
/// It still holds eight messages of 1 KiB, enough to fill a vector sponge's lanes.
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
    /// Forwarded from the inner byte hasher, the one digest shape here that batches.
    ///
    /// The `[u32; N]` and `[u64; N]` shapes wrap word sponges that hash a single message per call.
    /// The packed `[[uX; M]; N]` shapes already drive a vector permutation, one lane per packed slot.
    ///
    /// Both keep the trait default of one lane, which routes a Merkle tree through its packed arm.
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

        // A row too wide for the byte budget gains nothing from grouping.
        // Hash it straight from the serializing iterator, with no buffer at all.
        if row_bytes > ROW_BYTES_SCRATCH {
            for (digest, row) in out.iter_mut().zip(input.chunks_exact(row_len)) {
                *digest = self.hash_iter(row.iter().copied());
            }
            return;
        }

        // Serialize as many whole rows at a time as the budget holds.
        // The inner hasher then sees a long run of equal-length byte messages:
        //
        //     rows:    [ r_0 | r_1 | ... ]              row_len field elements each
        //     scratch: [ b_0 | b_1 | ... ]              row_bytes bytes each
        //
        // Rounding the group down to whole lane groups keeps every permutation of every call
        // fully occupied.
        //
        // Invariant: 1 <= rows_per_group <= ROW_BYTES_SCRATCH / row_bytes
        //     the wide-row case returned above, so at least one row fits the budget
        //     a budget too small for one lane group keeps every row that does fit
        let lanes = Inner::LANES.max(1);
        let rows_fitting = ROW_BYTES_SCRATCH / row_bytes;
        let rows_per_group = match rows_fitting / lanes {
            0 => rows_fitting,
            groups => groups * lanes,
        };

        // One buffer per call, refilled group by group, holding the largest group exactly.
        // Every byte of a group is written before it is read, so none of it is zeroed first.
        let mut scratch: Vec<u8> = Vec::with_capacity(rows_per_group.min(out.len()) * row_bytes);

        for (rows, digests) in input
            .chunks(rows_per_group * row_len)
            .zip(out.chunks_mut(rows_per_group))
        {
            // Serialize row by row so each message starts on its own row boundary.
            // Boundaries come from the declared serialized width of one field element.
            //
            // A stream of any other length shifts every later message.
            // Checking the running length after each row catches both directions.
            scratch.clear();
            for (index, row) in rows.chunks(row_len).enumerate() {
                scratch.extend(F::into_byte_stream(row.iter().copied()));
                assert_eq!(
                    scratch.len(),
                    (index + 1) * row_bytes,
                    "field serialization must be its declared width of {} bytes per element",
                    F::NUM_BYTES
                );
            }

            self.inner.hash_many(&scratch, digests);
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
    // Flattened iterators inhibit vectorization when pairing 32-bit elements into u64s.
    // Staging does not help the direct u64 mapping of 64-bit fields.
    const PREFER_CONTIGUOUS_INPUT: bool = M > 1 && F::NUM_BYTES == 4;

    fn hash_iter<I>(&self, input: I) -> [[u64; M]; N]
    where
        I: IntoIterator<Item = [F; M]>,
    {
        self.inner.hash_iter(F::into_parallel_u64_streams(input))
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::array;
    use core::cell::RefCell;

    use p3_field::PrimeCharacteristicRing;
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

    /// A byte hasher that records the digest count of every batched call it is handed.
    ///
    /// Three lanes is deliberately neither a power of two nor a divisor of the byte budget, so a
    /// group that is not lane aligned shows up as a count that three does not divide.
    #[derive(Clone)]
    struct LaneRecorder {
        calls: Rc<RefCell<Vec<usize>>>,
    }

    impl LaneRecorder {
        fn new() -> (Self, Rc<RefCell<Vec<usize>>>) {
            let calls = Rc::new(RefCell::new(Vec::new()));
            (
                Self {
                    calls: Rc::clone(&calls),
                },
                calls,
            )
        }
    }

    impl CryptographicHasher<u8, [u8; 4]> for LaneRecorder {
        const LANES: usize = 3;

        fn hash_iter<I: IntoIterator<Item = u8>>(&self, iter: I) -> [u8; 4] {
            MockHasher.hash_iter(iter)
        }

        fn hash_many(&self, input: &[u8], out: &mut [[u8; 4]]) {
            if out.is_empty() {
                return;
            }
            self.calls.borrow_mut().push(out.len());

            let len = input.len() / out.len();
            for (digest, message) in out.iter_mut().zip(input.chunks_exact(len)) {
                *digest = self.hash_iter(message.iter().copied());
            }
        }
    }

    /// A run of `rows * row_len` distinct field elements, laid out row by row.
    fn rows_of(rows: usize, row_len: usize) -> Vec<KoalaBear> {
        (0..rows * row_len)
            .map(|i| KoalaBear::from_u32(i as u32))
            .collect()
    }

    /// Digests of the same rows taken one at a time through the unbatched path.
    fn digests_one_by_one<H>(
        hasher: &SerializingHasher<H>,
        input: &[KoalaBear],
        row_len: usize,
    ) -> Vec<[u8; 4]>
    where
        H: CryptographicHasher<u8, [u8; 4]>,
    {
        input
            .chunks(row_len)
            .map(|row| hasher.hash_iter(row.iter().copied()))
            .collect()
    }

    #[test]
    fn hash_many_matches_the_unbatched_digests() {
        let hasher = SerializingHasher::new(MockHasher);

        // Row widths around the group boundaries: one element, a lane group, the widest row the
        // 8 KiB budget still groups, and one element past it.
        for row_len in [1, 3, 7, 100, 2048, 2049, 2100] {
            for rows in [1, 2, 3, 4, 7] {
                let input = rows_of(rows, row_len);
                let mut batched = vec![[0u8; 4]; rows];
                hasher.hash_many(&input, &mut batched);

                assert_eq!(
                    batched,
                    digests_one_by_one(&hasher, &input, row_len),
                    "rows {rows} of width {row_len}"
                );
            }
        }
    }

    #[test]
    fn hash_many_hands_the_inner_hasher_whole_lane_groups() {
        let (inner, calls) = LaneRecorder::new();
        let hasher = SerializingHasher::new(inner);

        // 400 bytes a row lets 20 rows fit the 8 KiB budget, which rounds down to six lane
        // groups of three.
        let input = rows_of(40, 100);
        let mut out = vec![[0u8; 4]; 40];
        hasher.hash_many(&input, &mut out);

        // Only the final call is short, and it is short because the input ran out.
        assert_eq!(*calls.borrow(), vec![18, 18, 4]);
        assert_eq!(out, digests_one_by_one(&hasher, &input, 100));
    }

    #[test]
    fn hash_many_keeps_single_rows_when_a_lane_group_will_not_fit() {
        let (inner, calls) = LaneRecorder::new();
        let hasher = SerializingHasher::new(inner);

        // 2800 bytes a row leaves room for two rows, fewer than the three a lane group needs, so
        // the group falls back to as many rows as do fit rather than overrunning the budget.
        let input = rows_of(3, 700);
        let mut out = vec![[0u8; 4]; 3];
        hasher.hash_many(&input, &mut out);

        assert_eq!(*calls.borrow(), vec![2, 1]);
        assert_eq!(out, digests_one_by_one(&hasher, &input, 700));
    }

    #[test]
    fn hash_many_hashes_a_row_wider_than_the_budget_on_its_own() {
        let (inner, calls) = LaneRecorder::new();
        let hasher = SerializingHasher::new(inner);

        // 2049 elements are 8196 bytes, past the 8 KiB budget, so grouping is skipped entirely
        // and every row is serialized straight into the sponge.
        let input = rows_of(4, 2049);
        let mut out = vec![[0u8; 4]; 4];
        hasher.hash_many(&input, &mut out);

        assert!(calls.borrow().is_empty());
        assert_eq!(out, digests_one_by_one(&hasher, &input, 2049));
    }

    #[test]
    fn hash_many_of_width_zero_rows_hashes_the_empty_message() {
        let hasher = SerializingHasher::new(MockHasher);

        // No input at all with digests requested means every row is empty.
        let mut out = vec![[1u8; 4]; 3];
        hasher.hash_many(&[] as &[KoalaBear], &mut out);

        let empty: [u8; 4] = hasher.hash_iter(core::iter::empty::<KoalaBear>());
        assert_eq!(out, vec![empty; 3]);
    }

    #[test]
    fn hash_many_reads_nothing_when_no_digests_are_requested() {
        let hasher = SerializingHasher::new(MockHasher);

        // A row length cannot be derived from zero digests, so the input is left untouched
        // instead of dividing by zero or tripping the multiple check.
        let mut out: [[u8; 4]; 0] = [];
        hasher.hash_many(&rows_of(3, 5), &mut out);
    }

    #[test]
    #[should_panic(expected = "must be a whole multiple")]
    fn hash_many_rejects_ragged_input() {
        let hasher = SerializingHasher::new(MockHasher);

        // Five elements cannot split into two equal rows.
        let mut out = [[0u8; 4]; 2];
        hasher.hash_many(&rows_of(5, 1), &mut out);
    }
}
