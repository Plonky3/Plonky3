/// A generic trait for cryptographic hashers that consume an arbitrary sequence of input items
/// and produce a fixed-size output.
///
/// This trait abstracts over hash functions in a flexible way, supporting both field elements,
/// scalars, or any other data type that implements `Clone`.
pub trait CryptographicHasher<Item: Clone, Out>: Clone {
    /// Number of equal-length messages this implementation hashes most efficiently in one call.
    ///
    /// One means every message is hashed on its own, which is the behaviour of a plain scalar sponge.
    /// A vectorized implementation reports how many independent states its permutation advances at once.
    ///
    /// Callers read this only to decide whether grouping messages is worth the bookkeeping.
    const LANES: usize = 1;

    /// Hash an iterator of input items.
    /// # Arguments
    /// - `input`: An iterator over items to be hashed.
    ///
    /// # Returns
    /// A fixed-size digest of type `Out`.
    fn hash_iter<I>(&self, input: I) -> Out
    where
        I: IntoIterator<Item = Item>;

    /// Hash an iterator of slices, by flattening it into a single stream of items.
    ///
    /// # Arguments
    /// - `input`: An iterator over slices of items to hash.
    ///
    /// # Returns
    /// A fixed-size digest of type `Out`.
    fn hash_iter_slices<'a, I>(&self, input: I) -> Out
    where
        I: IntoIterator<Item = &'a [Item]>,
        Item: 'a,
    {
        self.hash_iter(input.into_iter().flatten().cloned())
    }

    /// Hash a single slice of items.
    ///
    /// # Arguments
    /// - `input`: A slice of items to hash.
    ///
    /// # Returns
    /// A fixed-size digest of type `Out`.
    fn hash_slice(&self, input: &[Item]) -> Out {
        self.hash_iter_slices(core::iter::once(input))
    }

    /// Hash a single item.
    ///
    /// # Arguments
    /// - `input`: A single item to hash.
    ///
    /// # Returns
    /// A fixed-size digest of type `Out`.
    fn hash_item(&self, input: Item) -> Out {
        self.hash_slice(&[input])
    }

    /// Hash a batch of equal-length messages, one digest per message.
    ///
    /// All messages sit back to back in a single slice, so the common message length is
    /// the input length divided by the number of requested digests:
    ///
    /// ```text
    ///     input: [ msg_0 | msg_1 | ... | msg_{m-1} ]   m * len items
    ///     out:   [ dig_0 | dig_1 | ... | dig_{m-1} ]   m digests
    /// ```
    ///
    /// The default hashes the messages one at a time.
    /// An override exists purely to exploit vector hardware and must return the very same digests.
    ///
    /// # Panics
    ///
    /// Panics if the input length is not a whole multiple of the number of requested digests.
    fn hash_many(&self, input: &[Item], out: &mut [Out]) {
        // No digests requested means there is nothing to read from the input.
        if out.is_empty() {
            return;
        }

        // Every message has the same length, so the split is exact by contract.
        assert!(
            input.len().is_multiple_of(out.len()),
            "input length ({}) must be a whole multiple of the digest count ({})",
            input.len(),
            out.len()
        );
        let len = input.len() / out.len();

        // Zero-length messages all hash to the same digest.
        // They are also the one case a chunked walk over the input cannot express.
        if len == 0 {
            for digest in out.iter_mut() {
                *digest = self.hash_slice(&[]);
            }
            return;
        }

        // Walk the messages in order so the digests land in the caller's order.
        for (digest, message) in out.iter_mut().zip(input.chunks_exact(len)) {
            *digest = self.hash_slice(message);
        }
    }
}
