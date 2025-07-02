/// A generic trait for cryptographic hashers that consume an arbitrary sequence of input items
/// and produce a fixed-size output.
///
/// This trait abstracts over hash functions in a flexible way, supporting both field elements,
/// scalars, or any other data type that implements `Clone`.
pub trait CryptographicHasher<Item: Clone, Out>: Clone {
    /// Hash an iterator of input items into a single digest output.    ///
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
    /// A digest representing the hash of all concatenated items.
    fn hash_iter_slices<'a, I>(&self, input: I) -> Out
    where
        I: IntoIterator<Item = &'a [Item]>,
        Item: 'a,
    {
        self.hash_iter(input.into_iter().flatten().cloned())
    }

    /// Hash a single contiguous slice of items.
    ///
    /// # Arguments
    /// - `input`: A slice of items to hash.
    ///
    /// # Returns
    /// A digest of the input slice.
    fn hash_slice(&self, input: &[Item]) -> Out {
        self.hash_iter_slices(core::iter::once(input))
    }

    /// Hash a single item.
    ///
    /// # Arguments
    /// - `input`: A single item to hash.
    ///
    /// # Returns
    /// A digest of the single item.
    fn hash_item(&self, input: Item) -> Out {
        self.hash_slice(&[input])
    }
}
