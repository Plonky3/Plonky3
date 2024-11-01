use alloc::vec::Vec;

pub trait CryptographicHasher<Item: Clone, Out>: Clone {
    fn hash_iter<I>(&self, input: I) -> Out
    where
        I: IntoIterator<Item = Item>;

    fn hash_iter_slices<'a, I>(&self, input: I) -> Out
    where
        I: IntoIterator<Item = &'a [Item]>,
        Item: 'a,
    {
        self.hash_iter(input.into_iter().flatten().cloned())
    }

    fn hash_slice(&self, input: &[Item]) -> Out {
        self.hash_iter_slices(core::iter::once(input))
    }

    fn hash_item(&self, input: Item) -> Out {
        self.hash_slice(&[input])
    }
}

/// Transparent structure implementing `CryptographicHasher` by padding the
/// received elements to a predetermined length (with `Default`), panicking if
/// the input exceeds it.
///
/// Caution is advised when using this as a hasher, since it does not satisfy
/// most security properties required of a cryptographic hash function. For
/// instance, to avoid second-preimage attacks, all parties should know
/// beforehand the number of elements that will be fed as input before padding
/// (otherwise, an attacker could simply add Defaults at the end of
/// non-maximally-sized inputs).
#[derive(Clone)]
pub struct IdentityHasher<const N: usize>;

impl<Item, Out, const N: usize> CryptographicHasher<Item, Out> for IdentityHasher<N>
where
    Item: Clone + Default,
    Out: TryFrom<Vec<Item>>,
{
    fn hash_iter<I>(&self, input: I) -> Out
    where
        I: IntoIterator<Item = Item>,
    {
        let mut input_vec = input.into_iter().collect::<Vec<Item>>();

        if input_vec.len() > N {
            panic!("Input length is greater than the maximum number of items");
        }

        input_vec.resize(N, Item::default());

        match input_vec.try_into() {
            Ok(v) => v,
            Err(_) => panic!("Failed to convert the input iterator into the output type"),
        }
    }
}
