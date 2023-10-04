use core::marker::PhantomData;

use p3_field::PrimeField32;

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

/// Maps input field elements to their 4-byte little-endian encodings, and maps output of the form
/// `[u8; 32]` to `[F; 8]`.
#[derive(Copy, Clone)]
pub struct SerializingHasher32<F, Inner> {
    inner: Inner,
    _phantom_f: PhantomData<F>,
}

impl<F, Inner> SerializingHasher32<F, Inner> {
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            _phantom_f: PhantomData,
        }
    }
}

impl<F, Inner> CryptographicHasher<F, [F; 8]> for SerializingHasher32<F, Inner>
where
    F: PrimeField32,
    Inner: CryptographicHasher<u8, [u8; 32]>,
{
    fn hash_iter<I>(&self, input: I) -> [F; 8]
    where
        I: IntoIterator<Item = F>,
    {
        let inner_out = self.inner.hash_iter(
            input
                .into_iter()
                .flat_map(|x| x.as_canonical_u32().to_le_bytes()),
        );

        core::array::from_fn(|i| {
            let inner_out_chunk: [u8; 4] = inner_out[i * 4..(i + 1) * 4].try_into().unwrap();
            F::from_wrapped_u32(u32::from_le_bytes(inner_out_chunk))
        })
    }
}
