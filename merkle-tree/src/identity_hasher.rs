use alloc::vec::Vec;

use p3_symmetric::CryptographicHasher;

// pub struct IdentityHasher;

// // CryptographicHasher<BabyBear, [BabyBear; 8]>

// impl<Item, Out> CryptographicHasher<Item, Out> for IdentityHasher
// {
//     fn hash_iter<I>(&self, input: I) -> Out
//     where
//         I: IntoIterator<Item = Item> {
//             Out::try_from(input).expect("Failed to convert the input iterator into the output type")
//         }
// }

// Vec<BabyBear>: TryInto<[BabyBear; 8]>

#[derive(Clone)]
pub struct IdentityHasher<const N: usize>;

// CryptographicHasher<BabyBear, [BabyBear; 8]>

impl<Item, Out, const N: usize> CryptographicHasher<Item, Out> for IdentityHasher<N>
where
    Item: Clone + Default,
    Out: TryFrom<Vec<Item>>,
{
    fn hash_iter<I>(&self, input: I) -> Out
    where
        I: IntoIterator<Item = Item>,
    {
        // TODO try to avoid dummy cloning
        let mut input_vec = input.into_iter().collect::<Vec<Item>>();

        if input_vec.len() > N {
            panic!("Input length is greater than the maximum number of items");
        }

        // TODO is this a safe way to pad?
        input_vec.resize(N, Item::default());

        match input_vec.try_into() {
            Ok(v) => v,
            Err(_) => panic!("Failed to convert the input iterator into the output type"),
        }
    }
}
