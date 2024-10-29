use core::marker::PhantomData;

use p3_symmetric::PseudoCompressionFunction;

pub(crate) mod node_converter;
pub(crate) mod utils;

#[cfg(feature = "unsafe-conversion")]
pub(crate) mod unsafe_node_converter;

pub(crate) use node_converter::*;

use crate::pretty_hash_type;

// TODO add to doc: closely mimics CryptographicHasher but

// TODO decide if converting the input to a reference brings about performance
// improvements or at least doesn't incur overhead
trait NodeConverter<N1, N2> {
    fn to_n2(n1: N1) -> N2;

    fn to_n1(n2: N2) -> N1;
}

pub trait HybridPseudoCompressionFunction<T, const N: usize>: Clone {
    fn compress(&self, input: [T; N], sizes: &[usize], current_size: usize) -> T;
}

#[derive(Clone)]
pub struct SimpleHybridCompressor<
    C1,
    C2,
    W1,
    W2,
    const DIGEST_ELEMS_1: usize,
    const DIGEST_ELEMS_2: usize,
    NC,
> where
    C1: Clone,
    C2: Clone,
    W1: Clone,
    W2: Clone,
    NC: Clone,
{
    c1: C1,
    c2: C2,
    bottom_c1: bool,
    _marker: PhantomData<(W1, W2, NC)>,
}

impl<C1, C2, W1, W2, NC, const DIGEST_ELEMS_1: usize, const DIGEST_ELEMS_2: usize>
    SimpleHybridCompressor<C1, C2, W1, W2, DIGEST_ELEMS_1, DIGEST_ELEMS_2, NC>
where
    C1: Clone,
    C2: Clone,
    NC: Clone,
    W1: Clone,
    W2: Clone,
{
    pub fn new(c1: C1, c2: C2, bottom_c1: bool) -> Self {
        Self {
            c1,
            c2,
            bottom_c1,
            _marker: PhantomData,
        }
    }
}

impl<C1, C2, W1, W2, NC, const DIGEST_ELEMS_1: usize, const DIGEST_ELEMS_2: usize>
    HybridPseudoCompressionFunction<[W1; DIGEST_ELEMS_1], 2>
    for SimpleHybridCompressor<C1, C2, W1, W2, DIGEST_ELEMS_1, DIGEST_ELEMS_2, NC>
where
    C1: PseudoCompressionFunction<[W1; DIGEST_ELEMS_1], 2> + Clone,
    C2: PseudoCompressionFunction<[W2; DIGEST_ELEMS_2], 2> + Clone,
    W1: Clone,
    W2: Clone,
    NC: NodeConverter<[W1; DIGEST_ELEMS_1], [W2; DIGEST_ELEMS_2]> + Clone,
{
    fn compress(
        &self,
        input: [[W1; DIGEST_ELEMS_1]; 2],
        sizes: &[usize],
        current_size: usize,
    ) -> [W1; DIGEST_ELEMS_1] {
        if (current_size == sizes[0]) ^ self.bottom_c1 {
            // TODO potentially remove or handle differently to avoid overhead
            // log::debug!(
            //     " - compressing with H2 hash: {}, (sizes: {:?}, current_size: {})",
            //     &pretty_hash_type::<C2>(),
            //     sizes,
            //     current_size,
            // );

            let [input_0, input_1] = input;
            let input_w2 = [NC::to_n2(input_0), NC::to_n2(input_1)];
            NC::to_n1(self.c2.compress(input_w2))
        } else {
            // log::debug!(
            //     " - compressing with H1 hash: {}, (sizes: {:?}, current_size: {})",
            //     &pretty_hash_type::<C1>(),
            //     sizes,
            //     current_size,
            // );

            self.c1.compress(input)
        }
    }
}

// PackedNodeType for poseidon:
// - Morally: [[BabyBear; 8]; 4]
// - Really: [[BabyBear; 4]; 8]

// PackedNode for Blake3:
// - Morally: [[u8; 32]; 4]
// - Really: [[u8; 4]; 32]
