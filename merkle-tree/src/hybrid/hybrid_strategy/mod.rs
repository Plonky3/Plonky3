/// This module contains the logic for hybrid compression strategies, which
/// expose a common input type but can internally use different compressors
/// based on a size list and the current size (meant to represent the current
/// layer of a Merkle tree under construction).
use core::marker::PhantomData;

use p3_symmetric::PseudoCompressionFunction;

pub(crate) mod node_converter;
pub(crate) mod unsafe_node_converter;

/// A hybrid analogue of [`PseudoCompressionFunction`] for use in MMCS. It
/// exposes a single input type, regardless of the internal compressors it uses
/// and their particular input types. The only difference with
/// [`PseudoCompressionFunction`] is the addition of the `sizes` and
/// `current_size` arguments to the `compress` method, which give the
/// implementor information to decide which compressor to use. These are meant
/// to represent the numbers of rows of the matrices being committed to and the
/// number of nodes in the current tree level, respectively.
pub trait HybridPseudoCompressionFunction<T, const N: usize>: Clone {
    fn compress(&self, input: [T; N], sizes: &[usize], current_size: usize) -> T;
}

/// A converter between two types of nodes (for instance, `[BabyBear; 8]` and
/// `[u8; 32]` to be used in hybrid compressors using exactly two compressors
/// (for instance, [`Poseidon2`] and [`Blake3`]).
trait NodeConverter<N1, N2> {
    fn to_n2(n1: N1) -> N2;
    fn to_n1(n2: N2) -> N1;
}

/// A simple hybrid compressor using exactly two compressors: one to compress
/// the bottom-layer digests (and inject the next-to-bottom digests, if any) and
/// one to perform compression (and injection) at all other levels.
//
// Design consideration: Due to the need for `HybridPseudoCompressionFunction`
// to always receive the same type of input, which is chosen as `C1`'s input
// type in the `impl` below, whenever `C2` is used, one is forced to convert the
// input to `C1`'s input type, apply `C2`, and then convert back to (half of)
// `C1`'s input type. Therefore, if there are matrices of half the number of
// rows as the biggest matrix (so that `C2` would be used both to compress and
// inject), the structure is forced to perform an unfortunate unnecessary
// conversion:
//
// - First, we compress th bottom layer leaves:
//                  convert                    C2                            convert
//    Leaf type T1 ---------> C2's input type ----> half of C2's input type ---------> Half of C1's input type
//
// - Now, when we want to inject next-to-bottom-layer digests (compress them
//   with the previously compressed nodes), we need to convert the latter back
//   to C2's input type, which means that the last conversion in the step above
//   was unnecessary.
//
// As a redeeming fact, conversion can be very fast (such as the
// [`UnsafeNodeConverter`] implemented here). An alternative design would be to
// replace unique input type C1 by a new node type which can itself convert into
// several types, which would only happen at runtime if the hybrid compressor
// needs a different type than it receives. An obstacle to this approach is
// that (unsafe) conversion could become much slower, since one could no longer
// hard-cast arrays as we currently do.
#[derive(Clone)]
pub struct SimpleHybridCompressor<
    // Compressor 1
    C1,
    // Compressor 2
    C2,
    // Node-element type for `C1`
    W1,
    // Node-element type for `C2`
    W2,
    // Number of elements of type `W1` that form a node
    const DIGEST_ELEMS_1: usize,
    // Number of elements of type `W2` that form a node
    const DIGEST_ELEMS_2: usize,
    // Node converter
    NC,
> where
    C1: Clone,
    C2: Clone,
    W1: Clone,
    W2: Clone,
    NC: Clone,
{
    // Compressor 1
    c1: C1,
    // Compressor 2
    c2: C2,
    // Whether to use C1 (or not, i. e. C2) to compress the bottom layer. Note
    // that this can't be achieved by simply swapping the generics C1 and C2 on
    // the caller side due to the trait bounds in `HybridMerkleTree`.
    bottom_c1: bool,
    // Removing the generics W1, W2 and NC from this struct forces one to put
    // them in the `HybridPseudoCompressionFunction`, which has ugly
    // implications for the generics of `HybridMerkleTree`.
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
        // If we are at the bottom layer (XOR) C1 is the designated compressor
        // for the bottom layer: use C2
        if (current_size == sizes[0]) ^ self.bottom_c1 {
            let [input_0, input_1] = input;
            let input_w2 = [NC::to_n2(input_0), NC::to_n2(input_1)];
            NC::to_n1(self.c2.compress(input_w2))
        } else {
            // Otherwise, use C1
            self.c1.compress(input)
        }
    }
}
