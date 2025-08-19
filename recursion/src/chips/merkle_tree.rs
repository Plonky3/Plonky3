use core::{
    borrow::{Borrow, BorrowMut},
    cmp::Reverse,
    mem::transmute,
};

use alloc::vec;
use alloc::vec::Vec;
use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{PackedValue, PrimeCharacteristicRing, PrimeField64};
use p3_matrix::{Dimensions, Matrix, dense::RowMajorMatrix};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use p3_util::indices_arr;

const MAX_TREE_HEIGHT: usize = 32;
// const DIGEST_ELEMS: usize = 32;
pub struct MerkleTreeAir<F, H, C, const DIGEST_ELEMS: usize> {
    pub m_t: MerkleTreeMmcs<F, F, H, C, DIGEST_ELEMS>,
}

impl<F, H, C, const DIGEST_ELEMS: usize> BaseAir<F> for MerkleTreeAir<F, H, C, DIGEST_ELEMS>
where
    F: PackedValue,
    H: CryptographicHasher<F::Value, [F::Value; DIGEST_ELEMS]>
        + CryptographicHasher<F, [F; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[F::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
        + Sync,
    F::Value: Eq,
{
    fn width(&self) -> usize {
        NUM_MERKLE_TREE_COLS
    }
}

impl<AB: AirBuilder, H, C, const DIGEST_ELEMS: usize> Air<AB>
    for MerkleTreeAir<AB::F, H, C, DIGEST_ELEMS>
where
    AB::F: PackedValue,
    H: CryptographicHasher<
            <AB::F as PackedValue>::Value,
            [<AB::F as PackedValue>::Value; DIGEST_ELEMS],
        > + CryptographicHasher<AB::F, [AB::F; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[<AB::F as PackedValue>::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[AB::F; DIGEST_ELEMS], 2>
        + Sync,
    <AB::F as PackedValue>::Value: Eq,
{
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("The matrix is empty?"),
            main.row_slice(1).expect("The matrix only has 1 row?"),
        );
        let local: &MerkleTreeCols<AB::Var, DIGEST_ELEMS> = (*local).borrow();
        let next: &MerkleTreeCols<AB::Var, DIGEST_ELEMS> = (*next).borrow();

        // Receive the index and the initial root.

        // Assert that the height encoding is boolean.
        for i in 0..local.height_encoding.len() {
            builder.assert_bool(local.height_encoding[i].clone());
        }

        // Assert that there is at most one height encoding index that is equal to 1.
        let mut is_real = AB::Expr::ZERO;
        for i in 0..MAX_TREE_HEIGHT {
            is_real += local.height_encoding[i].clone();
        }
        builder.assert_bool(is_real.clone());

        // If the current row is a padding row, the next row must also be a padding row.
        let mut next_is_real = AB::Expr::ZERO;
        for i in 0..MAX_TREE_HEIGHT {
            next_is_real += next.height_encoding[i].clone();
        }
        builder
            .when(AB::Expr::ONE - is_real)
            .assert_zero(next_is_real);

        // Assert that the index bits are boolean.
        for i in 0..local.index_bits.len() {
            builder.assert_bool(local.index_bits[i].clone());
        }

        // Within the same execution, index bits are unchanged.
        for i in 0..local.index_bits.len() {
            builder
                .when_transition()
                .when(AB::Expr::ONE - local.is_final.clone())
                .assert_zero(local.index_bits[i].clone() - next.index_bits[i].clone());
        }
        // Assert that the height encoding is updated correctly.
        for i in 0..local.height_encoding.len() {
            builder
                .when(next.is_extra.clone())
                .when_transition()
                .assert_zero(local.height_encoding[i].clone() - next.height_encoding[i].clone());
            builder
                .when_transition()
                .when(AB::Expr::ONE - next.is_extra.clone())
                .assert_zero(
                    local.height_encoding[i].clone()
                        - local.height_encoding[(i + 1) % MAX_TREE_HEIGHT].clone(),
                );
        }
        builder
            .when_first_row()
            .assert_zero(AB::Expr::ONE - local.height_encoding[0].clone());

        // Assert that we reach the maximal height.
        let mut sum = AB::Expr::ZERO;
        for i in 0..MAX_TREE_HEIGHT {
            sum += local.height_encoding[i].clone() * AB::Expr::from_usize(i);
        }
        builder
            .when(local.is_final.clone())
            .assert_zero(sum - local.length.clone());

        let mut cur_to_hash = vec![AB::Expr::ZERO; 2 * DIGEST_ELEMS];
        for i in 0..DIGEST_ELEMS {
            for j in 0..DIGEST_ELEMS {
                cur_to_hash[i] += local.height_encoding[j].clone()
                    * (local.index_bits[j].clone() * local.sibling[j].clone()
                        + (AB::Expr::ONE - local.index_bits[j].clone()) * local.state[j].clone());
                cur_to_hash[DIGEST_ELEMS + i] += local.index_bits[j].clone()
                    * (local.index_bits[j].clone() * local.sibling[j].clone()
                        + (AB::Expr::ONE - local.height_encoding[j].clone())
                            * local.state[j].clone());
            }
            let tmp = cur_to_hash[i].clone();
            cur_to_hash[i] += (AB::Expr::ONE - local.is_extra.clone()) * tmp
                + AB::Expr::ONE * local.state[i].clone();
            let tmp = cur_to_hash[DIGEST_ELEMS + i].clone();
            cur_to_hash[DIGEST_ELEMS + i] += (AB::Expr::ONE - local.is_extra.clone()) * tmp
                + AB::Expr::ONE * local.sibling[i].clone();
        }
        // We send `(cur_hash, next_state)` to the Hash table to check the output, with filter `is_final`.
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct MerkleTreeCols<T, const DIGEST_ELEMS: usize> {
    // Bits of the leaf index we are currently verifying.
    pub index_bits: [T; DIGEST_ELEMS],
    // Length of the current index.
    pub length: T,
    // One-hot encoding of the height within the Merkle tree.
    pub height_encoding: [T; MAX_TREE_HEIGHT],
    // Sibling we are currently processing.
    pub sibling: [T; DIGEST_ELEMS],
    // Current state of the hash, which we are updating.
    pub state: [T; DIGEST_ELEMS],
    // Whether this is the final step of the Merkle
    // tree verification for this index.
    pub is_final: T,
    // Whether there is an extra step for the current height (due to batching).
    pub is_extra: T,
}

pub const NUM_MERKLE_TREE_COLS: usize = size_of::<MerkleTreeCols<u8, 32>>();
const fn make_col_map() -> MerkleTreeCols<usize, 32> {
    unsafe { transmute(indices_arr::<NUM_MERKLE_TREE_COLS>()) }
}

impl<T, const DIGEST_ELEMS: usize> Borrow<MerkleTreeCols<T, DIGEST_ELEMS>> for [T] {
    fn borrow(&self) -> &MerkleTreeCols<T, DIGEST_ELEMS> {
        debug_assert_eq!(self.len(), NUM_MERKLE_TREE_COLS);
        let (prefix, shorts, suffix) =
            unsafe { self.align_to::<MerkleTreeCols<T, DIGEST_ELEMS>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T, const DIGEST_ELEMS: usize> BorrowMut<MerkleTreeCols<T, DIGEST_ELEMS>> for [T] {
    fn borrow_mut(&mut self) -> &mut MerkleTreeCols<T, DIGEST_ELEMS> {
        debug_assert_eq!(self.len(), NUM_MERKLE_TREE_COLS);
        let (prefix, shorts, suffix) =
            unsafe { self.align_to_mut::<MerkleTreeCols<T, DIGEST_ELEMS>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

impl<F, H, C, const DIGEST_ELEMS: usize> MerkleTreeAir<F, H, C, DIGEST_ELEMS>
where
    F: PackedValue + PrimeField64,
    H: CryptographicHasher<F::Value, [F::Value; DIGEST_ELEMS]>,
    C: PseudoCompressionFunction<[F::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
        + Sync,
{
    pub fn generate_trace_rows<MerkleTreeMmcs>(
        &self,
        inputs: &[([F; DIGEST_ELEMS], usize, Vec<[F; DIGEST_ELEMS]>)], // root, index, height, siblings
        dimensions: &[Vec<Dimensions>],
    ) -> RowMajorMatrix<F> {
        let max_num_rows = dimensions
            .iter()
            .map(|dims| dims.iter().map(|d| d.height + 1).sum::<usize>())
            .sum::<usize>()
            .next_power_of_two();

        let trace_length = max_num_rows * NUM_MERKLE_TREE_COLS;

        let mut trace = RowMajorMatrix::new(F::zero_vec(trace_length), NUM_MERKLE_TREE_COLS);

        let (prefix, rows, suffix) = unsafe {
            trace
                .values
                .align_to_mut::<MerkleTreeCols<F, DIGEST_ELEMS>>()
        };
        assert!(prefix.is_empty(), "Alignment should match");
        assert!(suffix.is_empty(), "Alignment should match");
        assert_eq!(rows.len(), max_num_rows);

        let mut row_counter = 0;
        for (input, dims) in inputs.iter().zip(dimensions) {
            let heights_tallest_first = dims
                .iter()
                .sorted_by_key(|d| Reverse(d.height))
                .collect::<Vec<_>>();
            let mut cur_height_padded = heights_tallest_first[0].height.next_power_of_two();
            let initial_root = input.0;
            let mut state = initial_root;

            let mut index = input.1;
            let mut index_bits = [F::ZERO; DIGEST_ELEMS];
            for i in 0..DIGEST_ELEMS {
                index_bits[i] = F::from_usize((index >> i) & 1);
            }

            for (i, &sibling) in input.2.iter().enumerate() {
                let row = &mut rows[row_counter];
                row.state = initial_root;
                row.length = F::from_usize(heights_tallest_first[0].height);
                row_counter += 1;
                let (left, right) = if index & 1 == 0 {
                    (state, sibling)
                } else {
                    (sibling, state)
                };

                // Combine the current node with the sibling node to get the parent node.
                state = self.m_t.compress.compress([left, right]);
                index >>= 1;
                cur_height_padded >>= 1;
                row_counter += 1;

                let next_height = heights_tallest_first[i]
                    .map(|dims| dims.height)
                    .filter(|h| h.next_power_of_two() == cur_height_padded);
            }
        }

        trace
    }
}
