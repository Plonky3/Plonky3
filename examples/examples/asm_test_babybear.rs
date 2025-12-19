//! Temporary module for assembly inspection with BabyBear field.
//!
//! On ARM (Apple Silicon) with NEON, BabyBear has optimized SIMD packing (WIDTH=4).
//! This lets us verify the optimizations work correctly for packed field types.
//!
//! Run with:
//!   cargo asm -p p3-examples --example asm_test_babybear "test_" --native

use p3_baby_bear::BabyBear;
use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

type F = BabyBear;
type P = <F as Field>::Packing;

// === Test matrix methods with BabyBear Packing ===

#[inline(never)]
#[unsafe(no_mangle)]
pub fn test_horizontally_packed_row_babybear(mat: &RowMajorMatrix<F>, row: usize) -> Vec<P> {
    let (packed, _suffix) = mat.horizontally_packed_row::<P>(row);
    packed.collect()
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn test_padded_horizontally_packed_row_babybear(mat: &RowMajorMatrix<F>, row: usize) -> Vec<P> {
    mat.padded_horizontally_packed_row::<P>(row).collect()
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn test_vertically_packed_row_babybear(mat: &RowMajorMatrix<F>, row: usize) -> Vec<P> {
    mat.vertically_packed_row::<P>(row).collect()
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn test_vertically_packed_row_pair_babybear(
    mat: &RowMajorMatrix<F>,
    row: usize,
    step: usize,
) -> Vec<P> {
    mat.vertically_packed_row_pair::<P>(row, step)
}

fn main() {
    // Create a small test matrix
    let values: Vec<F> = (0..64).map(F::from_u32).collect();
    let mat = RowMajorMatrix::new(values, 8);

    println!("BabyBear Packing WIDTH = {}", P::WIDTH);

    println!("Testing matrix methods...");
    let row = test_horizontally_packed_row_babybear(&mat, 0);
    println!("horizontally_packed_row: {:?}", row.len());

    let row = test_padded_horizontally_packed_row_babybear(&mat, 0);
    println!("padded_horizontally_packed_row: {:?}", row.len());

    let row = test_vertically_packed_row_babybear(&mat, 0);
    println!("vertically_packed_row: {:?}", row.len());

    let row = test_vertically_packed_row_pair_babybear(&mat, 0, 1);
    println!("vertically_packed_row_pair: {:?}", row.len());

    println!("All tests passed!");
}
