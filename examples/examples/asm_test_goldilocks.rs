//! Temporary module for assembly inspection with Goldilocks field.
//!
//! On ARM (Apple Silicon) without AVX, Goldilocks::Packing = Goldilocks (WIDTH=1).
//! This lets us verify the scalar broadcast optimization works correctly for real field types.
//!
//! Run with:
//!   cargo asm -p p3-examples --example asm_test_goldilocks "test_" --native

use p3_field::integers::QuotientMap;
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

type G = Goldilocks;

// === Test matrix methods with Goldilocks ===

#[inline(never)]
#[unsafe(no_mangle)]
pub fn test_horizontally_packed_row_goldilocks(mat: &RowMajorMatrix<G>, row: usize) -> Vec<G> {
    let (packed, _suffix) = mat.horizontally_packed_row::<G>(row);
    packed.collect()
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn test_padded_horizontally_packed_row_goldilocks(
    mat: &RowMajorMatrix<G>,
    row: usize,
) -> Vec<G> {
    mat.padded_horizontally_packed_row::<G>(row).collect()
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn test_vertically_packed_row_goldilocks(mat: &RowMajorMatrix<G>, row: usize) -> Vec<G> {
    mat.vertically_packed_row::<G>(row).collect()
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn test_vertically_packed_row_pair_goldilocks(
    mat: &RowMajorMatrix<G>,
    row: usize,
    step: usize,
) -> Vec<G> {
    mat.vertically_packed_row_pair::<G>(row, step)
}

// === Test with array packing for comparison ===

#[inline(never)]
#[unsafe(no_mangle)]
pub fn test_horizontally_packed_row_goldilocks_array4(
    mat: &RowMajorMatrix<G>,
    row: usize,
) -> Vec<[G; 4]> {
    let (packed, _suffix) = mat.horizontally_packed_row::<[G; 4]>(row);
    packed.collect()
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn test_vertically_packed_row_goldilocks_array4(
    mat: &RowMajorMatrix<G>,
    row: usize,
) -> Vec<[G; 4]> {
    mat.vertically_packed_row::<[G; 4]>(row).collect()
}

fn main() {
    // Create a small test matrix
    let values: Vec<G> = (0..16).map(G::from_int).collect();
    let mat = RowMajorMatrix::new(values, 4);

    println!("Testing matrix methods...");
    let row = test_horizontally_packed_row_goldilocks(&mat, 0);
    println!("horizontally_packed_row: {:?}", row.len());

    let row = test_padded_horizontally_packed_row_goldilocks(&mat, 0);
    println!("padded_horizontally_packed_row: {:?}", row.len());

    let row = test_vertically_packed_row_goldilocks(&mat, 0);
    println!("vertically_packed_row: {:?}", row.len());

    let row = test_vertically_packed_row_pair_goldilocks(&mat, 0, 1);
    println!("vertically_packed_row_pair: {:?}", row.len());

    println!("All tests passed!");
}
