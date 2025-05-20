use core::ptr::{swap, swap_nonoverlapping};
#[cfg(feature = "parallel")]
use core::sync::atomic::{AtomicPtr, Ordering};

/// Log2 of the matrix dimension below which we use the base-case direct swap loop.
/// e.g. LB_BASE_CASE_LOG = 3 means base case is used for ≤ 8×8 submatrices
const LB_BASE_CASE_LOG: usize = 3;

/// Absolute size threshold (in elements) below which recursive swap stops
const BASE_CASE_ELEMENT_THRESHOLD: usize = 1 << LB_BASE_CASE_LOG;

#[cfg(feature = "parallel")]
/// Threshold (in number of elements) beyond which we enable parallel recursion
const PARALLEL_RECURSION_THRESHOLD: usize = 1 << 10;

/// Transpose a small square matrix in-place using element-wise swaps.
///
/// # Parameters
/// - `arr`: A mutable reference to a 1D array representing a larger row-major matrix.
/// - `lb_stride`: Log2 of the stride between rows in the array.
/// - `lb_size`: Log2 of the dimension of the square matrix to transpose.
/// - `x`: Offset (in rows and columns) from the top-left corner of the full array.
///
/// The matrix occupies a logical square region starting at `(x, x)` and of size `1 << lb_size`.
///
/// ## SAFETY
/// - All accesses to `arr` must be in-bounds.
/// - `lb_size <= lb_stride` must hold to prevent overlapping indices during swaps.
unsafe fn transpose_in_place_square_small<T>(
    arr: &mut [T],
    lb_stride: usize,
    lb_size: usize,
    x: usize,
) {
    unsafe {
        // Loop over upper triangle (excluding diagonal)
        for i in x + 1..x + (1 << lb_size) {
            for j in x..i {
                // Compute memory offsets and swap M[i, j] <-> M[j, i]
                swap(
                    arr.get_unchecked_mut(i + (j << lb_stride)),
                    arr.get_unchecked_mut((i << lb_stride) + j),
                );
            }
        }
    }
}

/// Recursively swaps two submatrices across the main diagonal as part of a larger transposition.
///
/// Given:
/// - Submatrix `A` of shape `(rows × cols)`
/// - Submatrix `B` of shape `(cols × rows)`
///
/// This function swaps element `A[i, j]` with `B[j, i]`, effectively transposing them
/// relative to each other.
///
/// `A` is assumed to be row-major, starting at pointer `a`, where A[i,j] = a[i * stride + j].
/// `B` is assumed to be row-major, starting at pointer `b`, where B[j,i] = b[j * stride + i].
///
/// The recursion always splits along the longer dimension to balance cache and workload.
///
/// # Safety
/// - `a` and `b` must be valid for `rows * cols` reads and writes.
/// - The regions pointed to by `a` and `b` must be disjoint.
/// - `stride` must be large enough to avoid overlapping accesses during index calculations.
pub(super) unsafe fn transpose_swap_square<T: Copy>(
    a: *mut T,
    b: *mut T,
    stride: usize,
    (rows, cols): (usize, usize),
) {
    let size = rows * cols;

    // Base case: directly swap A[i,j] with B[j,i] using pointer offsets
    if size < BASE_CASE_ELEMENT_THRESHOLD {
        for i in 0..rows {
            for j in 0..cols {
                let ai = i * stride + j;
                let bi = j * stride + i;
                unsafe {
                    swap_nonoverlapping(a.add(ai), b.add(bi), 1);
                }
            }
        }
        return;
    }

    #[cfg(feature = "parallel")]
    {
        // If large enough, split work recursively in parallel
        if size > PARALLEL_RECURSION_THRESHOLD {
            let a = AtomicPtr::new(a);
            let b = AtomicPtr::new(b);

            // Prefer splitting the longer dimension for better balance and locality
            if rows > cols {
                let top = rows / 2;
                let bottom = rows - top;
                rayon::join(
                    || {
                        let a = a.load(Ordering::Relaxed);
                        let b = b.load(Ordering::Relaxed);
                        unsafe {
                            transpose_swap_square(a, b, stride, (top, cols));
                        }
                    },
                    || {
                        let a = a.load(Ordering::Relaxed);
                        let b = b.load(Ordering::Relaxed);
                        unsafe {
                            transpose_swap_square(
                                a.add(top * stride),
                                b.add(top),
                                stride,
                                (bottom, cols),
                            );
                        }
                    },
                );
            } else {
                let left = cols / 2;
                let right = cols - left;
                rayon::join(
                    || {
                        let a = a.load(Ordering::Relaxed);
                        let b = b.load(Ordering::Relaxed);
                        unsafe {
                            transpose_swap_square(a, b, stride, (rows, left));
                        }
                    },
                    || {
                        let a = a.load(Ordering::Relaxed);
                        let b = b.load(Ordering::Relaxed);
                        unsafe {
                            transpose_swap_square(
                                a.add(left),
                                b.add(left * stride),
                                stride,
                                (rows, right),
                            );
                        }
                    },
                );
            }
            return;
        }
    }

    // Sequential case: same recursive logic without threading
    if rows > cols {
        let top = rows / 2;
        let bottom = rows - top;
        unsafe {
            transpose_swap_square(a, b, stride, (top, cols));
            transpose_swap_square(a.add(top * stride), b.add(top), stride, (bottom, cols));
        }
    } else {
        let left = cols / 2;
        let right = cols - left;
        unsafe {
            transpose_swap_square(a, b, stride, (rows, left));
            transpose_swap_square(a.add(left), b.add(left * stride), stride, (rows, right));
        }
    }
}

/// In-place recursive transposition of a square matrix of size `2^lb_size × 2^lb_size`,
/// embedded inside a larger row-major array at offset `(x, x)`.
///
/// Each matrix element `M[i,j]` is stored at:
/// ```text
/// \begin{equation}
///     \text{index}(i,j) = i + x + ((i + x) << lb_stride) + (j + x)
/// \end{equation}
/// ```
///
/// The matrix is recursively split into four quadrants:
/// ```text
/// +----+----+
/// | TL | TR |
/// +----+----+
/// | BL | BR |
/// +----+----+
/// ```
/// Transposition proceeds by:
/// 1. Recursively transposing `TL`
/// 2. Swapping `TR` and `BL` across the diagonal
/// 3. Recursively transposing `BR`
///
/// # Safety
/// - Assumes all accesses via `((i + x) << lb_stride) + (j + x)` are in-bounds.
/// - Requires `lb_size <= lb_stride` to avoid index overlap.
pub(crate) unsafe fn transpose_in_place_square<T>(
    arr: &mut [T],
    lb_stride: usize,
    lb_size: usize,
    x: usize,
) where
    T: Copy + Send + Sync,
{
    // If small, switch to base case
    if lb_size <= LB_BASE_CASE_LOG {
        unsafe {
            transpose_in_place_square_small(arr, lb_stride, lb_size, x);
        }
        return;
    }

    #[cfg(feature = "parallel")]
    {
        // Log2 of half the matrix dimension
        let lb_half = lb_size - 1;
        // Half the matrix size (e.g. 8 for 16×16)
        let half = 1 << lb_half;
        // Total number of elements in the full square matrix
        let elements = 1 << (2 * lb_size);

        if elements >= PARALLEL_RECURSION_THRESHOLD {
            // Shared base pointer for parallel recursion
            let base = AtomicPtr::new(arr.as_mut_ptr());
            // Total length of the backing array
            let len = arr.len();
            // Row stride in physical memory
            let stride = 1 << lb_stride;
            // Size of each quadrant (half x half)
            let dim = 1 << lb_half;

            // Coordinate each quadrant via `rayon::join`:
            // - TL and BR are recursive calls
            // - TR and BL are swapped directly
            rayon::join(
                || unsafe {
                    transpose_in_place_square(
                        core::slice::from_raw_parts_mut(base.load(Ordering::Relaxed), len),
                        lb_stride,
                        lb_half,
                        x,
                    )
                },
                || {
                    rayon::join(
                        // TR: starts at (x, x + half)
                        // BL: starts at (x + half, x)
                        || unsafe {
                            let ptr = base.load(Ordering::Relaxed);
                            transpose_swap_square(
                                ptr.add((x << lb_stride) + (x + half)),
                                ptr.add(((x + half) << lb_stride) + x),
                                stride,
                                (dim, dim),
                            );
                        },
                        || unsafe {
                            transpose_in_place_square(
                                core::slice::from_raw_parts_mut(base.load(Ordering::Relaxed), len),
                                lb_stride,
                                lb_half,
                                x + half,
                            )
                        },
                    )
                },
            );
            return;
        }
    }

    // Sequential version of above logic
    // Log2 of the new quadrant size (we're splitting the matrix in half)
    let lb_block_size = lb_size - 1;
    // Actual size of each quadrant (i.e., half the current matrix size)
    let block_size = 1 << lb_block_size;
    // Physical stride between rows in memory (in elements)
    let stride = 1 << lb_stride;
    // The size of each submatrix (used as a dimension for swapping TR/BL)
    let dim = block_size;
    // Raw pointer to the base of the array for manual offset calculations
    let ptr = arr.as_mut_ptr();

    unsafe {
        // Transpose TL quadrant (top-left)
        transpose_in_place_square(arr, lb_stride, lb_block_size, x);
        // Swap TR (top-right) with BL (bottom-left)
        transpose_swap_square(
            ptr.add((x << lb_stride) + (x + block_size)),
            ptr.add(((x + block_size) << lb_stride) + x),
            stride,
            (dim, dim),
        );
        // Transpose BR quadrant (bottom-right)
        transpose_in_place_square(arr, lb_stride, lb_block_size, x + block_size);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    /// Helper to create a square matrix of size `2^log_size` with elements `0..n^2`
    fn generate_matrix(log_size: usize) -> Vec<u32> {
        let size = 1 << log_size;
        (0..size * size).collect()
    }

    /// Reference transpose that returns a new vector (row-major layout)
    fn transpose_reference(input: &[u32], log_size: usize) -> Vec<u32> {
        let size = 1 << log_size;
        let mut transposed = vec![0; size * size];
        for i in 0..size {
            for j in 0..size {
                transposed[j * size + i] = input[i * size + j];
            }
        }
        transposed
    }

    #[test]
    fn transpose_square() {
        // Loop over matrix sizes:
        // Each size is of the form 2^log_size × 2^log_size
        for log_size in 1..=10 {
            // Compute the actual dimension: size = 2^log_size
            let size = 1 << log_size;

            // Generate a flat matrix of size×size elements
            let mut mat = generate_matrix(log_size);

            // Compute the reference result using a naive transpose implementation
            let expected = transpose_reference(&mat, log_size);

            // Perform the in-place transpose on `mat`.
            unsafe {
                transpose_in_place_square(&mut mat, log_size, log_size, 0);
            }

            // Compare the transposed matrix against the reference.
            assert_eq!(mat, expected, "Transpose failed for {size}x{size} matrix");
        }
    }
}
