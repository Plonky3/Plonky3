//! Equivalent matrix decomposition for efficient partial rounds.
//!
//! # Overview
//!
//! This module implements the sparse matrix optimization described in **Appendix B** of the Poseidon
//! paper (Grassi et al., USENIX Security 2021). It transforms the RP partial rounds
//! from their textbook form (dense MDS multiply per round, O(t^2) each) into an
//! equivalent form using sparse matrices (O(t) each).
//!
//! # Background: Textbook Partial Rounds
//!
//! In the unoptimized Poseidon, each partial round applies:
//!
//! ```text
//!   state <- M * SBox(state + rc)
//! ```
//!
//! where M is the dense txt MDS matrix, SBox applies x^D only to `state[0]`, and
//! rc is a full t-vector of round constants. The cost per round is dominated by the
//! dense matrix multiply: O(t^2).
//!
//! # The Sparse Factorization (Poseidon Paper, Appendix B, Eq. 5)
//!
//! The key insight is that M can be factored as:
//!
//! ```text
//!   M = M' * M''
//! ```
//!
//! where:
//!
//! ```text
//!   M' = ┌───┬───┐       M'' = ┌─────────┬─────┐
//!        │ 1 │ 0 │             │ M[0][0] │  v  │
//!        ├───┼───┤             ├─────────┼─────┤
//!        │ 0 │ M̂ │             │   ŵ     │  I  │
//!        └───┴───┘             └─────────┴─────┘
//! ```
//!
//! Here M̂ is the (t-1)x(t-1) submatrix `M[1..t, 1..t]`, v is the first row of M
//! (excluding `M[0][0]`), ŵ = M̂^{-1} * w where w is the first column of M
//! (excluding `M[0][0]`), and I is the (t-1)x(t-1) identity.
//!
//! Since the partial S-box only touches `state[0]`, the dense M' factor can be
//! "absorbed" into the next round's M, leaving only the sparse M'' to be applied
//! per round via an O(t) matrix-vector product.
//!
//! After iterating this factorization across all RP rounds, we obtain:
//! - One dense **transition matrix** m_i (applied once before the loop)
//! - RP **sparse matrices** S_r, each parameterized by vectors v_r and ŵ_r of length t-1
//!
//! # Round Constant Compression
//!
//! In parallel, the round constants are compressed via backward substitution.
//! Since M^{-1} is linear, we can "push" round constants backward through the
//! inverse matrix, accumulating them into the first partial round. After this
//! transformation:
//! - The first partial round uses a full t-vector of constants.
//! - Each subsequent partial round uses a single scalar constant (for `state[0]`).
//! - The last partial round has no additive constant at all.
//!
//! # Implementation Note
//!
//! This implementation follows the HorizenLabs reference
//! (`plain_implementations/src/poseidon/poseidon_params.rs`) which works on the
//! **transposed** MDS matrix internally and reverses the sparse matrix ordering
//! before returning.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::Field;

/// Dense NxN matrix multiplication: `C = A * B`.
fn matrix_mul<F: Field, const N: usize>(a: &[[F; N]; N], b: &[[F; N]; N]) -> [[F; N]; N] {
    let mut result = [[F::ZERO; N]; N];
    for i in 0..N {
        for j in 0..N {
            let mut sum = F::ZERO;
            for k in 0..N {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }
    result
}

/// Matrix-vector multiplication: `result = M * v`.
fn matrix_vec_mul<F: Field, const N: usize>(m: &[[F; N]; N], v: &[F; N]) -> [F; N] {
    core::array::from_fn(|i| F::dot_product(&m[i], v))
}

/// Matrix transpose: `result[i][j] = m[j][i]`.
fn matrix_transpose<F: Field, const N: usize>(m: &[[F; N]; N]) -> [[F; N]; N] {
    let mut result = [[F::ZERO; N]; N];
    for i in 0..N {
        for j in 0..N {
            result[i][j] = m[j][i];
        }
    }
    result
}

/// NxN matrix inverse via Gauss-Jordan elimination.
///
/// # Panics
///
/// Panics if the matrix is singular (i.e., not invertible over the field).
fn matrix_inverse<F: Field, const N: usize>(m: &[[F; N]; N]) -> [[F; N]; N] {
    // We work on [M | I] and reduce M to I, yielding [I | M^{-1}].
    let mut aug = vec![[F::ZERO; N]; N];
    let mut inv = [[F::ZERO; N]; N];

    // Initialize: aug = M, inv = I.
    for i in 0..N {
        aug[i] = m[i];
        inv[i][i] = F::ONE;
    }

    for col in 0..N {
        // Partial pivoting: find a row with a nonzero entry in this column.
        let pivot_row = (col..N)
            .find(|&r| aug[r][col] != F::ZERO)
            .expect("Matrix is singular");

        // Swap the pivot row into position.
        if pivot_row != col {
            aug.swap(col, pivot_row);
            inv.swap(col, pivot_row);
        }

        // Scale the pivot row so that aug[col][col] = 1.
        let pivot_inv = aug[col][col].inverse();
        for j in 0..N {
            aug[col][j] *= pivot_inv;
            inv[col][j] *= pivot_inv;
        }

        // Eliminate this column in all other rows.
        for i in 0..N {
            if i == col {
                continue;
            }
            let factor = aug[i][col];
            if factor == F::ZERO {
                continue;
            }
            // Snapshot the pivot row to avoid aliasing.
            let aug_col_row: [F; N] = aug[col];
            let inv_col_row: [F; N] = inv[col];
            for j in 0..N {
                aug[i][j] -= factor * aug_col_row[j];
                inv[i][j] -= factor * inv_col_row[j];
            }
        }
    }

    inv
}

/// Inverse of the (N-1)x(N-1) bottom-right submatrix: `m[1..N, 1..N]`.
///
/// This is M̂^{-1} from the sparse matrix factorization (Appendix B, Eq. 5 in the paper).
///
/// # Panics
///
/// Panics if the submatrix is singular. For an MDS matrix, every submatrix is
/// non-singular by definition, so this should never happen with valid parameters.
fn submatrix_inverse<F: Field, const N: usize>(m: &[[F; N]; N]) -> Vec<Vec<F>> {
    let n = N - 1;

    // Extract the (N-1)x(N-1) bottom-right submatrix.
    let mut sub = vec![vec![F::ZERO; n]; n];
    for i in 0..n {
        for j in 0..n {
            sub[i][j] = m[i + 1][j + 1];
        }
    }

    // Standard Gauss-Jordan on the submatrix.
    let mut inv = vec![vec![F::ZERO; n]; n];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = F::ONE;
    }

    for col in 0..n {
        let pivot_row = (col..n)
            .find(|&r| sub[r][col] != F::ZERO)
            .expect("Submatrix is singular");

        if pivot_row != col {
            sub.swap(col, pivot_row);
            inv.swap(col, pivot_row);
        }

        let pivot_inv = sub[col][col].inverse();
        for j in 0..n {
            sub[col][j] *= pivot_inv;
            inv[col][j] *= pivot_inv;
        }

        for i in 0..n {
            if i == col {
                continue;
            }
            let factor = sub[i][col];
            if factor == F::ZERO {
                continue;
            }
            let sub_col_row: Vec<F> = sub[col].clone();
            let inv_col_row: Vec<F> = inv[col].clone();
            for j in 0..n {
                sub[i][j] -= factor * sub_col_row[j];
                inv[i][j] -= factor * inv_col_row[j];
            }
        }
    }

    inv
}

/// Factor the dense MDS matrix into RP sparse matrices.
///
/// # Algorithm (following HorizenLabs)
///
/// The algorithm works on M^T (the transposed MDS matrix) and iterates RP times.
/// In each iteration, it extracts the sparse components (v, ŵ) from the current
/// accumulated matrix, then "peels off" one sparse factor by multiplying M^T back in.
///
/// After all RP iterations:
/// - The accumulated remainder becomes the dense transition matrix m_i.
/// - The sparse components (v_r, ŵ_r) are reversed to match forward application order.
///
/// # Returns
///
/// A tuple (m_i, v_collection, ŵ_collection) where:
/// - m_i is a dense WIDTHxWIDTH transition matrix, applied once before the partial round loop.
/// - `v_collection[r]` has WIDTH-1 elements: the first column of sparse factor S_r.
/// - `ŵ_collection[r]` has WIDTH-1 elements: the first row of sparse factor S_r.
#[allow(clippy::type_complexity)]
fn compute_equivalent_matrices<F: Field, const N: usize>(
    mds: &[[F; N]; N],
    rounds_p: usize,
) -> ([[F; N]; N], Vec<Vec<F>>, Vec<Vec<F>>) {
    let mut w_hat_collection = Vec::with_capacity(rounds_p);
    let mut v_collection = Vec::with_capacity(rounds_p);

    // Work on M^T (HorizenLabs convention).
    let mds_t = matrix_transpose(mds);
    let mut m_mul = mds_t;
    let mut m_i = [[F::ZERO; N]; N];

    for _ in 0..rounds_p {
        // Extract v = first row of m_mul (excluding [0,0]).
        // In the transposed domain, this corresponds to the first column of M''.
        let v: Vec<F> = (1..N).map(|j| m_mul[0][j]).collect();

        // Extract w = first column of m_mul (excluding [0,0]).
        let w: Vec<F> = (1..N).map(|i| m_mul[i][0]).collect();

        // Compute M̂^{-1} (inverse of the bottom-right submatrix).
        let m_hat_inv = submatrix_inverse::<F, N>(&m_mul);

        // Compute ŵ = M̂^{-1} * w (Eq. 5 in the paper).
        let w_hat: Vec<F> = (0..N - 1)
            .map(|i| {
                let mut sum = F::ZERO;
                for j in 0..N - 1 {
                    sum += m_hat_inv[i][j] * w[j];
                }
                sum
            })
            .collect();

        v_collection.push(v);
        w_hat_collection.push(w_hat);

        // Build m_i: identity-like matrix (zero out first row/column, set [0,0] = 1).
        // This is the M' factor that gets absorbed into the next iteration.
        m_i = m_mul;
        m_i[0][0] = F::ONE;
        for row in m_i.iter_mut().skip(1) {
            row[0] = F::ZERO;
        }
        for elem in m_i[0].iter_mut().skip(1) {
            *elem = F::ZERO;
        }

        // Accumulate: m_mul = M^T * m_i for the next iteration.
        m_mul = matrix_mul(&mds_t, &m_i);
    }

    // Transpose m_i back (HorizenLabs works in the transposed domain).
    let m_i_returned = matrix_transpose(&m_i);

    // Reverse the collections: HorizenLabs computes them in reverse order
    // (index RP-1 first, RP-2 second, ..., 0 last). After reversal, index 0
    // corresponds to the first partial round applied.
    v_collection.reverse();
    w_hat_collection.reverse();

    (m_i_returned, v_collection, w_hat_collection)
}

/// Compress round constants via backward substitution through M^{-1}.
///
/// # Algorithm
///
/// Starting from the last partial round's constants and working backward:
///
/// 1. Push the accumulated constant vector through M^{-1}.
/// 2. Extract the first element as the scalar constant for that round.
/// 3. Add the remaining elements to the previous round's constants.
///
/// After processing all rounds, the accumulated vector becomes the first partial
/// round's full WIDTH-vector of constants.
///
/// # Returns
///
/// A tuple of (full_vector, scalar_constants) where:
/// - The full vector has WIDTH elements, used for the first partial round.
/// - The scalar constants have RP-1 entries, one per remaining partial round.
fn equivalent_round_constants<F: Field, const N: usize>(
    partial_rc: &[[F; N]],
    mds_inv: &[[F; N]; N],
) -> ([F; N], Vec<F>) {
    let rounds_p = partial_rc.len();
    let mut opt_partial_rc = vec![F::ZERO; rounds_p];

    // Start with the last partial round's full constant vector.
    let mut tmp = partial_rc[rounds_p - 1];

    // Process rounds in reverse: from second-to-last down to first.
    for i in (0..rounds_p - 1).rev() {
        // Push the accumulated constants backward through M^{-1}.
        let inv_cip = matrix_vec_mul(mds_inv, &tmp);

        // The first element becomes the scalar constant for round i+1.
        opt_partial_rc[i + 1] = inv_cip[0];

        // Load round i's constants and add the remaining backward-substituted values.
        tmp = partial_rc[i];
        for j in 1..N {
            tmp[j] += inv_cip[j];
        }
    }

    // The accumulated vector is the first partial round's full constant vector.
    let first_round_constants = tmp;

    // Discard index 0 (round 0 uses the full vector, not a scalar).
    let opt_partial_rc = opt_partial_rc[1..].to_vec();

    (first_round_constants, opt_partial_rc)
}

/// Compute all optimized partial round constants from raw parameters.
///
/// Combines the round constant compression and sparse matrix factorization
/// into a single entry point, keeping the individual helpers private.
///
/// # Returns
///
/// A tuple of:
/// - The compressed first-round constant vector (WIDTH elements).
/// - The optimized scalar round constants (RP-1 entries).
/// - The dense transition matrix m_i.
/// - The per-round sparse v vectors.
/// - The per-round sparse ŵ vectors.
#[allow(clippy::type_complexity)]
pub(crate) fn compute_optimized_constants<F: Field, const N: usize>(
    mds: &[[F; N]; N],
    rounds_p: usize,
    partial_rc: &[[F; N]],
) -> ([F; N], Vec<F>, [[F; N]; N], Vec<Vec<F>>, Vec<Vec<F>>) {
    let mds_inv = matrix_inverse(mds);
    let (first_round_constants, opt_partial_rc) = equivalent_round_constants(partial_rc, &mds_inv);
    let (m_i, sparse_v, sparse_w_hat) = compute_equivalent_matrices(mds, rounds_p);
    (
        first_round_constants,
        opt_partial_rc,
        m_i,
        sparse_v,
        sparse_w_hat,
    )
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;

    /// Verify that the matrix inverse produces a correct inverse: M * M^{-1} = I.
    #[test]
    fn test_matrix_inverse_roundtrip() {
        let mut rng = SmallRng::seed_from_u64(42);
        let m: [[F; 4]; 4] = core::array::from_fn(|_| {
            core::array::from_fn(|_| {
                use rand::RngExt;
                rng.random()
            })
        });

        let m_inv = matrix_inverse(&m);
        let product = matrix_mul(&m, &m_inv);

        for (i, row) in product.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if i == j {
                    assert_eq!(val, F::ONE, "Diagonal [{i}][{j}] should be 1");
                } else {
                    assert_eq!(val, F::ZERO, "Off-diagonal [{i}][{j}] should be 0");
                }
            }
        }
    }

    /// Verify equivalence between textbook and optimized partial rounds on a 4x4 state.
    ///
    /// Textbook form: each round adds the full constant vector, applies the S-box to
    /// the first element, and multiplies by the dense MDS matrix.
    ///
    /// Optimized form: adds the full constant vector once, applies the dense transition
    /// matrix once, then loops over rounds applying the S-box to the first element,
    /// a scalar constant, and the sparse matrix multiply.
    #[test]
    fn test_partial_rounds_equivalence_4x4() {
        use p3_field::InjectiveMonomial;

        let mut rng = SmallRng::seed_from_u64(42);
        let mds: [[F; 4]; 4] = core::array::from_fn(|_| {
            core::array::from_fn(|_| {
                use rand::RngExt;
                rng.random()
            })
        });
        let rounds_p = 3;

        let partial_rc: Vec<[F; 4]> = (0..rounds_p)
            .map(|_| {
                core::array::from_fn(|_| {
                    use rand::RngExt;
                    rng.random()
                })
            })
            .collect();

        let mds_inv = matrix_inverse(&mds);

        let (first_rc, opt_rc) = equivalent_round_constants::<F, 4>(&partial_rc, &mds_inv);
        let (m_i, v_coll, w_hat_coll) = compute_equivalent_matrices::<F, 4>(&mds, rounds_p);

        let input: [F; 4] = core::array::from_fn(|i| F::from_u32(i as u32 + 1));

        // Textbook partial rounds: add full constant vector, S-box on first element,
        // then dense MDS multiply.
        let mut textbook_state = input;
        for rc in partial_rc.iter().take(rounds_p) {
            for (s, &c) in textbook_state.iter_mut().zip(rc.iter()) {
                *s += c;
            }
            textbook_state[0] = InjectiveMonomial::<7>::injective_exp_n(&textbook_state[0]);
            textbook_state = matrix_vec_mul(&mds, &textbook_state);
        }

        // Optimized partial rounds: add full constant vector once, apply dense
        // transition matrix once, then loop with S-box + scalar constant + sparse multiply.
        let mut opt_state = input;
        for i in 0..4 {
            opt_state[i] += first_rc[i];
        }
        opt_state = matrix_vec_mul(&m_i, &opt_state);
        for r in 0..rounds_p {
            opt_state[0] = InjectiveMonomial::<7>::injective_exp_n(&opt_state[0]);
            if r < rounds_p - 1 {
                opt_state[0] += opt_rc[r];
            }
            crate::internal::cheap_matmul(&mut opt_state, mds[0][0], &v_coll[r], &w_hat_coll[r]);
        }

        assert_eq!(
            textbook_state, opt_state,
            "Textbook and optimized partial rounds should match"
        );
    }
}
