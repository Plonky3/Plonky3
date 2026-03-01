//! Partial (internal) round layers for the Poseidon permutation.
//!
//! # Overview
//!
//! Partial rounds apply the S-box only to `state[0]`, leaving the other t-1 elements
//! unchanged through the nonlinear layer. This is the key efficiency insight of the Poseidon
//! design: partial rounds are much cheaper than full rounds, yet they efficiently
//! increase the algebraic degree to resist interpolation and Gröbner basis attacks.
//!
//! # Sparse Matrix Optimization
//!
//! In the textbook formulation, each partial round still multiplies by the full dense
//! MDS matrix M (cost O(t^2)). The Poseidon paper (Appendix B) factors M into a product
//! of sparse matrices, one per round. Each sparse matrix S_r has the structure:
//!
//! ```text
//!   S_r = ┌─────────┬───────────┐
//!         │ mds_0_0 │  ŵ_r      │    <- first row
//!         ├─────────┼───────────┤
//!         │  v_r    │    I      │    <- identity block
//!         └─────────┴───────────┘
//! ```
//!
//! where the top-left entry is a scalar, the first row holds a (t-1)-vector, the first
//! column holds a (t-1)-vector, and the bottom-right block is the (t-1)x(t-1) identity.
//!
//! Multiplying by S_r costs only O(t) operations instead of O(t^2). Over RP partial
//! rounds, this saves O(RP * t^2) -> O(RP * t).
//!
//! # Optimized Partial Round Structure
//!
//! After this transformation, the RP partial rounds become:
//!
//! ```text
//!   1. Add first_round_constants (full WIDTH-vector)
//!   2. Multiply by m_i (dense transition matrix, applied once)
//!   3. For each of RP rounds:
//!      a. S-box on state[0]:  state[0] = state[0]^D
//!      b. Add scalar constant to state[0] (all rounds except the last)
//!      c. Sparse matrix multiply via cheap_matmul
//! ```
//!
//! # References
//!
//! - Grassi et al., "Poseidon: A New Hash Function for Zero-Knowledge Proof Systems",
//!   USENIX Security 2021. <https://eprint.iacr.org/2019/458>
//! - HorizenLabs reference implementation: <https://github.com/HorizenLabs/poseidon2>

use alloc::vec::Vec;

use p3_field::{Algebra, Field, InjectiveMonomial, PrimeCharacteristicRing};

use crate::external::mds_multiply;

/// Pre-computed constants for the RP partial (internal) rounds.
///
/// These are produced by the sparse matrix decomposition in [`crate::utils`].
#[derive(Debug, Clone)]
pub struct PartialRoundConstants<F, const WIDTH: usize> {
    /// Full WIDTH-vector of optimized round constants, added once before the
    /// transition matrix m_i.
    ///
    /// This vector absorbs the original round constants from all RP partial rounds
    /// via backward substitution through M^{-1}.
    pub first_round_constants: [F; WIDTH],

    /// Dense transition matrix m_i, applied once before the partial round loop.
    ///
    /// This is the accumulated product of sparse matrix factors from the
    /// sparse matrix decomposition, transposed to match the HorizenLabs convention.
    pub m_i: [[F; WIDTH]; WIDTH],

    /// Per-round full first row of the sparse matrix, pre-assembled for
    /// branch-free dot product computation.
    ///
    /// `sparse_first_row[r] = [mds_0_0, ŵ_r[0], ŵ_r[1], ..., ŵ_r[WIDTH-2]]`
    ///
    /// where `mds_0_0` is the top-left entry of the original MDS matrix (same for
    /// all rounds) and `ŵ_r` is the per-round first-row vector from the sparse
    /// factorization.
    ///
    /// Length = RP. Stored in forward application order.
    pub sparse_first_row: Vec<[F; WIDTH]>,

    /// Per-round first-column vectors for the sparse matrix multiply
    /// (excluding the `[0,0]` entry).
    ///
    /// `v[r]` has WIDTH elements: `[v_r[0], v_r[1], ..., v_r[WIDTH-2], 0]`.
    /// Only the first WIDTH-1 entries are meaningful; the last is padding.
    ///
    /// Length = RP. Stored in forward application order.
    pub v: Vec<[F; WIDTH]>,

    /// Optimized scalar round constants for partial rounds 0 through RP-2.
    ///
    /// The last partial round has no additive constant (it was absorbed by the
    /// backward substitution). Length = RP - 1.
    pub round_constants: Vec<F>,
}

/// Construct a partial round layer from pre-computed constants.
pub trait PartialRoundLayerConstructor<F: Field, const WIDTH: usize> {
    /// Build the layer from the sparse-form optimized constants.
    fn new_from_constants(constants: PartialRoundConstants<F, WIDTH>) -> Self;
}

/// The partial (internal) round layer of the Poseidon permutation.
///
/// Implementors apply all RP partial rounds to the state.
///
/// Field-specific implementations (e.g., NEON, AVX2) can override the generic behavior.
pub trait PartialRoundLayer<R, const WIDTH: usize, const D: u64>: Sync + Clone
where
    R: PrimeCharacteristicRing,
{
    /// Apply all RP partial rounds to the state.
    fn permute_state(&self, state: &mut [R; WIDTH]);
}

/// Sparse matrix-vector multiplication in O(WIDTH) operations.
///
/// Replaces the full O(t^2) MDS multiply in each partial round. Computes
/// `state <- S * state` where S has the structure:
///
/// ```text
///   S = ┌─────────┬──────┐
///       │ mds_0_0 │  ŵ   │    state'[0] = mds_0_0 * s[0] + Σ ŵ[j] * s[j+1]
///       ├─────────┼──────┤
///       │    v    │   I  │    state'[i] = s[i] + v[i-1] * s[0],  for i ≥ 1
///       └─────────┴──────┘
/// ```
///
/// `first_row` contains the pre-assembled full first row `[mds_0_0, ŵ[0], ..., ŵ[WIDTH-2]]`,
/// enabling a branch-free dot product for the new `state[0]`.
///
/// `v` contains the first-column vector (excluding `[0,0]`): `[v[0], ..., v[WIDTH-2], 0]`.
///
/// The identity block means `state[1..]` is updated by a simple rank-1 correction
/// (add a multiple of the old `state[0]`), and `state[0]` is a dot product.
#[inline(always)]
pub fn cheap_matmul<F: Field, A: Algebra<F>, const WIDTH: usize>(
    state: &mut [A; WIDTH],
    first_row: &[F; WIDTH],
    v: &[F; WIDTH],
) {
    // Save state[0] before it is overwritten.
    let old_s0 = state[0].clone();

    // Compute new state[0] = dot(first_row, state).
    state[0] = A::mixed_dot_product(state, first_row);

    // Rank-1 update: state[i] += v[i-1] * old_s0, for i = 1..WIDTH.
    for i in 1..WIDTH {
        state[i] += old_s0.clone() * v[i - 1];
    }
}

/// Generic implementation of the partial round permutation.
///
/// See the module-level documentation for the algorithm structure.
#[inline]
pub fn partial_permute_state<
    F: Field,
    A: Algebra<F> + InjectiveMonomial<D>,
    const WIDTH: usize,
    const D: u64,
>(
    state: &mut [A; WIDTH],
    constants: &PartialRoundConstants<F, WIDTH>,
) {
    // Add the full first-round constant vector.
    for (s, &rc) in state.iter_mut().zip(constants.first_round_constants.iter()) {
        *s += rc;
    }

    // Apply the dense transition matrix m_i (once).
    mds_multiply(state, &constants.m_i);

    let rounds_p = constants.sparse_first_row.len();

    // Partial rounds 0..RP-2: S-box + scalar RC + sparse matmul.
    // The last round is handled separately to avoid a branch per iteration.
    for r in 0..rounds_p - 1 {
        // S-box on state[0] only.
        state[0] = state[0].injective_exp_n();

        // Add scalar round constant.
        state[0] += constants.round_constants[r];

        // Sparse matrix multiply.
        cheap_matmul(state, &constants.sparse_first_row[r], &constants.v[r]);
    }

    // Last partial round: S-box + sparse matmul (no round constant).
    state[0] = state[0].injective_exp_n();
    cheap_matmul(
        state,
        &constants.sparse_first_row[rounds_p - 1],
        &constants.v[rounds_p - 1],
    );
}
