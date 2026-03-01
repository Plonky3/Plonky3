//! Full (external) round layers for the Poseidon permutation.
//!
//! # Overview
//!
//! Full rounds apply the S-box to **every** state element, providing strong resistance
//! against statistical attacks (differential, linear, truncated differential, rebound).
//! The Poseidon paper requires at least RF = 6 full rounds for 128-bit security against
//! these attacks (see Section 5 and Appendix C of the paper).
//!
//! # Round Structure
//!
//! Each full round applies three operations in sequence:
//!
//! ```text
//!   state → AddRoundConstants → S-box(all elements) → MDS multiply → state'
//! ```
//!
//! The MDS multiply is dispatched via the [`Permutation`] trait, allowing concrete fields
//! to use fast convolution (e.g., Karatsuba) while generic `Algebra<F>` types fall back
//! to O(t^2) dense multiplication.
//!
//! # Cost
//!
//! Each full round costs t S-box evaluations + O(t^2) for the dense MDS multiply,
//! giving a total full-round cost of O(RF * t^2). Since RF is small (typically 8),
//! this is acceptable even for large t.

use alloc::vec::Vec;

use p3_field::{Algebra, Field, InjectiveMonomial, PrimeCharacteristicRing};
use p3_symmetric::Permutation;

/// Pre-computed constants for the full (external) rounds.
///
/// The full rounds are split equally: half before the partial rounds (initial),
/// and half after (terminal).
///
/// The MDS matrix is **not** stored here. It is dispatched through a permutation
/// trait at the call site. This allows concrete fields to use optimized
/// implementations (e.g., Karatsuba convolution) while generic algebra types
/// fall back to dense O(t^2) multiplication.
#[derive(Debug, Clone)]
pub struct FullRoundConstants<F, const WIDTH: usize> {
    /// Round constants for the initial full rounds.
    pub initial: Vec<[F; WIDTH]>,

    /// Round constants for the terminal full rounds.
    pub terminal: Vec<[F; WIDTH]>,
}

/// Construct a full round layer from pre-computed constants.
pub trait FullRoundLayerConstructor<F: Field, const WIDTH: usize> {
    /// Build the layer from the full-round constants.
    fn new_from_constants(constants: FullRoundConstants<F, WIDTH>) -> Self;
}

/// The full (external) round layer of the Poseidon permutation.
///
/// Implementors apply the RF/2 initial or terminal full rounds to the state.
/// Field-specific implementations (e.g., NEON, AVX2) can override the generic
/// behavior for better performance.
pub trait FullRoundLayer<R, const WIDTH: usize, const D: u64>: Sync + Clone
where
    R: PrimeCharacteristicRing,
{
    /// Apply the RF/2 initial full rounds.
    fn permute_state_initial(&self, state: &mut [R; WIDTH]);

    /// Apply the RF/2 terminal full rounds.
    fn permute_state_terminal(&self, state: &mut [R; WIDTH]);
}

/// Dense MDS matrix-vector multiplication.
///
/// Standard O(t^2) matrix-vector product. Used for the dense transition matrix
/// in partial rounds. Full rounds should prefer the trait-dispatched MDS multiply
/// for sub-O(t^2) performance on concrete fields.
#[inline]
pub fn mds_multiply<F: Field, A: Algebra<F>, const WIDTH: usize>(
    state: &mut [A; WIDTH],
    mds: &[[F; WIDTH]; WIDTH],
) {
    // Snapshot the current state before overwriting.
    let input = state.clone();

    // Compute each output element as a dot product of one MDS row with the input.
    for (out, row) in state.iter_mut().zip(mds.iter()) {
        *out = A::mixed_dot_product(&input, row);
    }
}

/// Apply the initial full rounds (generic implementation).
///
/// Each round: add round constants, S-box on all elements, MDS multiply.
/// The MDS multiply is dispatched via the permutation trait parameter.
#[inline]
pub fn full_round_initial_permute_state<
    F: Field,
    A: Algebra<F> + InjectiveMonomial<D>,
    Mds: Permutation<[A; WIDTH]>,
    const WIDTH: usize,
    const D: u64,
>(
    state: &mut [A; WIDTH],
    constants: &FullRoundConstants<F, WIDTH>,
    mds: &Mds,
) {
    for round_constants in &constants.initial {
        // AddRoundConstants: state[i] += rc[i].
        for (s, &rc) in state.iter_mut().zip(round_constants.iter()) {
            *s += rc;
        }
        // S-box: state[i] = state[i]^D for all i.
        for s in state.iter_mut() {
            *s = s.injective_exp_n();
        }
        // MixLayer: dispatched via Permutation trait.
        mds.permute_mut(state);
    }
}

/// Apply the terminal full rounds (generic implementation).
///
/// Same structure as the initial full rounds, but uses the terminal constants.
#[inline]
pub fn full_round_terminal_permute_state<
    F: Field,
    A: Algebra<F> + InjectiveMonomial<D>,
    Mds: Permutation<[A; WIDTH]>,
    const WIDTH: usize,
    const D: u64,
>(
    state: &mut [A; WIDTH],
    constants: &FullRoundConstants<F, WIDTH>,
    mds: &Mds,
) {
    for round_constants in &constants.terminal {
        // AddRoundConstants: state[i] += rc[i].
        for (s, &rc) in state.iter_mut().zip(round_constants.iter()) {
            *s += rc;
        }
        // S-box: state[i] = state[i]^D for all i.
        for s in state.iter_mut() {
            *s = s.injective_exp_n();
        }
        // MixLayer: dispatched via Permutation trait.
        mds.permute_mut(state);
    }
}
