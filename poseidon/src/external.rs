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
//! The dense MDS matrix is the same for every round. Only the round constants change.
//!
//! # Cost
//!
//! Each full round costs t S-box evaluations + O(t^2) for the dense MDS multiply,
//! giving a total full-round cost of O(RF * t^2). Since RF is small (typically 8),
//! this is acceptable even for large t.

use alloc::vec::Vec;

use p3_field::{Algebra, Field, InjectiveMonomial, PrimeCharacteristicRing};

/// Pre-computed constants for the full (external) rounds.
///
/// The RF full rounds are split equally: RF/2 initial rounds before the partial rounds,
/// and RF/2 terminal rounds after.
#[derive(Debug, Clone)]
pub struct FullRoundConstants<F, const WIDTH: usize> {
    /// Round constants for the RF/2 initial full rounds.
    ///
    /// Each entry is a WIDTH-vector added to the state at the start of one round.
    pub initial: Vec<[F; WIDTH]>,

    /// Round constants for the RF/2 terminal full rounds.
    ///
    /// Same structure as `initial`, but for the rounds after the partial rounds.
    pub terminal: Vec<[F; WIDTH]>,

    /// The dense t x t MDS matrix, shared by all full rounds.
    ///
    /// This matrix has branch number t+1, guaranteeing that any non-zero input
    /// difference activates at least t+1 S-boxes across two consecutive rounds
    /// (wide-trail argument).
    pub mds: [[F; WIDTH]; WIDTH],
}

/// Construct a full round layer from pre-computed constants.
pub trait FullRoundLayerConstructor<F: Field, const WIDTH: usize> {
    /// Build the layer from the full-round constants and the dense MDS matrix.
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

/// Dense MDS matrix-vector multiplication: `state <- MDS * state`.
///
/// This is the standard O(t^2) matrix-vector product used in every full round.
/// Partial rounds use a sparse variant instead (see the internal module).
#[inline]
pub fn mds_multiply<F: Field, A: Algebra<F>, const WIDTH: usize>(
    state: &mut [A; WIDTH],
    mds: &[[F; WIDTH]; WIDTH],
) {
    // Snapshot the current state before overwriting.
    let input = state.clone();

    // Compute each output element as a dot product of one MDS row with the input.
    for (out, row) in state.iter_mut().zip(mds.iter()) {
        *out = input
            .iter()
            .zip(row.iter())
            .map(|(x, &m)| x.clone() * m)
            .sum();
    }
}

/// Apply the RF/2 initial full rounds (generic implementation).
///
/// Each round: add round constants → S-box on all elements → dense MDS multiply.
#[inline]
pub fn full_round_initial_permute_state<
    F: Field,
    A: Algebra<F> + InjectiveMonomial<D>,
    const WIDTH: usize,
    const D: u64,
>(
    state: &mut [A; WIDTH],
    constants: &FullRoundConstants<F, WIDTH>,
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
        // MixLayer: state = MDS * state.
        mds_multiply(state, &constants.mds);
    }
}

/// Apply the RF/2 terminal full rounds (generic implementation).
///
/// Same structure as the initial full rounds, but uses the terminal round constants.
#[inline]
pub fn full_round_terminal_permute_state<
    F: Field,
    A: Algebra<F> + InjectiveMonomial<D>,
    const WIDTH: usize,
    const D: u64,
>(
    state: &mut [A; WIDTH],
    constants: &FullRoundConstants<F, WIDTH>,
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
        // MixLayer: state = MDS * state.
        mds_multiply(state, &constants.mds);
    }
}
