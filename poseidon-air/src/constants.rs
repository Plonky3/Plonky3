//! Pre-computed round constants for the Poseidon1 AIR.
//!
//! Stores constants in a form optimized for constraint evaluation, using
//! **forward constant substitution** to compress partial round constants.
//!
//! In partial rounds, only `state[0]` passes through the S-box. The constants
//! for `state[1..WIDTH]` are linear and can be folded through the MDS matrix:
//!
//! ```text
//!   Textbook (per round):                After substitution (per round):
//!     state[i] += rc[i]  for all i         state[0] += scalar_constant
//!     state[0] = S-box(state[0])           state[0] = S-box(state[0])
//!     state = MDS * state                  state = MDS * state
//!
//!                                        After all partial rounds:
//!                                          state += residual
//! ```
//!
//! This reduces storage from `WIDTH × RP` to `RP + WIDTH` field elements.

use p3_field::{Field, PrimeCharacteristicRing};
use p3_poseidon::PoseidonConstants;
use p3_poseidon::utils::{circulant_to_dense, forward_constant_substitution};
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

/// Pre-computed round constants for the Poseidon1 AIR.
///
/// Stores the constants in their post-substitution form, ready for direct use
/// in both trace generation and constraint evaluation.
///
/// # Type Parameters
///
/// - `F`: The prime field type.
/// - `WIDTH`: Permutation state width (`t` in the paper).
/// - `HALF_FULL_ROUNDS`: Number of full rounds per half (`RF/2`).
/// - `PARTIAL_ROUNDS`: Number of partial rounds (`RP`).
#[derive(Debug, Clone)]
pub struct RoundConstants<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    /// Round constants for the first `RF/2` full rounds.
    ///
    /// Each full round adds a complete `[F; WIDTH]` vector to the state
    /// before the S-box layer.
    pub(crate) beginning_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],

    /// Scalar constants for the `RP` partial rounds.
    ///
    /// After forward constant substitution, each partial round adds only
    /// a single scalar to `state[0]` before the S-box.
    pub(crate) partial_round_constants: [F; PARTIAL_ROUNDS],

    /// Residual vector from forward constant substitution.
    ///
    /// The `state[1..WIDTH]` constants that were folded forward through the MDS
    /// matrix accumulate into this residual, which is added to the state once
    /// after all partial rounds complete.
    pub(crate) partial_round_residual: [F; WIDTH],

    /// Round constants for the last `RF/2` full rounds.
    pub(crate) ending_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],

    /// Dense MDS matrix applied in every round (both full and partial).
    ///
    /// In Poseidon1, this is a full `WIDTH × WIDTH` MDS matrix.
    /// It is the same matrix for all rounds.
    pub(crate) mds_matrix: [[F; WIDTH]; WIDTH],
}

impl<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
    /// Construct `RoundConstants` from pre-computed values.
    pub const fn new(
        beginning_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
        partial_round_constants: [F; PARTIAL_ROUNDS],
        partial_round_residual: [F; WIDTH],
        ending_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
        mds_matrix: [[F; WIDTH]; WIDTH],
    ) -> Self {
        Self {
            beginning_full_round_constants,
            partial_round_constants,
            partial_round_residual,
            ending_full_round_constants,
            mds_matrix,
        }
    }

    /// Generate random round constants.
    pub fn from_rng<R: Rng>(rng: &mut R) -> Self
    where
        StandardUniform: Distribution<F> + Distribution<[F; WIDTH]>,
    {
        Self {
            beginning_full_round_constants: core::array::from_fn(|_| rng.random()),
            partial_round_constants: core::array::from_fn(|_| rng.random()),
            partial_round_residual: rng.random(),
            ending_full_round_constants: core::array::from_fn(|_| rng.random()),
            mds_matrix: core::array::from_fn(|_| rng.random()),
        }
    }
}

impl<F: Field, const WIDTH: usize, const HALF_FULL_ROUNDS: usize, const PARTIAL_ROUNDS: usize>
    RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
    /// Convert raw Poseidon1 parameters into AIR-optimized round constants.
    ///
    /// This method performs two transformations:
    ///
    /// 1. **Circulant-to-dense expansion**: Converts the MDS circulant column
    ///    into a full `WIDTH × WIDTH` dense matrix.
    ///
    /// 2. **Forward constant substitution**: Compresses the partial round
    ///    constants from `[F; WIDTH]` per round to a single scalar per round
    ///    plus a residual vector.
    ///
    /// # Panics
    ///
    /// Panics if `raw.rounds_f / 2 != HALF_FULL_ROUNDS` or `raw.rounds_p != PARTIAL_ROUNDS`.
    pub fn from_poseidon_constants(raw: &PoseidonConstants<F, WIDTH>) -> Self {
        // Verify round counts match the const generics.
        let half_f = raw.rounds_f / 2;
        assert_eq!(half_f, HALF_FULL_ROUNDS);
        assert_eq!(raw.rounds_p, PARTIAL_ROUNDS);

        // Split the flat round constant list into three sections:
        //   [0..half_f)                  → initial full rounds
        //   [half_f..half_f + RP)        → partial rounds
        //   [half_f + RP..)              → terminal full rounds
        let initial_rc = &raw.round_constants[..half_f];
        let partial_rc = &raw.round_constants[half_f..half_f + raw.rounds_p];
        let terminal_rc = &raw.round_constants[half_f + raw.rounds_p..];

        // Expand the circulant MDS column to a dense matrix.
        let mds = circulant_to_dense(&raw.mds_circ_col);

        // Apply forward constant substitution to compress partial round constants.
        let (scalar_constants, residual) = forward_constant_substitution(&mds, partial_rc);

        Self {
            beginning_full_round_constants: core::array::from_fn(|i| initial_rc[i]),
            partial_round_constants: core::array::from_fn(|i| scalar_constants[i]),
            partial_round_residual: residual,
            ending_full_round_constants: core::array::from_fn(|i| terminal_rc[i]),
            mds_matrix: mds,
        }
    }
}
