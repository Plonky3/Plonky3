use p3_field::{Field, PrimeCharacteristicRing};
use p3_poseidon::PoseidonConstants;
use p3_poseidon::utils::{circulant_to_dense, forward_constant_substitution};
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

/// Pre-computed round constants for the Poseidon1 AIR.
///
/// Uses forward constant substitution to reduce partial round constants
/// from WIDTH per round to 1 scalar per round plus a residual vector.
///
/// The dense MDS matrix is stored directly, so the AIR can apply it generically
/// over any `PrimeCharacteristicRing` without needing a separate trait.
#[derive(Debug, Clone)]
pub struct RoundConstants<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    pub(crate) beginning_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
    pub(crate) partial_round_constants: [F; PARTIAL_ROUNDS],
    /// Residual vector added after all partial rounds.
    ///
    /// This accounts for the `state[1..WIDTH]` constants that were folded forward
    /// through the MDS matrix during forward constant substitution.
    pub(crate) partial_round_residual: [F; WIDTH],
    pub(crate) ending_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
    /// Dense MDS matrix used in every round (full and partial).
    pub(crate) mds_matrix: [[F; WIDTH]; WIDTH],
}

impl<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
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
    /// Construct AIR round constants from raw Poseidon1 parameters.
    ///
    /// Applies forward constant substitution to compress the partial round constants
    /// from a full WIDTH-vector per round to a single scalar per round plus a residual.
    pub fn from_poseidon_constants(raw: &PoseidonConstants<F, WIDTH>) -> Self {
        let half_f = raw.rounds_f / 2;
        assert_eq!(half_f, HALF_FULL_ROUNDS);
        assert_eq!(raw.rounds_p, PARTIAL_ROUNDS);

        let initial_rc = &raw.round_constants[..half_f];
        let partial_rc = &raw.round_constants[half_f..half_f + raw.rounds_p];
        let terminal_rc = &raw.round_constants[half_f + raw.rounds_p..];

        let mds = circulant_to_dense(&raw.mds_circ_col);
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
