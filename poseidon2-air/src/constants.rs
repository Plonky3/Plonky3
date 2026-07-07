use core::fmt::{self, Display, Formatter};

use p3_field::PrimeCharacteristicRing;
use p3_poseidon2::ExternalLayerConstants;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

/// Error returned by [`RoundConstants::try_from_layers`] when the supplied constants
/// don't match this AIR's `WIDTH`/`HALF_FULL_ROUNDS`/`PARTIAL_ROUNDS` shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundConstantsShapeError {
    /// The number of initial (or terminal) full-round constant rows didn't match
    /// `HALF_FULL_ROUNDS`.
    FullRounds { expected: usize, got: usize },
    /// The number of partial-round constants didn't match `PARTIAL_ROUNDS`.
    PartialRounds { expected: usize, got: usize },
}

impl Display for RoundConstantsShapeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::FullRounds { expected, got } => {
                write!(f, "expected {expected} full-round constant rows, got {got}")
            }
            Self::PartialRounds { expected, got } => {
                write!(f, "expected {expected} partial-round constants, got {got}")
            }
        }
    }
}

/// Round constants for Poseidon2, in a format that's convenient for the AIR.
#[derive(Debug, Clone)]
pub struct RoundConstants<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    pub(crate) beginning_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
    pub(crate) partial_round_constants: [F; PARTIAL_ROUNDS],
    pub(crate) ending_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
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
        ending_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
    ) -> Self {
        Self {
            beginning_full_round_constants,
            partial_round_constants,
            ending_full_round_constants,
        }
    }

    pub fn from_rng<R: Rng>(rng: &mut R) -> Self
    where
        StandardUniform: Distribution<F> + Distribution<[F; WIDTH]>,
    {
        Self {
            beginning_full_round_constants: core::array::from_fn(|_| rng.sample(StandardUniform)),
            partial_round_constants: core::array::from_fn(|_| rng.sample(StandardUniform)),
            ending_full_round_constants: core::array::from_fn(|_| rng.sample(StandardUniform)),
        }
    }

    /// Build round constants from the canonical [`p3_poseidon2::Poseidon2`] constants,
    /// so this AIR can prove the same permutation instance that's actually used to
    /// hash (e.g. Merkle-tree compression), rather than only throwaway `from_rng` instances.
    pub fn try_from_layers(
        external: &ExternalLayerConstants<F, WIDTH>,
        internal: &[F],
    ) -> Result<Self, RoundConstantsShapeError> {
        let initial = external.get_initial_constants();
        let terminal = external.get_terminal_constants();
        if initial.len() != HALF_FULL_ROUNDS {
            return Err(RoundConstantsShapeError::FullRounds {
                expected: HALF_FULL_ROUNDS,
                got: initial.len(),
            });
        }
        if terminal.len() != HALF_FULL_ROUNDS {
            return Err(RoundConstantsShapeError::FullRounds {
                expected: HALF_FULL_ROUNDS,
                got: terminal.len(),
            });
        }
        if internal.len() != PARTIAL_ROUNDS {
            return Err(RoundConstantsShapeError::PartialRounds {
                expected: PARTIAL_ROUNDS,
                got: internal.len(),
            });
        }
        Ok(Self {
            beginning_full_round_constants: core::array::from_fn(|i| initial[i].clone()),
            partial_round_constants: core::array::from_fn(|i| internal[i].clone()),
            ending_full_round_constants: core::array::from_fn(|i| terminal[i].clone()),
        })
    }

    /// The initial (beginning) full-round constants.
    pub const fn beginning_full_round_constants(&self) -> &[[F; WIDTH]; HALF_FULL_ROUNDS] {
        &self.beginning_full_round_constants
    }

    /// The partial-round constants.
    pub const fn partial_round_constants(&self) -> &[F; PARTIAL_ROUNDS] {
        &self.partial_round_constants
    }

    /// The terminal (ending) full-round constants.
    pub const fn ending_full_round_constants(&self) -> &[[F; WIDTH]; HALF_FULL_ROUNDS] {
        &self.ending_full_round_constants
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    const WIDTH: usize = 16;
    const HALF_FULL_ROUNDS: usize = 4;
    const PARTIAL_ROUNDS: usize = 13;

    fn layers(
        half_full_rounds: usize,
        partial_rounds: usize,
    ) -> (ExternalLayerConstants<BabyBear, WIDTH>, Vec<BabyBear>) {
        let mut rng = SmallRng::seed_from_u64(0);
        let external = ExternalLayerConstants::new_from_rng(2 * half_full_rounds, &mut rng);
        let internal = rng
            .sample_iter(StandardUniform)
            .take(partial_rounds)
            .collect();
        (external, internal)
    }

    #[test]
    fn try_from_layers_accepts_matching_shape() {
        let (external, internal) = layers(HALF_FULL_ROUNDS, PARTIAL_ROUNDS);
        assert!(
            RoundConstants::<BabyBear, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>::try_from_layers(
                &external, &internal,
            )
            .is_ok()
        );
    }

    #[test]
    fn try_from_layers_reports_the_actual_full_round_length() {
        let (external, internal) = layers(HALF_FULL_ROUNDS - 1, PARTIAL_ROUNDS);
        let err =
            RoundConstants::<BabyBear, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>::try_from_layers(
                &external, &internal,
            )
            .unwrap_err();
        assert_eq!(
            err,
            RoundConstantsShapeError::FullRounds {
                expected: HALF_FULL_ROUNDS,
                got: HALF_FULL_ROUNDS - 1,
            }
        );
    }

    #[test]
    fn try_from_layers_reports_the_actual_partial_round_length() {
        let (external, internal) = layers(HALF_FULL_ROUNDS, PARTIAL_ROUNDS - 2);
        let err =
            RoundConstants::<BabyBear, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>::try_from_layers(
                &external, &internal,
            )
            .unwrap_err();
        assert_eq!(
            err,
            RoundConstantsShapeError::PartialRounds {
                expected: PARTIAL_ROUNDS,
                got: PARTIAL_ROUNDS - 2,
            }
        );
    }
}
