//! Generic challenger permutation config for the recursion circuit.
//!
//! Allows the verifier and circuit challenger to be parameterised by a permutation
//! config without naming a specific hash (e.g. Poseidon2).

use p3_circuit::ops::Poseidon2Config;

/// Config for the permutation used by the in-circuit challenger.
///
/// Implemented by concrete permutation configs (e.g. Poseidon2); the recursion
/// verifier and [`crate::CircuitChallenger`] use this trait so they do not depend
/// on a specific hash by name.
pub trait ChallengerPermConfig: Send + Sync {
    /// Extension degree used by the in-circuit permutation NPO (`Poseidon2Config::d()`).
    ///
    /// This need not match the STARK challenge extension `EF::DIMENSION` (e.g. base
    /// width-16 Poseidon2 with `d() == 1` can pair with a quartic or quintic challenge).
    fn extension_degree(&self) -> usize;

    /// Poseidon2 config if this is a Poseidon2 permutation; `None` otherwise.
    fn as_poseidon2(&self) -> Option<&Poseidon2Config>;
}

impl ChallengerPermConfig for Poseidon2Config {
    fn extension_degree(&self) -> usize {
        Self::d(*self)
    }

    fn as_poseidon2(&self) -> Option<&Poseidon2Config> {
        Some(self)
    }
}
