use p3_field::{Field, PrimeField, PrimeField32, PrimeField64};
use p3_maybe_rayon::prelude::*;
use p3_symmetric::CryptographicPermutation;
use tracing::instrument;

use crate::{
    CanObserve, CanSampleBits, CanSampleUniformBits, DuplexChallenger, MultiField32Challenger,
    UniformSamplingField,
};

/// Trait for challengers that support proof-of-work (PoW) grinding.
///
/// A `GrindingChallenger` can:
/// - Absorb a candidate witness into the transcript
/// - Sample random bitstrings to check the PoW condition
/// - Brute-force search for a valid witness that satisfies the PoW
///
/// This trait is typically used in protocols requiring computational effort
/// from the prover.
pub trait GrindingChallenger:
    CanObserve<Self::Witness> + CanSampleBits<usize> + Sync + Clone
{
    /// The underlying field element type used as the witness.
    type Witness: Field;

    /// Perform a brute-force search to find a valid PoW witness.
    ///
    /// Given a `bits` parameter, this function searches for a field element
    /// `witness` such that after observing it, the next `bits` bits that challenger outputs
    /// are all `0`.
    fn grind(&mut self, bits: usize) -> Self::Witness;

    /// Check whether a given `witness` satisfies the PoW condition.
    ///
    /// After absorbing the witness, the challenger samples `bits` random bits
    /// and verifies that all bits sampled are zero.
    ///
    /// Returns `true` if the witness passes the PoW check, `false` otherwise.
    #[must_use]
    fn check_witness(&mut self, bits: usize, witness: Self::Witness) -> bool {
        if bits == 0 {
            return true;
        }
        self.observe(witness);
        self.sample_bits(bits) == 0
    }
}

/// Trait for challengers that support proof-of-work (PoW) grinding with
/// guaranteed uniformly sampled bits.
pub trait UniformGrindingChallenger:
    GrindingChallenger + CanSampleUniformBits<Self::Witness>
{
    /// Grinds based on *uniformly sampled bits*. This variant is allowed to do rejection
    /// sampling if a value is sampled that would violate our uniformity requirement
    /// (chance of about 1/P).
    ///
    /// Use this together with `check_witness_uniform`.
    fn grind_uniform(&mut self, bits: usize) -> Self::Witness;

    /// Grinds based on *uniformly sampled bits*. This variant errors if a value is
    /// sampled, which would violate our uniformity requirement (chance of about 1/P).
    /// See the `UniformSamplingField` trait implemented for each field for details.
    ///
    /// Use this together with `check_witness_uniform_may_error`.
    fn grind_uniform_may_error(&mut self, bits: usize) -> Self::Witness;

    /// Check whether a given `witness` satisfies the PoW condition.
    ///
    /// After absorbing the witness, the challenger samples `bits` random bits
    /// *uniformly* and verifies that all bits sampled are zero. The uniform
    /// sampling implies we do rejection sampling in about ~1/P cases.
    ///
    /// Returns `true` if the witness passes the PoW check, `false` otherwise.
    fn check_witness_uniform(&mut self, bits: usize, witness: Self::Witness) -> bool {
        self.observe(witness);
        self.sample_uniform_bits::<true>(bits)
            .expect("Error impossible here due to resampling strategy")
            == 0
    }

    /// Check whether a given `witness` satisfies the PoW condition.
    ///
    /// After absorbing the witness, the challenger samples `bits` random bits
    /// *uniformly* and verifies that all bits sampled are zero. In about ~1/P
    /// cases this function may error if a sampled value lies outside a range
    /// in which we can guarantee uniform bits.
    ///
    /// Returns `true` if the witness passes the PoW check, `false` otherwise.
    fn check_witness_uniform_may_error(&mut self, bits: usize, witness: Self::Witness) -> bool {
        self.observe(witness);
        self.sample_uniform_bits::<false>(bits)
            .is_ok_and(|v| v == 0)
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> GrindingChallenger
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: PrimeField64,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    type Witness = F;

    #[instrument(name = "grind for proof-of-work witness", skip_all)]
    fn grind(&mut self, bits: usize) -> Self::Witness {
        assert!(bits < (usize::BITS as usize), "bit count must be valid");
        assert!((1 << bits) < F::ORDER_U64);

        let witness = (0..F::ORDER_U64)
            .into_par_iter()
            .map(|i| unsafe {
                // i < F::ORDER_U64 by construction so this is safe.
                F::from_canonical_unchecked(i)
            })
            .find_any(|witness| self.clone().check_witness(bits, *witness))
            .expect("failed to find witness");
        assert!(self.check_witness(bits, witness));
        witness
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> UniformGrindingChallenger
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: UniformSamplingField + PrimeField64,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    #[instrument(name = "grind uniform for proof-of-work witness", skip_all)]
    fn grind_uniform(&mut self, bits: usize) -> Self::Witness {
        // Call the generic grinder with the "resample" checking logic.
        self.grind_generic(bits, |challenger, witness| {
            challenger.check_witness_uniform(bits, witness)
        })
    }
    #[instrument(name = "grind uniform may error for proof-of-work witness", skip_all)]
    fn grind_uniform_may_error(&mut self, bits: usize) -> Self::Witness {
        // Call the generic grinder with the "error" checking logic.
        self.grind_generic(bits, |challenger, witness| {
            challenger.check_witness_uniform_may_error(bits, witness)
        })
    }
}
impl<F, P, const WIDTH: usize, const RATE: usize> DuplexChallenger<F, P, WIDTH, RATE>
where
    F: PrimeField64,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    /// A generic, private helper for PoW grinding, parameterized by the checking function.
    fn grind_generic<CHECK>(&mut self, bits: usize, check_fn: CHECK) -> F
    where
        CHECK: Fn(&mut Self, F) -> bool + Sync + Send,
    {
        // Maybe check that bits is greater than 0?
        assert!(bits < (usize::BITS as usize), "bit count must be valid");
        assert!(
            (1u64 << bits) < F::ORDER_U64,
            "bit count exceeds field order"
        );
        // The core parallel brute-force search logic.
        let witness = (0..F::ORDER_U64)
            .into_par_iter()
            .map(|i| unsafe {
                // This is safe as i is always in range.
                F::from_canonical_unchecked(i)
            })
            .find_any(|&witness| check_fn(&mut self.clone(), witness))
            .expect("failed to find proof-of-work witness");
        // Run the check one last time on the *original* challenger to update its state
        // and confirm the witness is valid.
        assert!(check_fn(self, witness));
        witness
    }
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize> GrindingChallenger
    for MultiField32Challenger<F, PF, P, WIDTH, RATE>
where
    F: PrimeField32,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    type Witness = F;

    #[instrument(name = "grind for proof-of-work witness", skip_all)]
    fn grind(&mut self, bits: usize) -> Self::Witness {
        assert!(bits < (usize::BITS as usize), "bit count must be valid");
        assert!((1 << bits) < F::ORDER_U32);
        let witness = (0..F::ORDER_U32)
            .into_par_iter()
            .map(|i| unsafe {
                // i < F::ORDER_U32 by construction so this is safe.
                F::from_canonical_unchecked(i)
            })
            .find_any(|witness| self.clone().check_witness(bits, *witness))
            .expect("failed to find witness");
        assert!(self.check_witness(bits, witness));
        witness
    }
}
