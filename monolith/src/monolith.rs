//! The Monolith permutation family.
//!
//! Monolith is a ZK-friendly permutation operating on a state of WIDTH
//! field elements over a prime field. Each round applies three layers:
//!
//! - **Bars**: non-linear S-box on the first NUM_BARS elements (field-specific)
//! - **Bricks**: Feistel Type-3 mixing, s_i <- s_i + s_{i-1}^2 (degree 2)
//! - **Concrete**: circulant MDS matrix multiplication (degree 1)
//!
//! The full permutation structure is:
//!
//! ```text
//!   Concrete(state)
//!   for i in 0..NUM_FULL_ROUNDS:
//!       Bars(state) -> Bricks(state) -> Concrete(state) -> AddRC(state, rc[i])
//!   Bars(state) -> Bricks(state) -> Concrete(state)
//! ```
//!
//! The paper defines two instantiations:
//! - Monolith-31 over Mersenne31 (p = 2^31 - 1), WIDTH in {16, 24}
//! - Monolith-64 over Goldilocks (p = 2^64 - 2^32 + 1), WIDTH in {8, 12}
//!
//! Both use 6 total rounds (NUM_FULL_ROUNDS = 5 rounds with constants + 1 final round without).

use core::array;

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use p3_mds::MdsPermutation;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{CryptographicPermutation, Permutation};
use sha3::Shake128;
use sha3::digest::{ExtendableOutput, Update};

use crate::bars::MonolithBars;
use crate::bars::goldilocks::MonolithBarsGoldilocks;
use crate::bars::mersenne31::MonolithBarsM31;

/// A Monolith permutation instance, generic over the field and Bars layer.
///
/// The const generic NUM_FULL_ROUNDS counts the rounds that include round
/// constant addition. The total number of rounds is NUM_FULL_ROUNDS + 1
/// (the final round omits the constant-addition step).
///
/// # Standard parameters
///
/// | Variant     | Field      | WIDTH | NUM_FULL_ROUNDS | NUM_BARS |
/// |-------------|------------|-------|-----------------|----------|
/// | Monolith-31 | Mersenne31 | 16    | 5               | 8        |
/// | Monolith-31 | Mersenne31 | 24    | 5               | 8        |
/// | Monolith-64 | Goldilocks | 8     | 5               | 4        |
/// | Monolith-64 | Goldilocks | 12    | 5               | 4        |
#[derive(Debug, Clone)]
pub struct Monolith<F, B, Mds, const WIDTH: usize, const NUM_FULL_ROUNDS: usize>
where
    F: PrimeCharacteristicRing,
    B: MonolithBars<F, WIDTH>,
    Mds: MdsPermutation<F, WIDTH>,
{
    /// Round constants for each of the NUM_FULL_ROUNDS rounds.
    /// The final round has no round constants.
    pub round_constants: [[F; WIDTH]; NUM_FULL_ROUNDS],

    /// The Bars layer implementation (field-specific S-box logic).
    pub bars: B,

    /// The MDS matrix used in the Concrete layer.
    pub mds: Mds,
}

impl<F, B, Mds, const WIDTH: usize, const NUM_FULL_ROUNDS: usize>
    Monolith<F, B, Mds, WIDTH, NUM_FULL_ROUNDS>
where
    F: PrimeCharacteristicRing + Copy,
    B: MonolithBars<F, WIDTH>,
    Mds: MdsPermutation<F, WIDTH>,
{
    /// Construct a new Monolith instance.
    pub fn new(bars: B, mds: Mds) -> Self {
        const {
            assert!(WIDTH >= 8);
            assert!(WIDTH <= 24);
            assert!(WIDTH.is_multiple_of(4));
        }

        // Derive round constants from SHAKE-128 seeded with the Monolith domain separator:
        //
        // "Monolith" || width || num_rounds || prime || limb_bits.
        let num_rounds = (NUM_FULL_ROUNDS + 1) as u8;
        let mut shake = Shake128::default();
        shake.update(b"Monolith");
        shake.update(&[WIDTH as u8, num_rounds]);
        shake.update(B::PRIME_BYTES);
        shake.update(B::LIMB_BITS);
        let mut shake = shake.finalize_xof();

        // Fill each round's constants by rejection-sampling field elements.
        let round_constants =
            array::from_fn(|_| array::from_fn(|_| B::random_field_element(&mut shake)));

        Self {
            round_constants,
            bars,
            mds,
        }
    }

    /// Concrete layer: multiply the state by the circulant MDS matrix.
    #[inline]
    pub fn concrete(&self, state: &mut [F; WIDTH]) {
        self.mds.permute_mut(state);
    }

    /// Add round constants element-wise to the state.
    #[inline]
    pub fn add_round_constants(state: &mut [F; WIDTH], round_constants: &[F; WIDTH]) {
        for (x, rc) in state.iter_mut().zip(round_constants) {
            *x += *rc;
        }
    }

    /// Bricks layer: Feistel Type-3 mixing with squaring.
    ///
    /// For i = 1, 2, ..., WIDTH-1: s_i <- s_i + s_{i-1}^2.
    /// The first element s_0 passes through unchanged.
    ///
    /// This is a degree-2 non-linear layer that provides diffusion
    /// when combined with the MDS Concrete layer.
    #[inline]
    pub fn bricks(state: &mut [F; WIDTH]) {
        // Iterate right-to-left so that state[i-1] is still its original
        // value when we read it for state[i] += state[i-1]^2.
        // This avoids cloning the entire state array.
        for i in (1..WIDTH).rev() {
            state[i] += state[i - 1].square();
        }
    }
}

impl<F, B, Mds, const WIDTH: usize, const NUM_FULL_ROUNDS: usize> Permutation<[F; WIDTH]>
    for Monolith<F, B, Mds, WIDTH, NUM_FULL_ROUNDS>
where
    F: PrimeCharacteristicRing + Copy + Sync,
    B: MonolithBars<F, WIDTH>,
    Mds: MdsPermutation<F, WIDTH>,
{
    fn permute_mut(&self, state: &mut [F; WIDTH]) {
        // Initial Concrete (no round constants).
        self.concrete(state);

        // NUM_FULL_ROUNDS rounds, each with Bars -> Bricks -> Concrete -> AddRC.
        for rc in &self.round_constants {
            self.bars.bars(state);
            Self::bricks(state);
            self.concrete(state);
            Self::add_round_constants(state, rc);
        }

        // Final round: Bars -> Bricks -> Concrete (no round constants).
        self.bars.bars(state);
        Self::bricks(state);
        self.concrete(state);
    }
}

impl<F, B, Mds, const WIDTH: usize, const NUM_FULL_ROUNDS: usize>
    CryptographicPermutation<[F; WIDTH]> for Monolith<F, B, Mds, WIDTH, NUM_FULL_ROUNDS>
where
    F: PrimeCharacteristicRing + Copy + Sync,
    B: MonolithBars<F, WIDTH>,
    Mds: MdsPermutation<F, WIDTH>,
{
}

/// Type alias for the Mersenne31 instantiation (Monolith-31).
pub type MonolithMersenne31<Mds, const WIDTH: usize, const NUM_FULL_ROUNDS: usize> =
    Monolith<Mersenne31, MonolithBarsM31, Mds, WIDTH, NUM_FULL_ROUNDS>;

/// Type alias for the Goldilocks instantiation with 8-bit lookups (Monolith-64, standard).
pub type MonolithGoldilocks8<Mds, const WIDTH: usize, const NUM_FULL_ROUNDS: usize> =
    Monolith<Goldilocks, MonolithBarsGoldilocks<8>, Mds, WIDTH, NUM_FULL_ROUNDS>;

/// Type alias for the Goldilocks instantiation with 16-bit lookups (Monolith-64, alternative).
pub type MonolithGoldilocks16<Mds, const WIDTH: usize, const NUM_FULL_ROUNDS: usize> =
    Monolith<Goldilocks, MonolithBarsGoldilocks<16>, Mds, WIDTH, NUM_FULL_ROUNDS>;

#[cfg(test)]
mod tests {
    use core::array;

    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::Permutation;

    use super::*;
    use crate::{MonolithMdsMatrixGoldilocks, MonolithMdsMatrixMersenne31};

    #[test]
    fn test_monolith_31_width_16() {
        // Known-answer test from the paper / reference implementation.
        // Input: [0, 1, 2, ..., 15], Width: 16, Rounds: 6 (NUM_FULL_ROUNDS=5).
        let bars = MonolithBarsM31;
        let mds = MonolithMdsMatrixMersenne31::<6>;
        let monolith: MonolithMersenne31<_, 16, 5> = MonolithMersenne31::new(bars, mds);

        let mut input: [Mersenne31; 16] = array::from_fn(Mersenne31::from_usize);

        // Expected output from the Monolith-31 reference test vector.
        let expected = [
            609156607, 290107110, 1900746598, 1734707571, 2050994835, 1648553244, 1307647296,
            1941164548, 1707113065, 1477714255, 1170160793, 93800695, 769879348, 375548503,
            1989726444, 1349325635,
        ]
        .map(Mersenne31::from_u32);

        // Use the Permutation trait method.
        monolith.permute_mut(&mut input);

        assert_eq!(input, expected);
    }

    #[test]
    fn test_s_box_fixed_points() {
        // The 8-bit S-box must have 0x00 and 0xFF as fixed points.
        // This is a requirement of the Kintsugi strategy for Mersenne primes.
        assert_eq!(MonolithBarsM31::s_box(0x00), 0x00);
        assert_eq!(MonolithBarsM31::s_box(0xFF), 0xFF);
    }

    #[test]
    fn test_final_s_box_fixed_points() {
        // The 7-bit S-box must have 0x00 and 0x7F as fixed points.
        assert_eq!(MonolithBarsM31::final_s_box(0x00), 0x00);
        assert_eq!(MonolithBarsM31::final_s_box(0x7F), 0x7F);
    }

    #[test]
    fn test_bricks_first_element_unchanged() {
        // The Feistel Type-3 Bricks layer leaves the first element unchanged.
        let mut state: [Mersenne31; 16] = array::from_fn(Mersenne31::from_usize);
        let first = state[0];
        Monolith::<Mersenne31, MonolithBarsM31, MonolithMdsMatrixMersenne31<6>, 16, 5>::bricks(
            &mut state,
        );
        assert_eq!(state[0], first);
    }

    #[test]
    fn test_bricks_second_element() {
        // For Bricks, state[1] should become state[1] + state[0]^2.
        let mut state: [Mersenne31; 16] = array::from_fn(Mersenne31::from_usize);
        let expected_1 = state[1] + state[0] * state[0];
        Monolith::<Mersenne31, MonolithBarsM31, MonolithMdsMatrixMersenne31<6>, 16, 5>::bricks(
            &mut state,
        );
        assert_eq!(state[1], expected_1);
    }

    #[test]
    fn test_permutation_deterministic() {
        // Two invocations with the same input must produce the same output.
        let bars = MonolithBarsM31;
        let mds = MonolithMdsMatrixMersenne31::<6>;
        let monolith: MonolithMersenne31<_, 16, 5> = MonolithMersenne31::new(bars, mds);

        let input: [Mersenne31; 16] = array::from_fn(Mersenne31::from_usize);

        let output1 = monolith.permute(input);
        let output2 = monolith.permute(input);
        assert_eq!(output1, output2);
    }

    #[test]
    fn test_monolith_64_width_8_deterministic() {
        // Construct a Monolith-64 instance with WIDTH=8 (compression mode).
        let bars = MonolithBarsGoldilocks::<8>;
        let mds = MonolithMdsMatrixGoldilocks;
        let monolith: MonolithGoldilocks8<_, 8, 5> = MonolithGoldilocks8::new(bars, mds);

        let input: [Goldilocks; 8] = array::from_fn(|i| Goldilocks::new(i as u64));

        // Verify deterministic output.
        let output1 = monolith.permute(input);
        let output2 = monolith.permute(input);
        assert_eq!(output1, output2);
    }

    #[test]
    fn test_monolith_64_width_12_deterministic() {
        // Construct a Monolith-64 instance with WIDTH=12 (sponge mode).
        let bars = MonolithBarsGoldilocks::<8>;
        let mds = MonolithMdsMatrixGoldilocks;
        let monolith: MonolithGoldilocks8<_, 12, 5> = MonolithGoldilocks8::new(bars, mds);

        let input: [Goldilocks; 12] = array::from_fn(|i| Goldilocks::new(i as u64));

        // Verify deterministic output.
        let output1 = monolith.permute(input);
        let output2 = monolith.permute(input);
        assert_eq!(output1, output2);
    }

    #[test]
    fn test_monolith_64_width_12_known_answer() {
        // Known-answer test from the HorizenLabs reference implementation.
        // Input: [0, 1, 2, ..., 11], Width: 12, Rounds: 6 (NUM_FULL_ROUNDS=5).
        // Reference: https://github.com/HorizenLabs/monolith (LOOKUP_BITS=8).
        let bars = MonolithBarsGoldilocks::<8>;
        let mds = MonolithMdsMatrixGoldilocks;
        let monolith: MonolithGoldilocks8<_, 12, 5> = MonolithGoldilocks8::new(bars, mds);

        let mut input: [Goldilocks; 12] = array::from_fn(|i| Goldilocks::new(i as u64));

        let expected = [
            5867581605548782913,
            588867029099903233,
            6043817495575026667,
            805786589926590032,
            9919982299747097782,
            6718641691835914685,
            7951881005429661950,
            15453177927755089358,
            974633365445157727,
            9654662171963364206,
            6281307445101925412,
            13745376999934453119,
        ]
        .map(Goldilocks::new);

        monolith.permute_mut(&mut input);

        assert_eq!(input, expected);
    }

    #[test]
    fn test_monolith_64_width_12_known_answer_lookup16() {
        // Known-answer test from the HorizenLabs reference implementation.
        // Input: [0, 1, 2, ..., 11], Width: 12, Rounds: 6, LOOKUP_BITS=16.
        // Reference: https://github.com/HorizenLabs/monolith
        let bars = MonolithBarsGoldilocks::<16>;
        let mds = MonolithMdsMatrixGoldilocks;
        let monolith: MonolithGoldilocks16<_, 12, 5> = MonolithGoldilocks16::new(bars, mds);

        let mut input: [Goldilocks; 12] = array::from_fn(|i| Goldilocks::new(i as u64));

        let expected = [
            15270549627416999494,
            2608801733076195295,
            2511564300649802419,
            14351608014180687564,
            4101801939676807387,
            234091379199311770,
            3560400203616478913,
            17913168886441793528,
            7247432905090441163,
            667535998170608897,
            5848119428178849609,
            7505720212650520546,
        ]
        .map(Goldilocks::new);

        monolith.permute_mut(&mut input);

        assert_eq!(input, expected);
    }

    #[test]
    fn test_monolith_64_bars_nontrivial() {
        // Verify the Bars layer actually changes non-trivial inputs.
        let bars = MonolithBarsGoldilocks::<8>;
        let mut state: [Goldilocks; 8] =
            array::from_fn(|i| Goldilocks::new((i as u64 + 1) * 0x0123_4567_89AB_CDEFu64));
        let original = state;

        <MonolithBarsGoldilocks<8> as MonolithBars<Goldilocks, 8>>::bars(&bars, &mut state);

        // The first 4 elements (NUM_BARS=4) should change.
        for i in 0..4 {
            assert_ne!(
                state[i], original[i],
                "element {i} should change after Bars"
            );
        }
        // Elements 4..7 should be unchanged.
        for i in 4..8 {
            assert_eq!(
                state[i], original[i],
                "element {i} should be unchanged after Bars"
            );
        }
    }
}
