//! The Monolith permutation, and hash functions built from it.

extern crate alloc;

use alloc::sync::Arc;
use p3_symmetric::mds::{MDSPermutation, NaiveMDSMatrix};
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake128, Shake128Reader,
};

use core::{iter, u64};

use p3_field::{AbstractField, PrimeField32};
use p3_mersenne_31::Mersenne31;
use p3_symmetric::compression::PseudoCompressionFunction;
use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation};

use crate::monolith_mds::monolith_mds;

// The Monolith-31 permutation.
// Assumes that F is a 31-bit field (e.g. Mersenne31).
pub struct Monolith31<F: PrimeField32, const WIDTH: usize, const NUM_ROUNDS: usize> {
    pub round_constants: [[F; WIDTH]; NUM_ROUNDS],
    pub mds: Box<dyn MDSPermutation<F, WIDTH>>,
    pub lookup1: Vec<u16>,
    pub lookup2: Vec<u16>,
}

impl<F: PrimeField32, const WIDTH: usize, const NUM_ROUNDS: usize> Monolith31<F, WIDTH, NUM_ROUNDS> {
    pub const NUM_BARS: usize = 8;

    pub fn new() -> Self {
        assert_eq!(F::bits(), 31);
        assert!(WIDTH >= 16);
        assert_eq!(WIDTH % 4, 0);

        let round_constants = Self::instantiate_round_constants();
        let lookup1 = Self::instantiate_lookup1();
        let lookup2 = Self::instantiate_lookup2();
        let mds = monolith_mds("Monolith");

        Self {
            round_constants,
            mds,
            lookup1,
            lookup2,
        }
    }

    fn s_box(y: u8) -> u8 {
        let y_rot_1 = (y >> 7) | (y << 1);
        let y_rot_2 = (y >> 6) | (y << 2);
        let y_rot_3 = (y >> 5) | (y << 3);

        let tmp = y ^ !y_rot_1 & y_rot_2 & y_rot_3;
        (tmp >> 7) | (tmp << 1)
    }

    pub fn final_s_box(y: u8) -> u8 {
        debug_assert_eq!(y >> 7, 0); // must be a 7-bit value

        let y_rot_1 = (y >> 6) | (y << 1);
        let y_rot_2 = (y >> 5) | (y << 2);

        let tmp = (y ^ !y_rot_1 & y_rot_2) & 0x7F;
        ((tmp >> 6) | (tmp << 1)) & 0x7F
    }

    fn instantiate_lookup1() -> Vec<u16> {
        (0..=u16::MAX)
            .map(|i| {
                let lo = (i >> 8) as u8;
                let hi = i as u8;
                ((Self::s_box(lo) as u16) << 8) | Self::s_box(hi) as u16
            })
            .collect()
    }

    fn instantiate_lookup2() -> Vec<u16> {
        (0..(1 << 15))
            .map(|i| {
                let lo = (i >> 8) as u8;
                let hi: u8 = i as u8;
                ((Self::final_s_box(lo) as u16) << 8) | Self::s_box(hi) as u16
            })
            .collect()
    }

    fn random_field_element(shake: &mut Shake128Reader) -> F {
        let val = loop {
            let mut rnd = [0u8; 4];
            shake.read(&mut rnd);
            let res = u32::from_le_bytes(rnd);
            if res < F::ORDER_U32 {
                break res;
            }
        };
        F::from_canonical_u32(val)
    }

    fn init_shake() -> Shake128Reader {
        let mut shake = Shake128::default();
        shake.update("Monolith".as_bytes());
        shake.update(&[WIDTH as u8, NUM_ROUNDS as u8]);
        shake.update(&F::ORDER_U32.to_le_bytes());
        shake.update(&[8, 8, 8, 7]);
        shake.finalize_xof()
    }

    fn instantiate_round_constants() -> [[F; WIDTH]; NUM_ROUNDS] {
        let mut shake = Self::init_shake();

        [[F::ZERO; WIDTH]; NUM_ROUNDS].map(|arr| {
            arr.map(|_| Self::random_field_element(&mut shake))
        })
    }

    pub fn concrete(&self, state_u64: &mut [u64; T], round_constants: Option<&[F; T]>) {
        // MDS multiplication
        // optionally add round constants
    }

    // Performs the Bricks layer in place on `state_u64`, an array of field elements
    // represented as `u64`s.
    pub fn bricks(state: &mut [u64; WIDTH]) {
        // Feistel Type-3
        for (x_, x) in (state.to_owned()).iter().zip(state.iter_mut().skip(1)) {
            // Every time at bricks the input is technically a u32, so we tell the compiler
            let mut tmp_square = (x_ & 0xFFFFFFFF_u64) * (x_ & 0xFFFFFFFF_u64);
            tmp_square %= F::ORDER_U64; // F::reduce64(&mut tmp_square);
            *x = (*x & 0xFFFFFFFF_u64) + (tmp_square & 0xFFFFFFFF_u64);
        }
    }

    // Performs the Bar operation in place on `el`, an element of `F` represented as a `u32`.
    pub fn bar(&self, el: &mut u32) {
        debug_assert!(*el < F::ORDER_U32);

        unsafe {
            // get_unchecked here is safe because lookup table 1 contains 2^16 elements
            let low = *self.lookup1.get_unchecked(*el as u16 as usize);

            // get_unchecked here is safe because lookup table 2 contains 2^15 elements,
            // and el >> 16 < 2^15 (since el < F::ORDER_U32 < 2^31)
            let high = *self
                .lookup2
                .get_unchecked((*el >> 16) as u16 as usize);
            *el = (high as u32) << 16 | low as u32
        }
    }

    // Performs the Bars layer 
    pub fn bars(&self, state_u64: &mut [u64; WIDTH]) {
        state_u64
            .iter_mut()
            .take(Self::NUM_BARS)
            .for_each(|el| {
                let mut tmp = *el as u32;
                self.bar(&mut tmp);
                *el = tmp as u64
            });
    }

    pub fn permutation(&self, input: &[F; WIDTH]) -> [F; WIDTH] {
        let mut state = [0; WIDTH];
        for (out, inp) in state.iter_mut().zip(input.iter()) {
            *out = inp.as_canonical_u32() as u64;
        }

        debug_assert_eq!(
            self.round_constants.len(),
            NUM_ROUNDS - 1
        );
        self.concrete(&mut state, None);
        for rc in self.round_constants.iter().map(Some).chain(iter::once(None)) {
            self.bars(&mut state);
            Self::bricks(&mut state);
            self.concrete(&mut state, Some(rc));
        }

        // Convert back
        let mut state_f = [F::ZERO; WIDTH];
        for (out, inp) in state_f.iter_mut().zip(state.iter()) {
            *out = F::from_canonical_u32(*inp as u32);
        }
        state_f
    }
}
