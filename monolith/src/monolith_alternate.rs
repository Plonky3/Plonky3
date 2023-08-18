//! The Monolith permutation, and hash functions built from it.

extern crate alloc;

use core::{iter, u64};

use p3_field::PrimeField32;
use p3_symmetric::mds::MDSPermutation;
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::{Shake128, Shake128Reader};

use crate::monolith_mds::monolith_mds;

// The Monolith-31 permutation.
// Assumes that F is a 31-bit field (e.g. Mersenne31).
pub struct Monolith31Alternate<F: PrimeField32, const WIDTH: usize, const NUM_ROUNDS: usize> {
    pub round_constants: Vec<[u64; WIDTH]>,
    pub mds: Box<dyn MDSPermutation<F, WIDTH>>,
    pub lookup1: Vec<u16>,
    pub lookup2: Vec<u16>,
}

impl<F: PrimeField32, const WIDTH: usize, const NUM_ROUNDS: usize>
    Monolith31Alternate<F, WIDTH, NUM_ROUNDS>
{
    pub const NUM_BARS: usize = 8;

    pub fn new() -> Self {
        assert_eq!(F::bits(), 31);
        assert!(WIDTH >= 16);
        assert_eq!(WIDTH % 4, 0);

        let round_constants = Self::instantiate_round_constants();
        let lookup1 = Self::instantiate_lookup1();
        let lookup2 = Self::instantiate_lookup2();
        let mds = monolith_mds("Monolith", NUM_ROUNDS);

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

    fn random_field_element(shake: &mut Shake128Reader) -> u64 {
        let val = loop {
            let mut rnd = [0u8; 4];
            shake.read(&mut rnd);
            let res = u32::from_le_bytes(rnd);
            if res < F::ORDER_U32 {
                break res;
            }
        };
        val as u64
    }

    fn init_shake() -> Shake128Reader {
        let mut shake = Shake128::default();
        shake.update("Monolith".as_bytes());
        shake.update(&[WIDTH as u8, NUM_ROUNDS as u8]);
        shake.update(&F::ORDER_U32.to_le_bytes());
        shake.update(&[8, 8, 8, 7]);
        shake.finalize_xof()
    }

    fn instantiate_round_constants() -> Vec<[u64; WIDTH]> {
        let mut shake = Self::init_shake();

        vec![[F::ZERO; WIDTH]; NUM_ROUNDS - 1]
            .iter()
            .map(|arr| arr.map(|_| Self::random_field_element(&mut shake)))
            .collect()
    }

    pub fn concrete(&self, state: &mut [u64; WIDTH], round_constants: Option<&[u64; WIDTH]>) {
        let state_f = state.map(|el| {
            let reduced = el % F::ORDER_U64;
            F::from_canonical_u64(reduced)
        });
        let new_state_f = self.mds.permute(state_f);

        for ((x, el), c) in state
            .iter_mut()
            .zip(new_state_f.iter())
            .zip(round_constants.unwrap_or(&[0; WIDTH]).iter())
        {
            *x = (el.as_canonical_u64() + c) % F::ORDER_U64;
        }
    }

    // Performs the Bricks layer in place on `state`, an array of field elements
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
            let high = *self.lookup2.get_unchecked((*el >> 16) as u16 as usize);
            *el = (high as u32) << 16 | low as u32
        }
    }

    // Performs the Bars layer in place on `state`, an array of field elements represented as `u64`s.
    pub fn bars(&self, state: &mut [u64; WIDTH]) {
        state.iter_mut().take(Self::NUM_BARS).for_each(|el| {
            let mut tmp = *el as u32;
            self.bar(&mut tmp);
            *el = tmp as u64
        });
    }

    pub fn permutation(&self, state: &mut [F; WIDTH]) {
        let mut state_u64 = [0; WIDTH];
        for (out, inp) in state_u64.iter_mut().zip(state.iter()) {
            *out = inp.as_canonical_u32() as u64;
        }

        self.concrete(&mut state_u64, None);

        for rc in self
            .round_constants
            .iter()
            .map(Some)
            .chain(iter::once(None))
        {
            self.bars(&mut state_u64);
            Self::bricks(&mut state_u64);
            self.concrete(&mut state_u64, rc);
        }

        // Convert back
        for (out, inp) in state.iter_mut().zip(state_u64.iter()) {
            *out = {
                let reduced = *inp % F::ORDER_U64;
                F::from_canonical_u64(reduced)
            };
        }
    }
}

mod tests {
    use p3_field::AbstractField;
    use p3_mersenne_31::Mersenne31;

    use crate::monolith_alternate::Monolith31Alternate;

    #[test]
    fn test_monolith_31_alternate() {
        let monolith: Monolith31Alternate<Mersenne31, 16, 6> = Monolith31Alternate::new();

        let mut input: [Mersenne31; 16] = [Mersenne31::ZERO; 16];
        for (i, inp) in input.iter_mut().enumerate() {
            *inp = Mersenne31::from_canonical_usize(i);
        }
        monolith.permutation(&mut input);

        assert_eq!(input[0], Mersenne31::from_canonical_u64(609156607));
        assert_eq!(input[1], Mersenne31::from_canonical_u64(290107110));
        assert_eq!(input[2], Mersenne31::from_canonical_u64(1900746598));
        assert_eq!(input[3], Mersenne31::from_canonical_u64(1734707571));
        assert_eq!(input[4], Mersenne31::from_canonical_u64(2050994835));
        assert_eq!(input[5], Mersenne31::from_canonical_u64(1648553244));
        assert_eq!(input[6], Mersenne31::from_canonical_u64(1307647296));
        assert_eq!(input[7], Mersenne31::from_canonical_u64(1941164548));
        assert_eq!(input[8], Mersenne31::from_canonical_u64(1707113065));
        assert_eq!(input[9], Mersenne31::from_canonical_u64(1477714255));
        assert_eq!(input[10], Mersenne31::from_canonical_u64(1170160793));
        assert_eq!(input[11], Mersenne31::from_canonical_u64(93800695));
        assert_eq!(input[12], Mersenne31::from_canonical_u64(769879348));
        assert_eq!(input[13], Mersenne31::from_canonical_u64(375548503));
        assert_eq!(input[14], Mersenne31::from_canonical_u64(1989726444));
        assert_eq!(input[15], Mersenne31::from_canonical_u64(1349325635));
    }
}
