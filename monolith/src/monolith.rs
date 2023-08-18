//! The Monolith permutation, and hash functions built from it.

extern crate alloc;

use core::iter;

use p3_field::PrimeField32;
use p3_symmetric::mds::MDSPermutation;
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::{Shake128, Shake128Reader};

use crate::monolith_mds::monolith_mds;

// The Monolith-31 permutation.
// Assumes that F is a 31-bit field (e.g. Mersenne31).
pub struct Monolith31<F: PrimeField32, const WIDTH: usize, const NUM_ROUNDS: usize> {
    // TODO: if possible, replace with [[F; WIDTH]; NUM_ROUNDS - 1]]
    pub round_constants: Vec<[F; WIDTH]>,
    pub mds: Box<dyn MDSPermutation<F, WIDTH>>,
    pub lookup1: Vec<u16>,
    pub lookup2: Vec<u16>,
}

impl<F: PrimeField32, const WIDTH: usize, const NUM_ROUNDS: usize>
    Monolith31<F, WIDTH, NUM_ROUNDS>
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

    fn instantiate_round_constants() -> Vec<[F; WIDTH]> {
        let mut shake = Self::init_shake();

        vec![[F::ZERO; WIDTH]; NUM_ROUNDS - 1]
            .iter()
            .map(|arr| arr.map(|_| Self::random_field_element(&mut shake)))
            .collect()
    }

    pub fn concrete(&self, state: &mut [F; WIDTH], round_constants: Option<&[F; WIDTH]>) {
        *state = self.mds.permute(*state);

        if let Some(round_constants) = round_constants {
            for (x, rc) in state.iter_mut().zip(round_constants.iter()) {
                *x += *rc;
            }
        }
    }

    pub fn bricks(state: &mut [F; WIDTH]) {
        // Feistel Type-3
        for (x, x_mut) in (state.to_owned()).iter().zip(state.iter_mut().skip(1)) {
            let x_mut_u64 = &mut x_mut.as_canonical_u64();
            let x_u64 = x.as_canonical_u64();
            // Every time at bricks the input is technically a u32, so we tell the compiler
            let mut tmp_square = (x_u64 & 0xFFFFFFFF_u64) * (x_u64 & 0xFFFFFFFF_u64);
            tmp_square %= F::ORDER_U64;
            *x_mut_u64 = (*x_mut_u64 & 0xFFFFFFFF_u64) + (tmp_square & 0xFFFFFFFF_u64);
            *x_mut_u64 %= F::ORDER_U64;
            *x_mut = F::from_canonical_u64(*x_mut_u64);
        }
    }

    pub fn bar(&self, el: F) -> F {
        let val = &mut el.as_canonical_u32();

        unsafe {
            // get_unchecked here is safe because lookup table 1 contains 2^16 elements
            let low = *self.lookup1.get_unchecked(*val as u16 as usize);

            // get_unchecked here is safe because lookup table 2 contains 2^15 elements,
            // and el >> 16 < 2^15 (since el < F::ORDER_U32 < 2^31)
            let high = *self.lookup2.get_unchecked((*val >> 16) as u16 as usize);
            *val = (high as u32) << 16 | low as u32
        }

        F::from_canonical_u32(*val)
    }

    pub fn bars(&self, state: &mut [F; WIDTH]) {
        state
            .iter_mut()
            .take(Self::NUM_BARS)
            .for_each(|el| *el = self.bar(*el));
    }

    pub fn permutation(&self, state: &mut [F; WIDTH]) {
        dbg!(state.clone());
        self.concrete(state, None);
        dbg!(state.clone());
        for rc in self
        .round_constants
        .iter()
        .map(Some)
        .chain(iter::once(None))
        {
            self.bars(state);
            Self::bricks(state);
            self.concrete(state, rc);
            dbg!(state.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_mersenne_31::Mersenne31;

    use crate::monolith::Monolith31;

    #[test]
    fn test_monolith_31() {
        let monolith: Monolith31<Mersenne31, 16, 6> = Monolith31::new();

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
