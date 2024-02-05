//! The Monolith-31 permutation.
//! With significant inspiration from https://extgit.iaik.tugraz.at/krypto/zkfriendlyhashzoo/

extern crate alloc;

use alloc::borrow::ToOwned;
use alloc::vec::Vec;

use p3_field::{AbstractField, PrimeField32};
use p3_mds::MdsPermutation;
use p3_mersenne_31::Mersenne31;
use sha3::digest::{ExtendableOutput, Update};
use sha3::{Shake128, Shake128Reader};

use crate::util::get_random_u32_le;

// The Monolith-31 permutation over Mersenne31.
// NUM_FULL_ROUNDS is the number of rounds - 1
// (used to avoid const generics because we need an array of length NUM_FULL_ROUNDS)
pub struct MonolithM31<Mds, const WIDTH: usize, const NUM_FULL_ROUNDS: usize>
where
    Mds: MdsPermutation<Mersenne31, WIDTH>,
{
    pub round_constants: [[Mersenne31; WIDTH]; NUM_FULL_ROUNDS],
    pub lookup1: Vec<u16>,
    pub lookup2: Vec<u16>,
    pub mds: Mds,
}

impl<Mds, const WIDTH: usize, const NUM_FULL_ROUNDS: usize> MonolithM31<Mds, WIDTH, NUM_FULL_ROUNDS>
where
    Mds: MdsPermutation<Mersenne31, WIDTH>,
{
    pub const NUM_BARS: usize = 8;

    pub fn new(mds: Mds) -> Self {
        assert!(WIDTH != 16, "Use MonolithM31Width16 instead");

        assert!(WIDTH >= 8);
        assert!(WIDTH <= 24);
        assert_eq!(WIDTH % 4, 0);

        let round_constants = Self::instantiate_round_constants();
        let lookup1 = Self::instantiate_lookup1();
        let lookup2 = Self::instantiate_lookup2();

        Self {
            round_constants,
            lookup1,
            lookup2,
            mds,
        }
    }

    fn s_box(y: u8) -> u8 {
        let tmp = y ^ !y.rotate_left(1) & y.rotate_left(2) & y.rotate_left(3);
        tmp.rotate_left(1)
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
                let hi = (i >> 8) as u8;
                let lo = i as u8;
                ((Self::s_box(hi) as u16) << 8) | Self::s_box(lo) as u16
            })
            .collect()
    }

    fn instantiate_lookup2() -> Vec<u16> {
        (0..(1 << 15))
            .map(|i| {
                let hi = (i >> 8) as u8;
                let lo: u8 = i as u8;
                ((Self::final_s_box(hi) as u16) << 8) | Self::s_box(lo) as u16
            })
            .collect()
    }

    fn random_field_element(shake: &mut Shake128Reader) -> Mersenne31 {
        let mut val = get_random_u32_le(shake);
        while val >= Mersenne31::ORDER_U32 {
            val = get_random_u32_le(shake);
        }

        Mersenne31::from_canonical_u32(val)
    }

    fn init_shake() -> Shake128Reader {
        let num_rounds = (NUM_FULL_ROUNDS + 1) as u8;

        let mut shake = Shake128::default();
        shake.update("Monolith".as_bytes());
        shake.update(&[WIDTH as u8, num_rounds]);
        shake.update(&Mersenne31::ORDER_U32.to_le_bytes());
        shake.update(&[8, 8, 8, 7]);
        shake.finalize_xof()
    }

    fn instantiate_round_constants() -> [[Mersenne31; WIDTH]; NUM_FULL_ROUNDS] {
        let mut shake = Self::init_shake();

        [[Mersenne31::zero(); WIDTH]; NUM_FULL_ROUNDS]
            .map(|arr| arr.map(|_| Self::random_field_element(&mut shake)))
    }

    pub fn concrete(&self, state: &mut [Mersenne31; WIDTH]) {
        self.mds.permute_mut(state);
    }

    pub fn add_round_constants(
        &self,
        state: &mut [Mersenne31; WIDTH],
        round_constants: &[Mersenne31; WIDTH],
    ) {
        // TODO: vectorize?
        for (x, rc) in state.iter_mut().zip(round_constants) {
            *x += *rc;
        }
    }

    pub fn bricks(state: &mut [Mersenne31; WIDTH]) {
        // Feistel Type-3
        for (x, x_mut) in (state.to_owned()).iter().zip(state.iter_mut().skip(1)) {
            *x_mut += x.square();
        }
    }

    pub fn bar(&self, el: Mersenne31) -> Mersenne31 {
        let val = &mut el.as_canonical_u32();

        unsafe {
            // get_unchecked here is safe because lookup table 1 contains 2^16 elements
            let low = *self.lookup1.get_unchecked(*val as u16 as usize);

            // get_unchecked here is safe because lookup table 2 contains 2^15 elements,
            // and el >> 16 < 2^15 (since el < Mersenne31::ORDER_U32 < 2^31)
            let high = *self.lookup2.get_unchecked((*val >> 16) as u16 as usize);
            *val = (high as u32) << 16 | low as u32
        }

        Mersenne31::from_canonical_u32(*val)
    }

    pub fn bars(&self, state: &mut [Mersenne31; WIDTH]) {
        state
            .iter_mut()
            .take(Self::NUM_BARS)
            .for_each(|el| *el = self.bar(*el));
    }

    pub fn permutation(&self, state: &mut [Mersenne31; WIDTH]) {
        self.concrete(state);
        for rc in self.round_constants {
            self.bars(state);
            Self::bricks(state);
            self.concrete(state);
            self.add_round_constants(state, &rc);
        }
        self.bars(state);
        Self::bricks(state);
        self.concrete(state);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::{AbstractField, PrimeField32};
    use p3_mersenne_31::Mersenne31;
    use zkhash::fields::const_f31::ConstF31;
    use zkhash::fields::f31::Field32;
    use zkhash::monolith_31::monolith_31::Monolith31 as MonolithM31Reference;
    use zkhash::monolith_31::monolith_31_instances::MONOLITH_CONST31_24_PARAMS;

    use crate::monolith::MonolithM31;
    use crate::monolith_mds::MonolithMdsMatrixM31;

    #[test]
    fn test_monolith_31() {
        let mds = MonolithMdsMatrixM31::<6>;
        let monolith: MonolithM31<_, 24, 5> = MonolithM31::new(mds);

        let mut state: [Mersenne31; 24] = [Mersenne31::zero(); 24];
        for (_, inp) in state.iter_mut().enumerate() {
            *inp = rand::random::<Mersenne31>();
        }

        let state_reference: [ConstF31; 24] = state
            .iter()
            .map(|x| ConstF31::from_u32((*x).as_canonical_u32()))
            .collect::<Vec<ConstF31>>()
            .try_into()
            .unwrap();

        monolith.permutation(&mut state);
        let output_converted: Vec<ConstF31> = state
            .iter()
            .map(|x| ConstF31::from_u32((*x).as_canonical_u32()))
            .collect::<Vec<ConstF31>>();

        let monolith_reference = MonolithM31Reference::new(&MONOLITH_CONST31_24_PARAMS);
        let ref_output = monolith_reference.permutation(&state_reference);

        assert_eq!(output_converted, ref_output);
    }
}
