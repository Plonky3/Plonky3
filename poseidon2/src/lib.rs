//! The Poseidon2 permutation.
//!
//! Reference: https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2.rs

#![no_std]

extern crate alloc;

use alloc::borrow::ToOwned;
use alloc::vec::Vec;

use p3_field::Field;
use p3_mds::MdsPermutation;
use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;

/// The Poseidon2 permutation.
#[derive(Clone)]
pub struct Poseidon2<F, Mds, const WIDTH: usize, const D: usize>
where
    F: Field,
    Mds: MdsPermutation<F, WIDTH>,
{
    rounds_f: usize,
    rounds_p: usize,
    mat_internal_diag_m_1: [F; WIDTH],
    round_constants: Vec<[F; WIDTH]>,
    #[allow(dead_code)]
    mds: Mds,
}

impl<F, Mds, const WIDTH: usize, const D: usize> Poseidon2<F, Mds, WIDTH, D>
where
    F: Field,
    Mds: MdsPermutation<F, WIDTH>,
{
    /// Create a new Poseidon2 configuration.
    pub fn new(
        rounds_f: usize,
        rounds_p: usize,
        mat_internal_diag_m_1: [F; WIDTH],
        round_constants: Vec<[F; WIDTH]>,
        mds: Mds,
    ) -> Self {
        Self {
            rounds_f,
            rounds_p,
            mat_internal_diag_m_1,
            round_constants,
            mds,
        }
    }

    pub fn new_from_rng<R: Rng>(rounds_f: usize, rounds_p: usize, mds: Mds, rng: &mut R) -> Self
    where
        Standard: Distribution<F>,
    {
        let mut mat_internal_diag_m_1 = [F::ZERO; WIDTH];
        for i in 0..WIDTH {
            mat_internal_diag_m_1[i] = rng.sample(Standard);
        }

        let mut round_constants = Vec::new();
        for _ in 0..rounds_f + rounds_p {
            let mut round_constant = [F::ZERO; WIDTH];
            for j in 0..WIDTH {
                round_constant[j] = rng.sample(Standard);
            }
            round_constants.push(round_constant);
        }

        Self {
            rounds_f,
            rounds_p,
            mat_internal_diag_m_1,
            round_constants,
            mds,
        }
    }

    #[inline]
    fn add_rc(&self, state: &[F], rc: &[F]) -> [F; WIDTH] {
        let mut result = [F::ZERO; WIDTH];
        for i in 0..WIDTH {
            result[i] = state[i] + rc[i];
        }
        result
    }

    #[inline]
    fn sbox_p(&self, input: &F) -> F {
        let input = *input;
        let mut input2 = input;
        input2 *= input2;

        match D {
            3 => {
                let mut out = input2;
                out *= input;
                out
            }
            5 => {
                let mut out = input2;
                out *= out;
                out *= input;
                out
            }
            7 => {
                let mut out = input2;
                out *= out;
                out *= input2;
                out *= input;
                out
            }
            _ => panic!("D not supported"),
        }
    }

    #[inline]
    fn sbox(&self, state: &[F]) -> [F; WIDTH] {
        let mut result = [F::ZERO; WIDTH];
        for i in 0..WIDTH {
            result[i] = self.sbox_p(&state[i]);
        }
        result
    }

    #[inline]
    fn matmul_m4(&self, state: &mut [F; WIDTH]) {
        let t = WIDTH;
        let t4 = t / 4;
        for i in 0..t4 {
            let idx = i * 4;

            let mut t0 = state[idx];
            t0 += state[idx + 1];

            let mut t1 = state[idx + 2];
            t1 += state[idx + 3];

            let mut t2 = state[idx + 1];
            t2 += t2;
            t2 += t1;

            let mut t3 = state[idx + 3];
            t3 += t3;
            t3 += t0;

            let mut t4 = t1;
            t4 += t4;
            t4 += t4;
            t4 += t3;

            let mut t5 = t0;
            t5 += t5;
            t5 += t5;
            t5 += t2;

            let mut t6 = t3;
            t6 += t5;

            let mut t7 = t2;
            t7 += t4;

            state[idx] = t6;
            state[idx + 1] = t5;
            state[idx + 2] = t7;
            state[idx + 3] = t4;
        }
    }

    fn matmul_external(&self, state: &mut [F; WIDTH]) {
        match WIDTH {
            2 => {
                let sum = state[0] + state[1];
                state[0] += sum;
                state[1] += sum;
            }
            3 => {
                let sum = state[0] + state[1] + state[2];
                state[0] += sum;
                state[1] += sum;
                state[2] += sum;
            }
            4 => {
                self.matmul_m4(state);
            }
            8 | 12 | 16 | 20 | 24 => {
                self.matmul_m4(state);

                let t = WIDTH;
                let t4 = t / 4;
                let mut stored = [F::ZERO; 4];
                for l in 0..4 {
                    stored[l] = state[l];
                    for j in 1..t4 {
                        stored[l] += state[4 * j + l];
                    }
                }
                for i in 0..WIDTH {
                    state[i] += stored[i % 4];
                }
            }
            _ => panic!("WIDTH not supported"),
        }
    }

    fn matmul_internal(&self, state: &mut [F; WIDTH], mat_internal_diag_m_1: &[F]) {
        match WIDTH {
            2 => {
                let mut sum = state[0];
                sum += state[1];
                state[0] += sum;
                state[1] += state[1];
                state[1] += sum;
            }
            3 => {
                let mut sum = state[0];
                sum += state[1];
                sum += state[2];
                state[0] += sum;
                state[1] += sum;
                state[2] += state[2];
                state[2] += sum;
            }
            4 | 8 | 12 | 16 | 20 | 24 => {
                let mut sum = state[0];
                for i in 1..WIDTH {
                    sum += state[i];
                }
                for i in 0..WIDTH {
                    state[i] *= mat_internal_diag_m_1[i];
                    state[i] += sum;
                }
            }
            _ => panic!("WIDTH not supported"),
        }
    }
}

impl<F, Mds, const WIDTH: usize, const D: usize> CryptographicPermutation<[F; WIDTH]>
    for Poseidon2<F, Mds, WIDTH, D>
where
    F: Field,
    Mds: MdsPermutation<F, WIDTH>,
{
    fn permute(&self, state: [F; WIDTH]) -> [F; WIDTH] {
        let mut state = state.to_owned();

        // self.mds.permute_mut(&mut state);
        self.matmul_external(&mut state);

        let rounds = self.rounds_f + self.rounds_p;
        let rounds_f_beggining = self.rounds_f / 2;
        for r in 0..rounds_f_beggining {
            state = self.add_rc(&state, &self.round_constants[r]);
            state = self.sbox(&state);
            // self.mds.permute_mut(&mut state);
            self.matmul_external(&mut state);
        }

        let p_end = rounds_f_beggining + self.rounds_p;
        for r in self.rounds_f..p_end {
            state[0] += self.round_constants[r][0];
            state[0] = self.sbox_p(&state[0]);
            // self.mds.permute_mut(&mut state);
            self.matmul_internal(&mut state, &self.mat_internal_diag_m_1);
        }

        for r in p_end..rounds {
            state = self.add_rc(&state, &self.round_constants[r]);
            state = self.sbox(&state);
            // self.mds.permute_mut(&mut state);
            self.matmul_external(&mut state);
        }

        state
    }
}

impl<F: Field, Mds, const T: usize, const D: usize> ArrayPermutation<F, T>
    for Poseidon2<F, Mds, T, D>
where
    F: Field,
    Mds: MdsPermutation<F, T>,
{
}
