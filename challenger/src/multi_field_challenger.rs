use alloc::vec;
use alloc::vec::Vec;
use alloc::string::String;

use p3_field::{reduce_64, split_64, ExtensionField, Field, PrimeField, PrimeField64};
use p3_symmetric::{CryptographicPermutation, Hash};

use crate::{CanObserve, CanSample, CanSampleBits, FieldChallenger};

#[derive(Clone)]
pub struct MultiFieldChallenger<F, PF, P, const WIDTH: usize>
where
    F: Clone,
    PF: Clone,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    sponge_state: [PF; WIDTH],
    input_buffer: Vec<F>,
    output_buffer: Vec<F>,
    permutation: P,
    num_f_elms: usize,
}

impl<F, PF, P, const WIDTH: usize> MultiFieldChallenger<F, PF, P, WIDTH>
where
    F: PrimeField64,
    PF: Field,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    pub fn new(permutation: P) -> Result<Self, String>
    {
        if F::order() >= PF::order() {
            return Err(String::from("F::order() must be less than PF::order()"));
        }
        let num_f_elms = PF::bits() / <F as Field>::bits();
        Ok(Self {
            sponge_state: [PF::default(); WIDTH],
            input_buffer: vec![],
            output_buffer: vec![],
            permutation,
            num_f_elms,
        })
    }
}

impl<F, PF, P, const WIDTH: usize> MultiFieldChallenger<F, PF, P, WIDTH>
where
    F: PrimeField64,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn duplexing(&mut self) {
        assert!(self.input_buffer.len() <= self.num_f_elms * WIDTH);

        for (i, f_chunk) in self.input_buffer.chunks(self.num_f_elms).enumerate() {
            self.sponge_state[i] = reduce_64(f_chunk);
        }
        self.input_buffer.clear();

        // Apply the permutation.
        self.permutation.permute_mut(&mut self.sponge_state);

        self.output_buffer.clear();
        for &pf_val in self.sponge_state.iter() {
            let mut f_vals = split_64(pf_val);
            f_vals.resize(self.num_f_elms, F::zero());

            for f_val in f_vals {
                self.output_buffer.push(f_val);
            }
        }
    }
}

impl<F, PF, P, const WIDTH: usize> FieldChallenger<F> for MultiFieldChallenger<F, PF, P, WIDTH>
where
    F: PrimeField64,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
}

impl<F, PF, P, const WIDTH: usize> CanObserve<F> for MultiFieldChallenger<F, PF, P, WIDTH>
where
    F: PrimeField64,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn observe(&mut self, value: F) {
        // Any buffered output is now invalid.
        self.output_buffer.clear();

        self.input_buffer.push(value);

        if self.input_buffer.len() == self.num_f_elms * WIDTH {
            self.duplexing();
        }
    }
}

impl<F, PF, const N: usize, P, const WIDTH: usize> CanObserve<[F; N]> for MultiFieldChallenger<F, PF, P, WIDTH>
where
    F: PrimeField64,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn observe(&mut self, values: [F; N]) {
        for value in values {
            self.observe(value);
        }
    }
}

impl<F, PF, const N: usize, P, const WIDTH: usize> CanObserve<Hash<F, PF, N>>
    for MultiFieldChallenger<F, PF, P, WIDTH>
where
    F: PrimeField64,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn observe(&mut self, values: Hash<F, PF, N>) {
        for pf_val in values {
            let mut f_vals = split_64(pf_val);
            f_vals.resize(self.num_f_elms, F::zero());

            for f_val in f_vals {
                self.observe(f_val);
            }
        }
    }
}

// for TrivialPcs
impl<F, PF, P, const WIDTH: usize> CanObserve<Vec<Vec<F>>> for MultiFieldChallenger<F, PF, P, WIDTH>
where
    F: PrimeField64,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn observe(&mut self, valuess: Vec<Vec<F>>) {
        for values in valuess {
            for value in values {
                self.observe(value);
            }
        }
    }
}

impl<F, EF, PF, P, const WIDTH: usize> CanSample<EF> for MultiFieldChallenger<F, PF, P, WIDTH>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn sample(&mut self) -> EF {
        EF::from_base_fn(|_| {
            // If we have buffered inputs, we must perform a duplexing so that the challenge will
            // reflect them. Or if we've run out of outputs, we must perform a duplexing to get more.
            if !self.input_buffer.is_empty() || self.output_buffer.is_empty() {
                self.duplexing();
            }

            self.output_buffer
                .pop()
                .expect("Output buffer should be non-empty")
        })
    }
}

impl<F, PF, P, const WIDTH: usize> CanSampleBits<usize> for MultiFieldChallenger<F, PF, P, WIDTH>
where
    F: PrimeField64,
    PF: PrimeField,
    P: CryptographicPermutation<[PF; WIDTH]>
{
    fn sample_bits(&mut self, bits: usize) -> usize {
        debug_assert!(bits < (usize::BITS as usize));
        debug_assert!((1 << bits) < F::ORDER_U64);
        let rand_f: F = self.sample();
        let rand_usize = rand_f.as_canonical_u64() as usize;
        rand_usize & ((1 << bits) - 1)
    }
}
