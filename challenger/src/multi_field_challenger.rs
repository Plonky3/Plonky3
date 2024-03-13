use alloc::vec;
use alloc::vec::Vec;
use alloc::string::String;

use num_traits::identities::Zero;

use p3_field::{ExtensionField, Field, PrimeField, PrimeField32};
use p3_symmetric::Hash;

use crate::{CanObserve, CanSample, CanSampleBits, FieldChallenger};

#[derive(Clone)]
pub struct MultiFieldChallenger<F, PF, const WIDTH: usize, Inner>
where
    Inner: Clone
{
    inner: Inner,
    input_buffer: Vec<F>,
    output_buffer: Vec<F>,
    num_f_elms: usize,
    alpha: PF,
}

impl<F, PF, const WIDTH: usize, Inner> MultiFieldChallenger<F, PF, WIDTH, Inner>
where
    F: PrimeField32,
    PF: Field,
    Inner: Clone,
{
    pub fn new(inner: Inner) -> Result<Self, String>
    {
        if F::order() >= PF::order() {
            return Err(String::from("F::order() must be less than PF::order()"));
        }
        let num_f_elms = (PF::bits() + <F as Field>::bits() - 1) / <F as Field>::bits();
        Ok(Self {
            inner,
            input_buffer: vec![],
            output_buffer: vec![],
            num_f_elms,
            alpha: PF::from_canonical_u32(F::ORDER_U32),
        })
    }
}

impl<F, PF, const WIDTH: usize, Inner> MultiFieldChallenger<F, PF, WIDTH, Inner>
where
    F: PrimeField32,
    PF: Field + Clone,
    Inner: CanObserve<PF> + Clone,
{
    fn inner_observe(&mut self) {
        assert!(self.input_buffer.len() <= self.num_f_elms);

        let mut sum = PF::zero();
        for &term in self.input_buffer.iter().rev() {
            sum = sum * self.alpha + PF::from_canonical_u32(term.as_canonical_u32());
        }        

        self.inner.observe(sum);
        self.input_buffer.clear();
    }
}

impl<F, PF, const WIDTH: usize, Inner> FieldChallenger<F> for MultiFieldChallenger<F, PF, WIDTH, Inner>
where
    F: PrimeField32,
    PF: PrimeField + Clone + Sync,
    Inner: CanObserve<PF> + CanSample<PF> + Clone + Sync,
{
}

impl<F, PF, const WIDTH: usize, Inner> CanObserve<F> for MultiFieldChallenger<F, PF, WIDTH, Inner>
where
    F: PrimeField32,
    PF: Field + Clone,
    Inner: CanObserve<PF> + Clone,
{
    fn observe(&mut self, value: F) {
        // Any buffered output is now invalid.
        self.output_buffer.clear();

        self.input_buffer.push(value);

        if self.input_buffer.len() == self.num_f_elms {
            self.inner_observe();
        }
    }
}

impl<F, PF, const N: usize, const WIDTH: usize, Inner> CanObserve<[F; N]> for MultiFieldChallenger<F, PF, WIDTH, Inner>
where
    F: PrimeField32,
    PF: Field + Clone,
    Inner: CanObserve<PF> + Clone,
{
    fn observe(&mut self, values: [F; N]) {
        for value in values {
            self.observe(value);
        }
    }
}

impl<F, PF, const N: usize, const WIDTH: usize, Inner> CanObserve<Hash<F, PF, N>>
    for MultiFieldChallenger<F, PF, WIDTH, Inner>
where
    F: PrimeField32,
    PF: Field + Clone,
    Inner: CanObserve<PF> + Clone,
{
    fn observe(&mut self, values: Hash<F, PF, N>) {
        if !self.input_buffer.is_empty() {
            self.inner_observe();
        }

        for value in values {
            self.inner.observe(value);
        }
    }
}

// for TrivialPcs
impl<F, PF, const WIDTH: usize, Inner> CanObserve<Vec<Vec<F>>> for MultiFieldChallenger<F, PF, WIDTH, Inner>
where
    F: PrimeField32,
    PF: Field + Clone,
    Inner: CanObserve<PF> + Clone,
{
    fn observe(&mut self, valuess: Vec<Vec<F>>) {
        for values in valuess {
            for value in values {
                self.observe(value);
            }
        }
    }
}

impl<F, EF, PF, const WIDTH: usize, Inner> CanSample<EF> for MultiFieldChallenger<F, PF, WIDTH, Inner>
where
    F: PrimeField32,
    EF: ExtensionField<F>,
    PF: PrimeField + Clone,
    Inner: CanObserve<PF> + CanSample<PF> + Clone,
{
    fn sample(&mut self) -> EF {
        EF::from_base_fn(|_| {
            if !self.input_buffer.is_empty() {
                self.inner_observe();
            }

            if self.output_buffer.is_empty() {
                // Sample from the inner challenger
                let mut val = self.inner.sample().as_canonical_biguint();

                let alpha = self.alpha.as_canonical_biguint();

                for _ in 0..self.num_f_elms {
                    let rem = val.clone() % alpha.clone();
                    val /= alpha.clone();

                    if rem.is_zero() {
                        self.output_buffer.push(F::zero());
                    } else {
                        let digits = rem.to_u32_digits();
                        debug_assert!(digits.len() <= 1);
                        self.output_buffer.push(F::from_canonical_u32(rem.to_u32_digits()[0]));
                    }
                }
            }

            self.output_buffer
                .pop()
                .expect("Output buffer should be non-empty")
        })
    }
}

impl<F, PF, const WIDTH: usize, Inner> CanSampleBits<usize> for MultiFieldChallenger<F, PF, WIDTH, Inner>
where
    F: PrimeField32,
    PF: PrimeField + Clone,
    Inner: CanSample<PF> + CanObserve<PF> + Clone,
{
    fn sample_bits(&mut self, bits: usize) -> usize {
        debug_assert!(bits < (usize::BITS as usize));
        debug_assert!((1 << bits) < F::ORDER_U32);
        let rand_f: F = self.sample();
        let rand_usize = rand_f.as_canonical_u32() as usize;
        rand_usize & ((1 << bits) - 1)
    }
}
