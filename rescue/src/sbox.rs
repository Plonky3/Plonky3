use std::marker::PhantomData;

use p3_field::{AbstractField, PrimeField, PrimeField64};

use crate::util::get_inverse;

pub trait SboxLayers<F: AbstractField, const WIDTH: usize>: Clone
where
    F::F: PrimeField,
{
    fn sbox_layer(&self, state: &mut [F; WIDTH]);

    fn inverse_sbox_layer(&self, state: &mut [F; WIDTH]);
}

#[derive(Copy, Clone, Default)]
pub struct BasicSboxLayer<F: AbstractField> {
    alpha: u64,
    alpha_inv: u64,
    _phantom_f: PhantomData<F>,
}

impl<F: AbstractField> BasicSboxLayer<F>
where
    F::F: PrimeField64,
{
    pub fn new(alpha: u64, alpha_inv: u64) -> Self {
        Self {
            alpha,
            alpha_inv,
            _phantom_f: PhantomData,
        }
    }

    pub fn for_alpha(alpha: u64) -> Self {
        Self::new(alpha, get_inverse::<F::F>(alpha))
    }
}

impl<F: AbstractField, const WIDTH: usize> SboxLayers<F, WIDTH> for BasicSboxLayer<F>
where
    F::F: PrimeField64,
{
    fn sbox_layer(&self, state: &mut [F; WIDTH]) {
        for x in state.iter_mut() {
            *x = x.exp_u64(self.alpha);
        }
    }

    fn inverse_sbox_layer(&self, state: &mut [F; WIDTH]) {
        for x in state.iter_mut() {
            *x = x.exp_u64(self.alpha_inv);
        }
    }
}
