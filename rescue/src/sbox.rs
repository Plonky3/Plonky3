use core::marker::PhantomData;

use p3_field::{AbstractField, PrimeField, PrimeField64};

use crate::util::get_inverse;

pub trait SboxLayers<AF, const WIDTH: usize>: Clone + Sync
where
    AF: AbstractField,
    AF::F: PrimeField,
{
    fn sbox_layer(&self, state: &mut [AF; WIDTH]);

    fn inverse_sbox_layer(&self, state: &mut [AF; WIDTH]);
}

#[derive(Copy, Clone, Debug)]
pub struct BasicSboxLayer<F: PrimeField> {
    alpha: u64,
    alpha_inv: u64,
    _phantom: PhantomData<F>,
}

impl<F: PrimeField> BasicSboxLayer<F> {
    pub const fn new(alpha: u64, alpha_inv: u64) -> Self {
        Self {
            alpha,
            alpha_inv,
            _phantom: PhantomData,
        }
    }

    pub fn for_alpha(alpha: u64) -> Self
    where
        F: PrimeField64,
    {
        Self::new(alpha, get_inverse::<F>(alpha))
    }
}

impl<AF, const WIDTH: usize> SboxLayers<AF, WIDTH> for BasicSboxLayer<AF::F>
where
    AF: AbstractField,
    AF::F: PrimeField,
{
    fn sbox_layer(&self, state: &mut [AF; WIDTH]) {
        for x in state.iter_mut() {
            *x = x.exp_u64(self.alpha);
        }
    }

    fn inverse_sbox_layer(&self, state: &mut [AF; WIDTH]) {
        for x in state.iter_mut() {
            *x = x.exp_u64(self.alpha_inv);
        }
    }
}
