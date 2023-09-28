use std::marker::PhantomData;

use p3_field::{AbstractField, PrimeField, PrimeField64};

use crate::util::get_inverse;

pub trait SboxLayers<AF, const WIDTH: usize>: Clone
where
    AF: AbstractField,
    AF::F: PrimeField,
{
    fn sbox_layer(&self, state: &mut [AF; WIDTH]);

    fn inverse_sbox_layer(&self, state: &mut [AF; WIDTH]);
}

#[derive(Copy, Clone, Default)]
pub struct BasicSboxLayer<AF: AbstractField> {
    alpha: u64,
    alpha_inv: u64,
    _phantom_f: PhantomData<AF>,
}

impl<AF> BasicSboxLayer<AF>
where
    AF: AbstractField,
    AF::F: PrimeField,
{
    pub fn new(alpha: u64, alpha_inv: u64) -> Self {
        Self {
            alpha,
            alpha_inv,
            _phantom_f: PhantomData,
        }
    }

    pub fn for_alpha(alpha: u64) -> Self
    where
        AF::F: PrimeField64,
    {
        Self::new(alpha, get_inverse::<AF::F>(alpha))
    }
}

impl<AF, const WIDTH: usize> SboxLayers<AF, WIDTH> for BasicSboxLayer<AF>
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
