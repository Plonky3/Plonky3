use core::marker::PhantomData;

use p3_field::{FieldAlgebra, PrimeField, PrimeField64};

use crate::util::get_inverse;

pub trait SboxLayers<FA, const WIDTH: usize>: Clone + Sync
where
    FA: FieldAlgebra,
    FA::F: PrimeField,
{
    fn sbox_layer(&self, state: &mut [FA; WIDTH]);

    fn inverse_sbox_layer(&self, state: &mut [FA; WIDTH]);
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

impl<FA, const WIDTH: usize> SboxLayers<FA, WIDTH> for BasicSboxLayer<FA::F>
where
    FA: FieldAlgebra,
    FA::F: PrimeField,
{
    fn sbox_layer(&self, state: &mut [FA; WIDTH]) {
        for x in state.iter_mut() {
            *x = x.exp_u64(self.alpha);
        }
    }

    fn inverse_sbox_layer(&self, state: &mut [FA; WIDTH]) {
        for x in state.iter_mut() {
            *x = x.exp_u64(self.alpha_inv);
        }
    }
}
