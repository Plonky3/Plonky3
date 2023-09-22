use p3_field::{PrimeField, PrimeField64};

// use crate::util::get_inverse;

pub trait InverseSboxLayer<F: PrimeField, const WIDTH: usize, const ALPHA: u64>: Clone {
    fn inverse_sbox_layer(&self, state: &mut [F; WIDTH], alpha_inv: &u64);
}

#[derive(Copy, Clone, Default)]
pub struct BasicInverseSboxLayer;

impl<F: PrimeField64, const WIDTH: usize, const ALPHA: u64> InverseSboxLayer<F, WIDTH, ALPHA>
    for BasicInverseSboxLayer
{
    fn inverse_sbox_layer(&self, state: &mut [F; WIDTH], alpha_inv: &u64) {
        for x in state {
            *x = x.exp_u64(*alpha_inv);
        }
    }
}
