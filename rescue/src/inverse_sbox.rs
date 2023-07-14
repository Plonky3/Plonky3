use p3_field::{PrimeField, PrimeField64};

use crate::util::get_inverse;

pub trait InverseSboxLayer<F: PrimeField, const WIDTH: usize, const ALPHA: u64> {
    fn inverse_sbox_layer(&self, state: &mut [F; WIDTH]);
}

pub struct BasicInverseSboxLayer;

impl<F: PrimeField64, const WIDTH: usize, const ALPHA: u64> InverseSboxLayer<F, WIDTH, ALPHA>
    for BasicInverseSboxLayer
{
    fn inverse_sbox_layer(&self, state: &mut [F; WIDTH]) {
        let alpha_inv = get_inverse::<F>(ALPHA);
        for x in state {
            *x = x.exp_u64(alpha_inv);
        }
    }
}
