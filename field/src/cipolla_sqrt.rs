use core::fmt::Debug;
use num_bigint::BigUint;
use num_traits::One;
use num_integer::Integer;

use crate::Field;

#[derive(Clone, Copy, Debug)]
struct ChipollaExtension<F: Field> {
    real: F,
    imag: F,
}

impl <F: Field> ChipollaExtension<F> {
    fn new(real: F, imag: F) -> Self {
        Self { real, imag }
    }

    fn one() -> Self {
        Self::new(F::one(), F::zero())
    }  


    fn mul(&self, other: Self, nonresidue: F) -> Self {
        Self::new(self.real * other.real + nonresidue * self.imag * other.imag, self.real * other.imag + self.imag * other.real)
    }

    fn pow(&self, exp: BigUint, nonresidue: F) -> Self {
        let mut result = Self::one();
        let mut base = *self;
        let bits = exp.bits();

        for i in 0..bits{
            if exp.bit(i) {
                result = result.mul(base, nonresidue);
            }
            base = base.mul(base, nonresidue);
        }
        result
    }
}


pub (crate) fn cipolla_sqrt<F: Field>(n: F) -> Option<F> {
    if n.is_zero() || n.is_one() {
        return Some(n);
    }

    if !n.is_square() {
        return None;
    }

    let order = F::order();

    {
        let (d, m) = order.div_mod_floor(&BigUint::from(4u8));
        if m == BigUint::from(3u8) {
            return Some(n.pow(&(d + BigUint::one())));
        }
    }

    let g = F::generator();

    let mut a = F::one();
    let mut nonresidue = F::one() - n;

    while nonresidue.is_square() {
        a *= g;
        nonresidue = a.square() - n;
    }
        

    let chipolla_power = (&order+BigUint::one())/BigUint::from(2u8);
    let mut x = ChipollaExtension::new(a, F::one());

    x = x.pow(chipolla_power, nonresidue);

    Some(x.real)
    
}