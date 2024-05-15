use core::marker::PhantomData;

use crate::{
    AbstractExtension, AbstractExtensionAlgebra, AbstractField, Extension, Field, HasBase,
};

pub type Complex<AF> = AbstractExtension<AF, ComplexAlgebra<<AF as AbstractField>::F>>;

/// A field for which `p = 3 (mod 4)`. Equivalently, `-1` is not a square,
/// so the complex extension can be defined `F[X]/(X^2+1)`.
pub trait ComplexExtendable: Field {
    const COMPLEX_GEN: [Self; 2];

    /// The two-adicity of `p+1`, the order of the circle group.
    const CIRCLE_TWO_ADICITY: usize;

    fn circle_two_adic_generator(bits: usize) -> Complex<Self>;
}

#[derive(Debug)]
pub struct ComplexAlgebra<F>(PhantomData<F>);

impl<F: ComplexExtendable> HasBase for ComplexAlgebra<F> {
    type Base = F;
}

impl<F: ComplexExtendable> AbstractExtensionAlgebra for ComplexAlgebra<F> {
    const D: usize = 2;
    type Repr<AF: AbstractField<F = F>> = [AF; 2];
    const GEN: Self::Repr<F> = F::COMPLEX_GEN;

    fn mul<AF: AbstractField<F = F>>(
        a: AbstractExtension<AF, Self>,
        b: AbstractExtension<AF, Self>,
    ) -> AbstractExtension<AF, Self> {
        AbstractExtension([
            a[0].clone() * b[0].clone() - a[1].clone() * b[1].clone(),
            a[0].clone() * b[1].clone() + a[1].clone() * b[0].clone(),
        ])
    }

    fn square<AF: AbstractField<F = Self::Base>>(
        a: AbstractExtension<AF, Self>,
    ) -> AbstractExtension<AF, Self> {
        AbstractExtension([
            a[0].clone().square() - a[1].clone().square(),
            a[0].clone() * a[1].clone().double(),
        ])
    }

    fn repeated_frobenius<AF: AbstractField<F = F>>(
        a: AbstractExtension<AF, Self>,
        count: usize,
    ) -> AbstractExtension<AF, Self> {
        if count % 2 == 0 {
            a
        } else {
            a.conj()
        }
    }

    fn inverse(a: Extension<Self>) -> Extension<Self> {
        a.conj() * a.norm().inverse()
    }
}

impl<AF: AbstractField> Complex<AF>
where
    AF::F: ComplexExtendable,
{
    pub fn new(real: AF, imag: AF) -> Self {
        Self([real, imag])
    }
    pub fn real(&self) -> AF {
        self[0].clone()
    }
    pub fn imag(&self) -> AF {
        self[1].clone()
    }
    pub fn conj(mut self) -> Self {
        self[1] = -self[1].clone();
        self
    }
    pub fn norm(&self) -> AF {
        self.real().square() + self.imag().square()
    }
}

#[macro_export]
macro_rules! generate_circle_gens {
    ($gen:expr, $two_adicity:expr) => {{
        let mut gens: [_; $two_adicity] = unsafe { core::mem::zeroed() };
        let mut i = 0;
        while i < $two_adicity {
            gens[i] = $gen;
            i += 1;
        }
        gens
    }};
}
