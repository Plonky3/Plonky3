use crate::{AbstractExtensionAlgebra, AbstractField, Extension, Field};

pub type Complex<AF> = Extension<AF, ComplexAlgebra>;

/// A field for which `p = 3 (mod 4)`. Equivalently, `-1` is not a square,
/// so the complex extension can be defined `F[X]/(X^2+1)`.
pub trait ComplexExtendable: Field {
    const COMPLEX_GEN: [Self; 2];

    /// The two-adicity of `p+1`, the order of the circle group.
    const CIRCLE_TWO_ADICITY: usize;

    fn circle_two_adic_generator(bits: usize) -> [Self; 2];
}

#[derive(Debug)]
pub struct ComplexAlgebra;

impl<F: ComplexExtendable> AbstractExtensionAlgebra<F> for ComplexAlgebra {
    const D: usize = 2;
    type Repr<AF: AbstractField<F = F>> = [AF; 2];
    const GEN: Self::Repr<F> = F::COMPLEX_GEN;

    fn mul<AF: AbstractField<F = F>>(
        a: Extension<AF, Self>,
        b: Extension<AF, Self>,
    ) -> Extension<AF, Self> {
        Extension([
            a[0].clone() * b[0].clone() - a[1].clone() * b[1].clone(),
            a[0].clone() * b[1].clone() + a[1].clone() * b[0].clone(),
        ])
    }

    fn repeated_frobenius(a: Extension<F, Self>, count: usize) -> Extension<F, Self> {
        if count % 2 == 0 {
            a
        } else {
            a.conj()
        }
    }

    fn inverse(a: Extension<F, Self>) -> Extension<F, Self> {
        a.conj() / a.norm()
    }
}

impl<AF: AbstractField> Extension<AF, ComplexAlgebra>
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
        self[0].clone()
    }
    pub fn conj(mut self) -> Self {
        self[1] = -self[1].clone();
        self
    }
    pub fn norm(&self) -> AF {
        self.real().square() + self.imag().square()
    }
}
