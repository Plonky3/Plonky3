use core::{marker::PhantomData, mem, ops::Mul, slice};

use crate::AbstractField;

pub trait Algebra<Base, const D: usize> {
    fn mul(l: [Base; D], r: [Base; D]) -> [Base; D];
}

#[repr(transparent)]
pub struct Ext<Base, const D: usize, A>([Base; D], PhantomData<A>);

unsafe trait CastableSubtower<Sup>: Sized {
    const D: usize;
}

impl<Base, const D: usize, A, Rhs> Mul<Rhs> for Ext<Base, D, A>
where
    A: Algebra<Base, D>,
    Rhs: CastableSubtower<Ext<Base, D, A>>,
    Rhs: AbstractField,
{
    type Output = Self;
    fn mul(mut self, rhs: Rhs) -> Self::Output {
        match Rhs::D {
            1 => {
                // same
                // cast rhs to me
                let rhs = unsafe { mem::transmute_copy::<Rhs, Self>(&rhs) };
                Self(A::mul(self.0, rhs.0), PhantomData)
            }
            d => {
                // smaller
                // cast me to array of rhs
                let s: &mut [Rhs] =
                    unsafe { slice::from_raw_parts_mut(self.0.as_mut_ptr() as *mut Rhs, d) };
                for lhs in s {
                    *lhs *= rhs.clone();
                }
                self
            }
        }
    }
}
