use core::{marker::PhantomData, ops::Mul};

use crate::{AbstractExtensionField, AbstractField};

pub struct Nil;
pub struct Cons<A, const D: usize, T>(PhantomData<(A, T)>);

pub trait ToArr<F> {
    type Arr;
}
impl<F> ToArr<F> for Nil {
    type Arr = F;
}
impl<F, A, const D: usize, T> ToArr<F> for Cons<A, D, T>
where
    T: ToArr<F>,
{
    type Arr = [T::Arr; D];
}

#[repr(transparent)]
pub struct Ext<F, Tower: ToArr<F>>(Tower::Arr);

trait Algebra<Base, const D: usize> {
    fn mul(l: [Base; D], r: [Base; D]) -> [Base; D];
}

type Rev<L> = <L as CanRev<Nil>>::Output;
trait CanRev<Acc> {
    type Output;
}
impl<Acc> CanRev<Acc> for Nil {
    type Output = Acc;
}
impl<Acc, A, const D: usize, T> CanRev<Acc> for Cons<A, D, T>
where
    T: CanRev<Cons<A, D, Acc>>,
{
    type Output = <T as CanRev<Cons<A, D, Acc>>>::Output;
}

trait CastableSubtowerRev<Sup> {
    const D: usize;
}
impl CastableSubtowerRev<Nil> for Nil {
    const D: usize = 1;
}
impl<A, const D: usize, T> CastableSubtowerRev<Cons<A, D, T>> for Nil
where
    Nil: CastableSubtowerRev<T>,
{
    const D: usize = D * Nil::D;
}
impl<A, const D: usize, T1, T2> CastableSubtowerRev<Cons<A, D, T1>> for Cons<A, D, T2>
where
    T2: CastableSubtowerRev<T1>,
{
    const D: usize = T2::D;
}

impl<F, Tower: ToArr<F>, Rhs> Mul<Rhs> for Ext<F, Tower> {
    type Output = Self;
    fn mul(self, rhs: Rhs) -> Self::Output {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct A;
    struct B;
    struct C;

    trait SameType<T> {}
    impl<T> SameType<T> for T {}

    fn same<T, U: SameType<T>>() {}

    #[rustfmt::skip]
    fn rev() {
        same::<
            Rev<Nil>,
                Nil,
        >();
        same::<
            Rev<Cons<A, 1, Nil>>,
                Cons<A, 1, Nil>,
        >();
        same::<
            Rev<Cons<A, 1, Cons<B, 2, Nil>>>,
                Cons<B, 2, Cons<A, 1, Nil>>,
        >();
        same::<
            Rev<Cons<A, 1, Cons<B, 2, Cons<C, 3, Nil>>>>,
                Cons<C, 3, Cons<B, 2, Cons<A, 1, Nil>>>,
        >();
    }
}
