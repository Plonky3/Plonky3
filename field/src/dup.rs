/// Cheap duplication for prime-characteristic ring elements used in hot paths.
///
///  - [`Copy`] scalars and packed fields use a trivial copy
///  - non-[`Copy`] rings (e.g. symbolic expressions) should implement this with [`Clone::clone`]
///
/// It defaults to a trivial copy for types that are both [`Copy`] and [`Clone`].
pub trait Dup: Clone {
    fn dup(&self) -> Self;
}

impl<T: Copy + Clone> Dup for T {
    #[inline(always)]
    fn dup(&self) -> Self {
        *self
    }
}
