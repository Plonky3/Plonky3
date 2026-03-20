/// Cheap duplication for prime-characteristic ring elements used in hot paths.
///
///  - [`Copy`] scalars and packed fields use a trivial copy
///  - non-[`Copy`] rings (e.g. symbolic expressions) should implement this with [`Clone::clone`]
///
/// It defaults to a trivial copy for types that are both [`Copy`] and [`Clone`].
///
/// # Usage
///
/// It is recommended to rely on the [`Dup`] trait in downstream implementations for any type
/// that is used in hot paths, such as trace cells in constraint evaluation for instance.
pub trait Dup: Clone {
    fn dup(&self) -> Self;
}

impl<T: Copy + Clone> Dup for T {
    #[inline(always)]
    fn dup(&self) -> Self {
        *self
    }
}
