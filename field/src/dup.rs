/// Light supertrait of `Clone` to help speed up hot paths for types implementing `Copy`.
///
///  - [`Copy`] types (e.g. scalars and packed fields) use a trivial copy
///  - non-[`Copy`] types (e.g. symbolic expressions) should implement this with [`Clone::clone`]
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
