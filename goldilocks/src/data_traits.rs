use p3_field::TwoAdicField;

/// TwoAdicData contains constants needed to imply TwoAdicField for Goldilocks fields.
pub trait TwoAdicData: TwoAdicField {
    /// ArrayLike should usually be `&'static [Goldilocks]`.
    type ArrayLike: AsRef<[Self]> + Sized;

    /// A list of generators of 2-adic subgroups.
    /// The i'th element must be a 2^i root of unity and the i'th element squared must be the i-1'th element.
    const TWO_ADIC_GENERATORS: Self::ArrayLike;
}
