/// The side of a multiset equality to which an entry belongs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BusDirection {
    /// Add the entry to the multiset being produced.
    Push,
    /// Add the entry to the multiset being consumed.
    Pull,
}

impl BusDirection {
    /// Both multiset sides in stable push-then-pull order.
    pub const ALL: [Self; 2] = [Self::Push, Self::Pull];

    /// Position of this direction in push-then-pull protocol arrays.
    #[must_use]
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Push => 0,
            Self::Pull => 1,
        }
    }
}
