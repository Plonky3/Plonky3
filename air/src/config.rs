use crate::{AirTypes, ConstraintConsumer};
use core::marker::PhantomData;

pub trait AirConfig {
    // type Var: Copy;
    // type Exp;
    type Types: AirTypes; //<Var = Self::Var, Exp = Self::Exp>;

    // type Window: for<'a> AirWindow<'a, <<Self as AirConfig>::Types as AirTypes>::Var>;

    type CC: ConstraintConsumer<Self::Types>;

    const NUM_ROUNDS: usize;

    /// Number of rows in a constraint evaluation window.
    const WINDOW_SIZE: usize;
}

pub struct GenericAirConfig<Types, CC, const NUM_ROUNDS: usize, const WINDOW_SIZE: usize>
where
    Types: AirTypes,
    CC: ConstraintConsumer<Types>,
{
    _phantom_types: PhantomData<Types>,
    _phantom_cc: PhantomData<CC>,
}

impl<Types, CC, const NUM_ROUNDS: usize, const WINDOW_SIZE: usize> AirConfig
    for GenericAirConfig<Types, CC, NUM_ROUNDS, WINDOW_SIZE>
where
    Types: AirTypes,
    CC: ConstraintConsumer<Types>,
{
    // type Var = Types::Var;
    // type Exp = Types::Exp;
    type Types = Types;
    type CC = CC;
    const NUM_ROUNDS: usize = NUM_ROUNDS;
    const WINDOW_SIZE: usize = WINDOW_SIZE;
}

// trait StarkConfig {
//     const NUM_ROUNDS: usize;
//     const WINDOW_SIZE: usize;
// }
