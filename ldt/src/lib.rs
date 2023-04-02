//! Low-degree tests (LDTs).

#![no_std]

pub trait LDT {
    type Error;

    fn test() -> Result<(), Self::Error>;
}

// TODO: PCS from LDT.
