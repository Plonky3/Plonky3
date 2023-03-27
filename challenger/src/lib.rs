#![no_std]

extern crate alloc;

pub mod duplex_challenger;
pub mod hash_challenger;

use hyperfield::field::{Field, FieldExtension};

pub trait Challenger<F: Field> {
    fn observe_element(&mut self, element: F);

    fn observe_extension_element<FE: FieldExtension<Base = F>>(&mut self, _element: FE) {
        todo!()
    }
}
