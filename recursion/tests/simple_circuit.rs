use p3_baby_bear::BabyBear;
use p3_recursion::circuit_builder::circuit_builder::{CircuitBuilder, CircuitError};
use p3_recursion::circuit_builder::gates::arith_gates::{AddGate, SubGate};

type Val = BabyBear;

#[test]
pub fn test_simple_circuit() -> Result<(), CircuitError> {
    let mut builder = CircuitBuilder::<Val>::new();

    let a = builder.new_wire();
    let b = builder.new_wire();
    let c = builder.new_wire();
    let d = builder.new_wire();
    let e = builder.new_wire();

    AddGate::add_to_circuit(&mut builder, a, b, c);
    SubGate::add_to_circuit(&mut builder, c, d, e);

    builder.set_wire_value(a, Val::new(10))?;
    builder.set_wire_value(b, Val::new(5))?;
    builder.set_wire_value(d, Val::new(2))?;

    builder.generate()
}
