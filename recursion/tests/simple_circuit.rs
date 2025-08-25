use std::marker::PhantomData;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::testing::TrivialPcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_recursion::air::alu::cols::{AddTable, SubTable};
use p3_recursion::air::asic::Asic;
use p3_recursion::circuit_builder::circuit_builder::{CircuitBuilder, CircuitError};
use p3_recursion::circuit_builder::gates::arith_gates::{AddGate, SubGate};
use p3_recursion::prover::prove;
use p3_recursion::verifier::verify;
use p3_uni_stark::StarkConfig;
use rand::SeedableRng;
use rand::rngs::SmallRng;

type Value = BabyBear;
type Challenge = BinomialExtensionField<Value, 4>;
type Perm = Poseidon2BabyBear<16>;
type Dft = Radix2DitParallel<Value>;
type Challenger = DuplexChallenger<Value, Perm, 16, 8>;
type Pcs = TrivialPcs<Value, Radix2DitParallel<Value>>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

#[test]
pub fn test_simple_circuit() -> Result<(), CircuitError> {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    let dft = Dft::default();
    let pcs = TrivialPcs {
        dft,
        log_n: 0,
        _phantom: PhantomData,
    };
    let challenger = Challenger::new(perm);
    let config = MyConfig::new(pcs, challenger);

    let mut builder = CircuitBuilder::new();

    let a = builder.new_wire();
    let b = builder.new_wire();
    let c = builder.new_wire();
    let d = builder.new_wire();
    let e = builder.new_wire();

    AddGate::add_to_circuit(&mut builder, a, b, c);
    SubGate::add_to_circuit(&mut builder, c, d, e);

    let asic = Asic {
        asic: vec![Box::new(AddTable {}), Box::new(SubTable {})],
    };

    builder.set_wire_value(a, Value::new(10))?;
    builder.set_wire_value(b, Value::new(5))?;
    builder.set_wire_value(d, Value::new(2))?;

    let all_events = builder.generate()?;

    let proof = prove(&config, asic, all_events);

    verify(&config, proof).unwrap();

    Ok(())
}
