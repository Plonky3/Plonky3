use std::marker::PhantomData;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::testing::TrivialPcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_recursion::air::AluAir;
use p3_recursion::air::alu::air::FieldOperation;
use p3_recursion::air::alu::cols::{AddTable, SubTable};
use p3_recursion::air::asic::Asic;
use p3_recursion::circuit_builder::circuit_builder::{CircuitBuilder, CircuitError};
use p3_recursion::circuit_builder::gates::arith_gates::{AddGate, SubGate};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::SeedableRng;
use rand::rngs::SmallRng;

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

    let asic = Asic {
        asic: vec![Box::new(AddTable {}), Box::new(SubTable {})],
    };

    builder.set_wire_value(a, Val::new(10))?;
    builder.set_wire_value(b, Val::new(5))?;
    builder.set_wire_value(d, Val::new(2))?;

    let all_events = builder.generate()?;
    let traces = asic.generate_trace(&all_events);

    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2BabyBear<16>;
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    type Dft = Radix2DitParallel<Val>;
    let dft = Dft::default();

    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;

    type Pcs = TrivialPcs<Val, Radix2DitParallel<Val>>;
    let pcs = TrivialPcs {
        dft,
        log_n: 0,
        _phantom: PhantomData,
    };
    let challenger = Challenger::new(perm);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs, challenger);

    let add_air: AluAir<1> = AluAir {
        op: FieldOperation::Add,
    };
    let sub_air: AluAir<1> = AluAir {
        op: FieldOperation::Sub,
    };

    let proof_add = prove(&config, &add_air, traces[0].clone(), &vec![]);
    let serialized_proof = postcard::to_allocvec(&proof_add).expect("unable to serialize proof");
    tracing::debug!("serialized_proof len: {} bytes", serialized_proof.len());

    let deserialized_proof =
        postcard::from_bytes(&serialized_proof).expect("unable to deserialize proof");

    verify(&config, &add_air, &deserialized_proof, &vec![]).unwrap();

    let proof_sub = prove(&config, &sub_air, traces[1].clone(), &vec![]);
    let serialized_proof = postcard::to_allocvec(&proof_sub).expect(
        "unable to
    serialize proof",
    );
    tracing::debug!("serialized_proof len: {} bytes", serialized_proof.len());
    let deserialized_proof =
        postcard::from_bytes(&serialized_proof).expect("unable to deserialize proof");
    verify(&config, &sub_air, &deserialized_proof, &vec![]).unwrap();

    Ok(())
}
