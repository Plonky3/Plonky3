use p3_field::Field;

use crate::circuit_builder::{CircuitBuilder, CircuitError, WireId};
use crate::gates::gate::Gate;

pub struct AddGate<F: Field> {
    inputs: Vec<WireId>,
    outputs: Vec<WireId>,
    _marker: std::marker::PhantomData<F>,
}

const BINOP_N_INPUTS: usize = 2;
const BINOP_N_OUTPUTS: usize = 1;

impl<F: Field> AddGate<F> {
    pub fn new(inputs: Vec<WireId>, outputs: Vec<WireId>) -> Self {
        assert!(inputs.len() == BINOP_N_INPUTS);
        assert!(outputs.len() == BINOP_N_OUTPUTS);

        AddGate {
            inputs,
            outputs,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn add_to_circuit(builder: &mut CircuitBuilder<F>, a: WireId, b: WireId, c: WireId) -> () {
        let gate = AddGate::new(vec![a, b], vec![c]);
        builder.add_gate(Box::new(gate));
    }
}

impl<F: Field> Gate<F> for AddGate<F> {
    fn n_inputs(&self) -> usize {
        BINOP_N_INPUTS
    }

    fn n_outputs(&self) -> usize {
        BINOP_N_OUTPUTS
    }

    fn generate(&self, builder: &mut CircuitBuilder<F>) -> Result<(), CircuitError> {
        self.check_shape(self.inputs.len(), self.outputs.len());

        let input1 = builder.get_wire_value(self.inputs[0])?;
        let input2 = builder.get_wire_value(self.inputs[1])?;

        if input1.is_none() || input2.is_none() {
            return Err(CircuitError::InputNotSet);
        }

        let res = input1.unwrap() + input2.unwrap();
        builder.set_wire_value(self.outputs[0], res)?;

        Ok(())
    }
}

pub struct SubGate<F: Field> {
    inputs: Vec<WireId>,
    outputs: Vec<WireId>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field> SubGate<F> {
    pub fn new(inputs: Vec<WireId>, outputs: Vec<WireId>) -> Self {
        assert!(inputs.len() == BINOP_N_INPUTS);
        assert!(outputs.len() == BINOP_N_OUTPUTS);

        SubGate {
            inputs,
            outputs,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn add_to_circuit(builder: &mut CircuitBuilder<F>, a: WireId, b: WireId, c: WireId) -> () {
        let gate = SubGate::new(vec![a, b], vec![c]);
        builder.add_gate(Box::new(gate));
    }
}

impl<F: Field> Gate<F> for SubGate<F> {
    fn n_inputs(&self) -> usize {
        2
    }

    fn n_outputs(&self) -> usize {
        1
    }

    fn generate(&self, builder: &mut CircuitBuilder<F>) -> Result<(), CircuitError> {
        self.check_shape(self.inputs.len(), self.outputs.len());

        let input1 = builder.get_wire_value(self.inputs[0])?;
        let input2 = builder.get_wire_value(self.inputs[1])?;

        if input1.is_none() || input2.is_none() {
            return Err(CircuitError::InputNotSet);
        }

        let res = input1.unwrap() - input2.unwrap();
        builder.set_wire_value(self.outputs[0], res)?;

        Ok(())
    }
}

pub struct MulGate<F: Field> {
    inputs: Vec<WireId>,
    outputs: Vec<WireId>,
    _marker: std::marker::PhantomData<F>,
}
impl<F: Field> MulGate<F> {
    pub fn new(inputs: Vec<WireId>, outputs: Vec<WireId>) -> Self {
        assert!(inputs.len() == BINOP_N_INPUTS);
        assert!(outputs.len() == BINOP_N_OUTPUTS);

        MulGate {
            inputs,
            outputs,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn add_to_circuit(builder: &mut CircuitBuilder<F>, a: WireId, b: WireId, c: WireId) -> () {
        let gate = MulGate::new(vec![a, b], vec![c]);
        builder.add_gate(Box::new(gate));
    }
}

impl<F: Field> Gate<F> for MulGate<F> {
    fn n_inputs(&self) -> usize {
        2
    }

    fn n_outputs(&self) -> usize {
        1
    }

    fn generate(&self, builder: &mut CircuitBuilder<F>) -> Result<(), CircuitError> {
        self.check_shape(self.inputs.len(), self.outputs.len());

        let input1 = builder.get_wire_value(self.inputs[0])?;
        let input2 = builder.get_wire_value(self.inputs[1])?;

        if input1.is_none() || input2.is_none() {
            return Err(CircuitError::InputNotSet);
        }

        let res = input1.unwrap() * input2.unwrap();
        builder.set_wire_value(self.outputs[0], res)?;

        Ok(())
    }
}
