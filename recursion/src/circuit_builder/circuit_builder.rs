use p3_field::Field;

use crate::circuit_builder::gates::gate::Gate;

pub(crate) type WireId = usize;

pub struct CircuitBuilder<F: Field> {
    wires: Vec<Option<F>>,
    gate_instances: Vec<Box<dyn Gate<F>>>,
}

impl<F: Field> CircuitBuilder<F> {
    pub fn new() -> Self {
        CircuitBuilder {
            wires: Vec::new(),
            gate_instances: Vec::new(),
        }
    }

    pub fn new_wire(&mut self) -> WireId {
        self.wires.push(None);
        self.wires.len() - 1
    }

    pub fn add_gate(&mut self, gate: Box<dyn Gate<F>>) {
        self.gate_instances.push(gate);
    }

    pub fn wires(&mut self) -> &Vec<Option<F>> {
        &mut self.wires
    }

    pub fn get_wire_value(&self, id: WireId) -> Result<Option<F>, CircuitError> {
        if id >= self.wires.len() {
            Err(CircuitError::InvalidWireId)
        } else {
            Ok(self.wires[id])
        }
    }

    pub fn set_wire_value(&mut self, id: WireId, value: F) -> Result<(), CircuitError> {
        let prev = self.get_wire_value(id)?;
        if let Some(val) = prev {
            if val != value {
                return Err(CircuitError::WireSetTwice);
            }
        } else {
            self.wires[id] = Some(value);
        }
        Ok(())
    }

    pub fn get_instances(&self) -> &Vec<Box<dyn Gate<F>>> {
        &self.gate_instances
    }

    pub fn generate(&mut self) -> Result<(), CircuitError> {
        let gate_instances = core::mem::take(&mut self.gate_instances);
        for gate in gate_instances.into_iter() {
            gate.generate(self)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum CircuitError {
    InvalidWireId,
    InputNotSet,
    WireSetTwice,
}
