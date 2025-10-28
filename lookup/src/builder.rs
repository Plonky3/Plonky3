//! Builder for collecting interactions from AIR evaluation.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::Field;

use crate::interaction::{Interaction, MessageBuilder};

/// A builder that collects interactions.
#[derive(Debug, Default)]
pub struct InteractionCollector<F: Field> {
    /// Collected send interactions
    sends: Vec<Interaction<F>>,
    /// Collected receive interactions
    receives: Vec<Interaction<F>>,
}

impl<F: Field> InteractionCollector<F> {
    /// Creates a new interaction collector.
    pub const fn new() -> Self {
        Self {
            sends: vec![],
            receives: vec![],
        }
    }

    /// Consumes the builder and returns the collected interactions.
    ///
    /// # Returns
    ///
    /// A tuple of `(sends, receives)` where each is a vector of interactions.
    pub fn into_interactions(self) -> (Vec<Interaction<F>>, Vec<Interaction<F>>) {
        (self.sends, self.receives)
    }

    /// Returns references to the collected interactions without consuming the builder.
    pub fn interactions(&self) -> (&[Interaction<F>], &[Interaction<F>]) {
        (&self.sends, &self.receives)
    }
}

impl<F: Field> MessageBuilder<F> for InteractionCollector<F> {
    fn send(&mut self, mut interaction: Interaction<F>) {
        // Make the multiplicity negative for sends
        interaction.multiplicity = -interaction.multiplicity;
        self.sends.push(interaction);
    }

    fn receive(&mut self, interaction: Interaction<F>) {
        // Multiplicity stays positive for receives
        self.receives.push(interaction);
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_uni_stark::SymbolicExpression;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_collector_basic() {
        let mut collector = InteractionCollector::<F>::new();

        // Send an interaction
        collector.send(Interaction {
            values: vec![SymbolicExpression::Constant(F::ONE)],
            multiplicity: SymbolicExpression::Constant(F::TWO),
        });

        // Receive an interaction
        collector.receive(Interaction {
            values: vec![SymbolicExpression::Constant(F::new(42))],
            multiplicity: SymbolicExpression::Constant(F::ONE),
        });

        let (sends, receives) = collector.into_interactions();

        assert_eq!(sends.len(), 1);
        assert_eq!(receives.len(), 1);

        // Verify that send multiplicity was negated
        if let SymbolicExpression::Constant(m) = sends[0].multiplicity {
            assert_eq!(m, -F::TWO);
        } else {
            panic!("Expected constant multiplicity");
        }

        // Verify receive multiplicity stayed positive
        if let SymbolicExpression::Constant(m) = receives[0].multiplicity {
            assert_eq!(m, F::ONE);
        } else {
            panic!("Expected constant multiplicity");
        }
    }

    #[test]
    fn test_multiple_interactions() {
        let mut collector = InteractionCollector::<F>::new();

        // Multiple sends and receives
        for i in 0..5 {
            collector.send(Interaction {
                values: vec![SymbolicExpression::Constant(F::new(i))],
                multiplicity: SymbolicExpression::Constant(F::ONE),
            });

            collector.receive(Interaction {
                values: vec![SymbolicExpression::Constant(F::new(i * 10))],
                multiplicity: SymbolicExpression::Constant(F::ONE),
            });
        }

        let (sends, receives) = collector.interactions();

        assert_eq!(sends.len(), 5);
        assert_eq!(receives.len(), 5);
    }
}
