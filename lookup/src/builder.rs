//! Builder for collecting interactions from AIR evaluation.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::Field;

use crate::interaction::{Interaction, MessageBuilder};

/// A builder that collects interactions in insertion order.
#[derive(Debug, Default)]
pub struct InteractionCollector<F: Field> {
    /// Collected interactions (preserves insertion order)
    interactions: Vec<Interaction<F>>,
}

impl<F: Field> InteractionCollector<F> {
    /// Creates a new interaction collector.
    pub const fn new() -> Self {
        Self {
            interactions: vec![],
        }
    }

    /// Consumes the builder and returns the collected interactions.
    pub fn into_interactions(self) -> Vec<Interaction<F>> {
        self.interactions
    }

    /// Returns a reference to the collected interactions without consuming the builder.
    pub fn interactions(&self) -> &[Interaction<F>] {
        &self.interactions
    }
}

impl<F: Field> MessageBuilder<F> for InteractionCollector<F> {
    fn send(&mut self, mut interaction: Interaction<F>) {
        // Make the multiplicity negative for sends
        interaction.multiplicity = -interaction.multiplicity;
        self.interactions.push(interaction);
    }

    fn receive(&mut self, interaction: Interaction<F>) {
        // Multiplicity stays positive for receives
        self.interactions.push(interaction);
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

        let interactions = collector.into_interactions();

        assert_eq!(interactions.len(), 2);

        // Verify that send multiplicity was negated (first interaction)
        if let SymbolicExpression::Constant(m) = interactions[0].multiplicity {
            assert_eq!(m, -F::TWO);
        } else {
            panic!("Expected constant multiplicity");
        }

        // Verify receive multiplicity stayed positive (second interaction)
        if let SymbolicExpression::Constant(m) = interactions[1].multiplicity {
            assert_eq!(m, F::ONE);
        } else {
            panic!("Expected constant multiplicity");
        }
    }

    #[test]
    fn test_multiple_interactions() {
        let mut collector = InteractionCollector::<F>::new();

        // Multiple sends and receives (interleaved)
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

        let interactions = collector.interactions();

        // Should have 10 interactions total (5 sends + 5 receives, interleaved)
        assert_eq!(interactions.len(), 10);

        // Verify they alternate: send (negative), receive (positive), send, receive, ...
        for i in 0..5 {
            // Send should have negative multiplicity
            if let SymbolicExpression::Constant(m) = interactions[i * 2].multiplicity {
                assert_eq!(m, -F::ONE);
            } else {
                panic!("Expected constant multiplicity for send");
            }

            // Receive should have positive multiplicity
            if let SymbolicExpression::Constant(m) = interactions[i * 2 + 1].multiplicity {
                assert_eq!(m, F::ONE);
            } else {
                panic!("Expected constant multiplicity for receive");
            }
        }
    }
}
