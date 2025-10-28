//! Builder for collecting interactions from AIR evaluation.

use alloc::vec;
use alloc::vec::Vec;

use p3_air::{AirBuilder, AirBuilderWithPublicValues, PairBuilder};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Entry, SymbolicExpression, SymbolicVariable};

use crate::interaction::{Interaction, MessageBuilder};

/// A builder that collects interactions in insertion order.
///
/// This builder supports interaction discovery during AIR evaluation.
#[derive(Debug)]
pub struct InteractionCollector<F: Field> {
    /// Preprocessed trace (symbolic variables)
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    /// Main trace (symbolic variables for rows 0 and 1)
    main: RowMajorMatrix<SymbolicVariable<F>>,
    /// Collected interactions (preserves insertion order)
    interactions: Vec<Interaction<F>>,
    /// Public values (empty for interaction collection)
    public_values: Vec<F>,
}

impl<F: Field> InteractionCollector<F> {
    /// Creates a new interaction collector with the specified trace widths.
    ///
    /// # Arguments
    /// * `preprocessed_width` - Width of the preprocessed trace (0 if no preprocessed columns)
    /// * `main_width` - Width of the main trace
    pub fn new(preprocessed_width: usize, main_width: usize) -> Self {
        let preprocessed_width = preprocessed_width.max(1);

        // Create symbolic variables for preprocessed trace (row 0 only)
        let prep_values: Vec<SymbolicVariable<F>> = (0..preprocessed_width)
            .map(|col| SymbolicVariable::new(Entry::Preprocessed { offset: 0 }, col))
            .collect();

        // Create symbolic variables for main trace (rows 0 and 1 for transitions)
        let mut main_values: Vec<SymbolicVariable<F>> = Vec::with_capacity(main_width * 2);
        main_values.extend(
            (0..main_width).map(|col| SymbolicVariable::new(Entry::Main { offset: 0 }, col)),
        );
        main_values.extend(
            (0..main_width).map(|col| SymbolicVariable::new(Entry::Main { offset: 1 }, col)),
        );

        Self {
            preprocessed: RowMajorMatrix::new(prep_values, preprocessed_width),
            main: RowMajorMatrix::new(main_values, main_width),
            interactions: vec![],
            public_values: vec![],
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

impl<F: Field> Default for InteractionCollector<F> {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl<F: Field> AirBuilder for InteractionCollector<F> {
    type F = F;
    type Expr = SymbolicExpression<F>;
    type Var = SymbolicVariable<F>;
    type M = RowMajorMatrix<Self::Var>;

    fn main(&self) -> Self::M {
        self.main.clone()
    }

    fn is_first_row(&self) -> Self::Expr {
        SymbolicExpression::IsFirstRow
    }

    fn is_last_row(&self) -> Self::Expr {
        SymbolicExpression::IsLastRow
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        assert_eq!(
            size, 2,
            "InteractionCollector only supports transition windows of size 2"
        );
        SymbolicExpression::IsTransition
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, _x: I) {
        // No-op for interaction collection
    }
}

impl<F: Field> PairBuilder for InteractionCollector<F> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed.clone()
    }
}

impl<F: Field> AirBuilderWithPublicValues for InteractionCollector<F> {
    type PublicVar = F;

    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
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
        let mut collector = InteractionCollector::<F>::new(0, 1);

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
        let mut collector = InteractionCollector::<F>::new(0, 1);

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
